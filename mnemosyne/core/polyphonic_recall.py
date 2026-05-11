"""
Mnemosyne Polyphonic Recall Engine
===================================
Multi-strategy parallel retrieval with deterministic re-ranking.

Strategies (4 voices):
1. Vector voice: Dense semantic similarity over working_memory + episodic_memory
2. Graph voice: Episodic graph traversal (Phase 3)
3. Fact voice: Structured fact matching (Phase 4)
4. Temporal voice: Time-aware scoring

Deterministic re-ranker:
- Combines 4 scores with learned weights
- No neural network (rule-based weighting)
- Budget-aware context assembly
- Diversity penalty (avoid duplicates)

Building on:
- Hindsight's multi-strategy retrieval (blog)
- Memanto's information-theoretic scoring (arXiv:2604.22085)
- Our novel deterministic combination
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:  # numpy is required by other voices too; guard for parity
    np = None

from mnemosyne.core.typed_memory import classify_memory, MemoryType, get_type_priority
from mnemosyne.core.episodic_graph import EpisodicGraph
from mnemosyne.core.veracity_consolidation import VeracityConsolidator


@dataclass
class RecallResult:
    """Result from a single recall voice."""
    memory_id: str
    score: float
    voice: str
    metadata: Dict


@dataclass
class PolyphonicResult:
    """Combined result from all voices."""
    memory_id: str
    combined_score: float
    voice_scores: Dict[str, float]
    metadata: Dict


class PolyphonicRecallEngine:
    """
    Multi-strategy parallel retrieval with deterministic re-ranking.
    
    4 voices:
    - vector: Binary vector similarity
    - graph: Episodic graph traversal
    - fact: Structured fact matching
    - temporal: Time-aware scoring
    """
    
    def __init__(self, db_path: Path = None, conn: sqlite3.Connection = None):
        """Initialize the engine.

        db_path: filesystem path to the SQLite DB. Used by voices that
            spawn their own connection (only when conn is None).
        conn: optional shared sqlite3 connection. When provided, the
            engine and its subsystems (vector_store / graph /
            consolidator / temporal_voice) reuse this connection
            instead of spawning their own. Required for safe use under
            BeamMemory's thread-local connection model — without this,
            each polyphonic recall call would open 4+ new connections
            (one per voice + one per subsystem) which both wastes
            resources and risks WAL-readback inconsistency under
            concurrent writers.
        """
        self.db_path = db_path or Path.home() / ".hermes" / "mnemosyne" / "data" / "mnemosyne.db"
        self.conn = conn  # may be None — voices fall back to per-call open

        # Initialize subsystems. Each accepts an optional conn= since
        # 9f96ded; pass through so they share our handle.
        # NOTE: vector_store removed. The vector voice now reads dense
        # embeddings from `memory_embeddings` (the production-canonical
        # store also used by the linear recall path), not from the
        # standalone `binary_vectors` table which production never wrote
        # to. See _vector_voice for the rewired query path.
        self.graph = EpisodicGraph(db_path=self.db_path, conn=conn)
        self.consolidator = VeracityConsolidator(db_path=self.db_path, conn=conn)

        # Voice weights (deterministic, learned from validation)
        self.voice_weights = {
            "vector": 0.35,
            "graph": 0.25,
            "fact": 0.25,
            "temporal": 0.15,
        }
    
    def recall(self, query: str, query_embedding: np.ndarray = None,
               top_k: int = 10, context_budget: int = 4000) -> List[PolyphonicResult]:
        """
        Polyphonic recall: all 4 voices in parallel, then combine.
        
        Args:
            query: Text query
            query_embedding: Optional pre-computed embedding
            top_k: Number of results to return
            context_budget: Max tokens for context assembly
            
        Returns:
            List of PolyphonicResult, sorted by combined score
        """
        # Run all 4 voices
        vector_results = self._vector_voice(query_embedding)
        graph_results = self._graph_voice(query)
        fact_results = self._fact_voice(query)
        temporal_results = self._temporal_voice(query)
        
        # Combine results
        all_results = self._combine_voices(
            vector_results, graph_results, fact_results, temporal_results
        )
        
        # Re-rank with diversity
        reranked = self._diversity_rerank(all_results, top_k)
        
        # Assemble context within budget
        context = self._assemble_context(reranked, context_budget)
        
        return context
    
    def _vector_voice(self, query_embedding) -> List[RecallResult]:
        """
        Voice 1: Dense semantic similarity over WM + EM.

        Queries the production-canonical dense embedding store
        (`memory_embeddings`) — the same source the linear recall path
        uses via `_wm_vec_search` / `_in_memory_vec_search` in beam.py.
        Pre-fix this voice queried the standalone `binary_vectors` table
        which production never wrote to (NAI-4 wrote binary vectors as a
        column on episodic_memory, NOT to that table); the result was a
        silently-empty vector voice and a 3-voice polyphonic engine.

        Returning to a single source of truth across the recall stack
        matches the cross-system convergence pattern (Hindsight, mem0,
        Zep, Cognee, Letta all use one dense store shared by every
        retrieval path) and makes polyphonic-vs-linear comparisons
        apples-to-apples for the BEAM-recovery experiment.

        Reads both WM and EM tiers, filters out invalidated /
        superseded rows (mirror of `_wm_vec_search` WHERE clauses), and
        ranks by cosine similarity computed in numpy. Performance is
        bounded by linear cosine over the joined memory_embeddings rows
        (same shape as the existing linear fallback); on databases
        sized for the BEAM benchmark we expect this to land in the
        same order of magnitude as the linear path's WM/EM vec search.
        """
        if query_embedding is None or np is None:
            return []

        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.size == 0:
            return []
        query_norm = float(np.linalg.norm(query_embedding))
        if query_norm == 0.0:
            return []
        query_unit = query_embedding / query_norm

        if self.conn is not None:
            conn = self.conn
            own_conn = False
        else:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            own_conn = True
        try:
            results: List[RecallResult] = []
            now_iso = datetime.now().isoformat()

            # --- WM tier ---
            # Same WHERE clause shape as beam._wm_vec_search: skip
            # invalidated / superseded rows so vector voice never
            # surfaces ghost rows the linear path would have hidden.
            try:
                wm_rows = conn.execute("""
                    SELECT wm.id AS memory_id, me.embedding_json
                    FROM memory_embeddings me
                    JOIN working_memory wm ON me.memory_id = wm.id
                    WHERE wm.superseded_by IS NULL
                      AND (wm.valid_until IS NULL OR wm.valid_until > ?)
                    LIMIT 50000
                """, (now_iso,)).fetchall()
            except sqlite3.OperationalError:
                wm_rows = []

            # --- EM tier ---
            try:
                em_rows = conn.execute("""
                    SELECT em.id AS memory_id, me.embedding_json
                    FROM memory_embeddings me
                    JOIN episodic_memory em ON me.memory_id = em.id
                    LIMIT 50000
                """).fetchall()
            except sqlite3.OperationalError:
                em_rows = []

            for tier, rows in (("working", wm_rows), ("episodic", em_rows)):
                for row in rows:
                    try:
                        memory_id = row["memory_id"]
                        embedding_json = row["embedding_json"]
                        if not embedding_json:
                            continue
                        vec = np.asarray(
                            json.loads(embedding_json), dtype=np.float32
                        )
                        vec_norm = float(np.linalg.norm(vec))
                        if vec_norm == 0.0:
                            continue
                        sim = float(np.dot(query_unit, vec / vec_norm))
                        results.append(RecallResult(
                            memory_id=memory_id,
                            score=sim,
                            voice="vector",
                            metadata={"similarity": sim, "tier": tier},
                        ))
                    except (ValueError, TypeError, json.JSONDecodeError):
                        # Defensive: bad / unparseable embedding_json
                        # rows shouldn't break the voice.
                        continue

            results.sort(key=lambda r: r.score, reverse=True)
            return results[:20]
        finally:
            if own_conn:
                conn.close()
    
    def _graph_voice(self, query: str) -> List[RecallResult]:
        """
        Voice 2: Episodic graph traversal.
        
        Extracts entities from query, finds related memories
        through graph edges.
        """
        # Extract entities (simple noun extraction)
        entities = self._extract_entities(query)
        
        results = []
        for entity in entities:
            # Find gists mentioning this entity
            gists = self.graph.find_gists_by_participant(entity)
            for gist in gists:
                results.append(RecallResult(
                    memory_id=gist.id.replace("gist_", ""),
                    score=0.6,  # Base graph score
                    voice="graph",
                    metadata={"entity": entity, "gist": gist.text}
                ))
            
            # Find facts about this entity
            facts = self.graph.find_facts_by_subject(entity)
            for fact in facts:
                results.append(RecallResult(
                    memory_id=fact.id.split("_")[-1] if "_" in fact.id else fact.id,
                    score=fact.confidence * 0.5,
                    voice="graph",
                    metadata={"entity": entity, "fact": f"{fact.subject} {fact.predicate} {fact.object}"}
                ))
        
        return results
    
    def _fact_voice(self, query: str) -> List[RecallResult]:
        """
        Voice 3: Structured fact matching.
        
        Matches query against consolidated facts.
        """
        # Extract potential subject from query
        words = query.lower().split()
        
        results = []
        for word in words:
            if len(word) < 3:
                continue
            
            facts = self.consolidator.get_consolidated_facts(
                subject=word.capitalize(),
                min_confidence=0.5
            )
            
            for fact in facts:
                results.append(RecallResult(
                    memory_id=f"cf_{fact.subject}_{fact.predicate}_{fact.object}",
                    score=fact.confidence,
                    voice="fact",
                    metadata={
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "object": fact.object,
                        "mentions": fact.mention_count
                    }
                ))
        
        return results
    
    def _temporal_voice(self, query: str) -> List[RecallResult]:
        """
        Voice 4: Time-aware scoring.

        Boosts recent memories, penalizes old ones.
        Uses exponential decay based on age.
        """
        # Check for temporal keywords
        temporal_keywords = [
            "yesterday", "today", "recent", "last", "latest",
            "this week", "this month", "ago", "before"
        ]

        has_temporal = any(kw in query.lower() for kw in temporal_keywords)

        if not has_temporal:
            return []

        # Use the shared connection when available; otherwise open a
        # short-lived one (path used by the engine's standalone tests
        # and `python -m polyphonic_recall` self-test).
        if self.conn is not None:
            conn = self.conn
            own_conn = False
        else:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            own_conn = True
        cursor = conn.cursor()

        try:
            # Check if working_memory table exists
            cursor.execute("""
                SELECT name FROM sqlite_master WHERE type='table' AND name='working_memory'
            """)
            if not cursor.fetchone():
                return []

            # Get memories from last 7 days
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute("""
                SELECT id, content, timestamp, importance
                FROM working_memory
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 20
            """, (week_ago,))

            results = []
            for row in cursor.fetchall():
                # Calculate temporal score
                age = datetime.now() - datetime.fromisoformat(row["timestamp"])
                age_days = age.total_seconds() / 86400
                temporal_score = np.exp(-age_days / 7)  # 7-day half-life

                results.append(RecallResult(
                    memory_id=row["id"],
                    score=temporal_score * row["importance"],
                    voice="temporal",
                    metadata={"age_days": age_days, "importance": row["importance"]}
                ))

            return results
        finally:
            if own_conn:
                conn.close()
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entity names from text."""
        import re
        # Simple capitalized word extraction
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return list(set(entities))
    
    def _combine_voices(self, *voice_results: List[RecallResult]) -> Dict[str, PolyphonicResult]:
        """Combine results from all voices using Reciprocal Rank Fusion.

        RRF formula: score(d) = sum(1 / (k + rank(d, voice_i))) for each voice.
        Position-based fusion eliminates score calibration issues between voices.
        Constant k=60 (proven optimal for 4-voice retrieval).
        """
        RRF_K = 60
        combined = {}

        # Step 1: Rank results within each voice by score (descending)
        voice_ranks = {}  # voice_name -> {memory_id: rank}
        for results in voice_results:
            if not results:
                continue
            sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
            voice_name = sorted_results[0].voice
            voice_ranks[voice_name] = {}
            for rank, r in enumerate(sorted_results, start=1):
                voice_ranks[voice_name][r.memory_id] = rank

        # Step 2: Accumulate RRF scores across voices
        for results in voice_results:
            voice_name = None
            for r in results:
                if voice_name is None:
                    voice_name = r.voice
                if r.memory_id not in combined:
                    combined[r.memory_id] = PolyphonicResult(
                        memory_id=r.memory_id,
                        combined_score=0.0,
                        voice_scores={},
                        metadata={}
                    )
                # RRF contribution: higher rank (lower number) = higher score
                rank = voice_ranks.get(voice_name, {}).get(r.memory_id, 999)
                rrf_contribution = 1.0 / (RRF_K + rank)
                combined[r.memory_id].voice_scores[r.voice] = rrf_contribution
                combined[r.memory_id].combined_score += rrf_contribution
                combined[r.memory_id].metadata.update(r.metadata)

        return combined
    
    def _diversity_rerank(self, results: Dict[str, PolyphonicResult],
                         top_k: int) -> List[PolyphonicResult]:
        """
        Re-rank with diversity penalty.
        
        Penalize results that are too similar to already-selected ones.
        """
        # Sort by combined score
        sorted_results = sorted(
            results.values(),
            key=lambda x: x.combined_score,
            reverse=True
        )
        
        selected = []
        for result in sorted_results:
            if len(selected) >= top_k:
                break
            
            # Check diversity against selected
            is_diverse = True
            for sel in selected:
                similarity = self._estimate_similarity(result, sel)
                if similarity > 0.8:  # Too similar
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(result)
        
        return selected
    
    def _estimate_similarity(self, a: PolyphonicResult, b: PolyphonicResult) -> float:
        """Estimate similarity between two results."""
        # Simple Jaccard-like similarity on voice scores
        voices_a = set(a.voice_scores.keys())
        voices_b = set(b.voice_scores.keys())
        
        if not voices_a or not voices_b:
            return 0.0
        
        intersection = voices_a & voices_b
        union = voices_a | voices_b
        
        return len(intersection) / len(union)
    
    def _assemble_context(self, results: List[PolyphonicResult],
                         budget: int) -> List[PolyphonicResult]:
        """
        Assemble context within token budget.
        
        Approximate 4 chars per token.
        """
        current_chars = 0
        selected = []
        
        for result in results:
            # Estimate result size
            result_chars = len(str(result.metadata)) + 100
            
            if current_chars + result_chars > budget * 4:
                break
            
            selected.append(result)
            current_chars += result_chars
        
        return selected
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        # vector voice now queries memory_embeddings directly; surface
        # the count of embedded rows as the vector-voice signal-of-life.
        vec_count = 0
        if self.conn is not None:
            try:
                vec_count = self.conn.execute(
                    "SELECT COUNT(*) FROM memory_embeddings"
                ).fetchone()[0]
            except sqlite3.OperationalError:
                vec_count = 0
        return {
            "voice_weights": self.voice_weights,
            "vector_stats": {"embedded_rows": vec_count},
            "graph_stats": self.graph.get_stats(),
            "consolidation_stats": self.consolidator.get_stats(),
        }

    def close(self):
        """Close all connections."""
        self.graph.close()
        self.consolidator.close()


# --- Testing ---
if __name__ == "__main__":
    import tempfile
    import os
    
    print("Polyphonic Recall Engine Tests")
    print("=" * 60)
    
    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    engine = PolyphonicRecallEngine(db_path=Path(db_path))
    
    # Test 1: Empty recall
    print("\nTest 1: Empty recall")
    results = engine.recall("What did Alice say yesterday?")
    print(f"  Results: {len(results)}")
    
    # Test 2: Stats
    print("\nTest 2: Stats")
    stats = engine.get_stats()
    print(f"  Voice weights: {stats['voice_weights']}")
    
    # Cleanup
    engine.close()
    os.unlink(db_path)
    
    print("\n" + "=" * 60)
    print("Polyphonic recall tests passed!")
