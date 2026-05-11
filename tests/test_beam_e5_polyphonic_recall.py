"""Regression tests for E5 — wire PolyphonicRecallEngine under feature flag.

Pre-E5: `PolyphonicRecallEngine` (mnemosyne/core/polyphonic_recall.py)
existed as a complete 4-voice + RRF + diversity-rerank + budget-aware
context-assembly implementation, but it was dead code — no production
caller imported it. Commit 9f96ded's "polyphonic recall" was actually
inline graph_bonus + fact_bonus added to BeamMemory's linear scorer
(beam.py:2101-2156), not the engine.

Post-E5:
  - `MNEMOSYNE_POLYPHONIC_RECALL=1` activates the polyphonic engine
    inside `BeamMemory.recall()`
  - Default (flag unset or "0"): existing linear scorer runs unchanged —
    production behavior preserved
  - Flag ON: engine produces ranked candidates via RRF fusion across 4
    voices (vector + graph + fact + temporal), diversity-reranks them,
    and assembles context within budget. The inline graph_bonus and
    fact_bonus terms in the linear scorer are bypassed (engine handles
    those itself).
  - Per-result `voice_scores` field carries provenance — operators can
    see WHICH voices contributed to a given ranking.
  - Engine reuses BeamMemory's shared sqlite connection rather than
    spawning 4+ new connections per recall call.

This unblocks the experiment's "strongest available recall layer" —
all arms run with flag=ON to test against the polyphonic engine
instead of the inline bonus shortcut.
"""

import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def disable_llm(monkeypatch):
    """Force deterministic non-LLM paths."""
    monkeypatch.setattr(
        "mnemosyne.core.local_llm.llm_available", lambda: False
    )


def _seed_recallable_content(beam, items):
    """Seed working_memory + episodic_memory with content that recall
    can find. items is a list of (content, source, importance) tuples."""
    for content, source, importance in items:
        beam.remember(content, source=source, importance=importance)


class TestE5FeatureFlag:

    def test_flag_off_uses_linear_scorer(self, temp_db, monkeypatch, disable_llm):
        """[E5] When MNEMOSYNE_POLYPHONIC_RECALL is unset or '0', recall
        runs the existing linear scorer. Production behavior unchanged.

        The signal that we're on the linear path: result dicts do NOT
        carry a `voice_scores` field (that's only populated by the
        polyphonic engine)."""
        monkeypatch.delenv("MNEMOSYNE_POLYPHONIC_RECALL", raising=False)

        beam = BeamMemory(session_id="e5-off", db_path=temp_db)
        _seed_recallable_content(beam, [
            ("Alice mentioned the deploy timeline", "conversation", 0.7),
            ("Bob owns the auth refactor", "fact", 0.8),
        ])

        results = beam.recall("deploy", top_k=10)
        assert results, "recall returned 0 — sanity check"
        for r in results:
            assert "voice_scores" not in r, (
                f"linear scorer leaked voice_scores into result: {r}"
            )

    def test_flag_off_explicit_zero(self, temp_db, monkeypatch, disable_llm):
        """Setting the env var to '0' is the explicit way to opt out;
        must behave identically to unset."""
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")

        beam = BeamMemory(session_id="e5-off-zero", db_path=temp_db)
        _seed_recallable_content(beam, [
            ("explicit-zero content here", "test", 0.5),
        ])

        results = beam.recall("explicit-zero", top_k=10)
        for r in results:
            assert "voice_scores" not in r

    def test_flag_on_uses_polyphonic_engine(
        self, temp_db, monkeypatch, disable_llm
    ):
        """[E5] Flag ON: results carry voice_scores provenance — the
        signal that the engine ran instead of the linear scorer.
        Per-signal observability is a hard requirement (the
        OpenViking-style 'show me which voice contributed').
        """
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")

        beam = BeamMemory(session_id="e5-on", db_path=temp_db)
        _seed_recallable_content(beam, [
            ("Alice deployed the auth service last Tuesday", "convo", 0.7),
            ("Bob filed a bug about the auth refactor", "convo", 0.7),
            ("Carol approved Alice's deploy plan", "convo", 0.8),
        ])

        results = beam.recall("Alice deploy auth", top_k=10)
        assert results, "polyphonic recall returned 0 — engine wired wrong"
        # At least one result must carry voice_scores; an empty dict is
        # not enough (would mean RRF combined nothing). The engine's
        # RRF accumulates contributions from each contributing voice.
        any_with_voices = any(
            r.get("voice_scores") for r in results
        )
        assert any_with_voices, (
            f"no result carries voice_scores; engine didn't run or its "
            f"output wasn't mapped back. Got: {results}"
        )

    def test_flag_on_then_off_swaps_paths(
        self, temp_db, monkeypatch, disable_llm
    ):
        """The flag is read PER CALL, not at __init__ time — operators
        can toggle the engine without rebuilding BeamMemory. Critical
        for A/B experiments inside the same process.

        Uses a query with a capitalized entity ('Alice') so the graph
        voice has something to bite on without requiring embeddings."""
        beam = BeamMemory(session_id="e5-toggle", db_path=temp_db)
        _seed_recallable_content(beam, [
            ("Alice asked about the toggle behavior", "convo", 0.6),
            ("Alice followed up with another comment", "convo", 0.6),
        ])

        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        on_results = beam.recall("Alice toggle", top_k=10)

        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
        off_results = beam.recall("Alice toggle", top_k=10)

        # ON path produces voice_scores; OFF path doesn't.
        on_has_voices = any(r.get("voice_scores") for r in on_results)
        off_has_voices = any("voice_scores" in r for r in off_results)
        assert on_has_voices, (
            f"flag=ON didn't engage the engine. on_results={on_results}"
        )
        assert not off_has_voices, "flag=OFF still ran the engine"


class TestE5EnginePlumbing:

    def test_engine_accepts_shared_connection(self, temp_db):
        """[E5 connection reuse] PolyphonicRecallEngine.__init__ must
        accept conn= so BeamMemory can share its thread-local
        connection. Without this each recall call would spawn 4+ new
        SQLite connections (one per voice + one for the engine itself)
        — wasteful under load and inconsistent with the post-9f96ded
        EpisodicGraph / VeracityConsolidator / BinaryVectorStore
        pattern."""
        from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine

        # Bring up the schema via BeamMemory so the engine has tables to query.
        beam = BeamMemory(session_id="e5-conn", db_path=temp_db)
        shared_conn = beam.conn

        # The constructor must accept conn= without raising.
        engine = PolyphonicRecallEngine(db_path=temp_db, conn=shared_conn)

        # Verify the engine's subsystems also use the shared connection
        # (this is the whole point of accepting it).
        assert engine.vector_store.conn is shared_conn, (
            "vector_store is not using the shared connection"
        )
        assert engine.graph.conn is shared_conn, (
            "graph is not using the shared connection"
        )
        assert engine.consolidator.conn is shared_conn, (
            "consolidator is not using the shared connection"
        )

    def test_engine_recall_handles_empty_corpus(
        self, temp_db, monkeypatch, disable_llm
    ):
        """Flag ON against an empty DB: graceful empty return, no crash."""
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(session_id="e5-empty", db_path=temp_db)

        results = beam.recall("anything", top_k=10)
        # Empty list is acceptable; the contract is "doesn't crash" not
        # "returns hits when there are no rows."
        assert isinstance(results, list)

    def test_engine_recall_handles_no_embeddings(
        self, temp_db, monkeypatch, disable_llm
    ):
        """Flag ON when embeddings are unavailable: vector voice
        returns empty, but graph/fact/temporal voices can still
        contribute. No crash."""
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        # Disable embeddings by monkeypatching available() to False.
        monkeypatch.setattr(
            "mnemosyne.core.embeddings.available", lambda: False
        )

        beam = BeamMemory(session_id="e5-noemb", db_path=temp_db)
        _seed_recallable_content(beam, [
            ("Alice did things yesterday", "convo", 0.6),
        ])

        # Use a temporal-keyword query so the temporal voice can
        # contribute even without entities matching the graph voice.
        results = beam.recall("yesterday Alice", top_k=10)
        # No crash. Results may or may not be non-empty depending on
        # which voices found matches.
        assert isinstance(results, list)


class TestE5ResultShape:

    def test_polyphonic_results_have_combined_score(
        self, temp_db, monkeypatch, disable_llm
    ):
        """Flag ON: every result dict carries the RRF combined_score
        in the `score` field (existing shape) so downstream
        consumers don't need a special case."""
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(session_id="e5-shape", db_path=temp_db)
        _seed_recallable_content(beam, [
            ("Carol approved the launch", "convo", 0.8),
            ("Carol pushed back on the date", "convo", 0.7),
        ])

        results = beam.recall("Carol", top_k=10)
        for r in results:
            assert "score" in r, f"missing score field: {r}"
            assert isinstance(r["score"], (int, float))
            assert r["score"] > 0, f"non-positive score: {r}"

    def test_polyphonic_results_have_content(
        self, temp_db, monkeypatch, disable_llm
    ):
        """Flag ON: result dicts include the actual memory content,
        not just an id. Downstream consumers compose prompts from
        `content`, so the engine path must fetch and map it back from
        working_memory / episodic_memory."""
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(session_id="e5-content", db_path=temp_db)
        unique_token = "e5contentmarkerxyz"
        _seed_recallable_content(beam, [
            (f"row carrying {unique_token} content", "convo", 0.7),
        ])

        results = beam.recall(unique_token, top_k=10)
        # The engine may or may not surface the row depending on
        # whether any of its voices match (FTS isn't part of the
        # engine; entity extraction is the graph voice's only input).
        # If we get hits, they MUST have content.
        for r in results:
            assert r.get("content"), (
                f"result has no content; engine didn't map row back: {r}"
            )
