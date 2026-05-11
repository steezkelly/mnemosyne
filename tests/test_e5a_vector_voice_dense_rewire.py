"""
Regression tests for E5.a — vector voice dense rewire
=====================================================

Pre-fix: ``PolyphonicRecallEngine._vector_voice`` queried the standalone
``binary_vectors`` table that production never wrote to (NAI-4 stored
binary vectors as a column on ``episodic_memory`` instead). The vector
voice silently returned ``[]`` on every call, so polyphonic recall was
effectively 3-voice (graph + fact + temporal) in production.

Post-fix: the vector voice queries ``memory_embeddings`` directly —
the same dense embedding store the linear recall path uses via
``_wm_vec_search`` / ``_in_memory_vec_search``. Single source of
truth, both WM and EM tiers covered, no schema migration.

These tests pin:
  - vector voice returns candidates when memory_embeddings is populated
  - results are ranked by cosine similarity (closest first)
  - WM and EM tiers are both covered
  - invalidated / superseded WM rows are excluded (parity with linear)
  - vector voice returns [] when query_embedding is None (preserves the
    pre-fix fallback contract — fastembed-unavailable callers don't get
    crashes)
  - vector voice returns [] when memory_embeddings is empty (no false
    positives from the now-removed standalone table)
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    return tmp_path / "mnemosyne_e5a.db"


def _seed_embedding(conn, memory_id: str, vec: np.ndarray) -> None:
    """Insert a row into memory_embeddings for a memory_id."""
    conn.execute(
        "INSERT OR REPLACE INTO memory_embeddings "
        "(memory_id, embedding_json, model) VALUES (?, ?, ?)",
        (memory_id, json.dumps(vec.astype(np.float32).tolist()), "test-model"),
    )
    conn.commit()


def _unit_vec(seed: int, dim: int = 384) -> np.ndarray:
    """Deterministic unit vector for a given seed."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Core rewire: vector voice reads memory_embeddings, not binary_vectors
# ---------------------------------------------------------------------------


def test_vector_voice_returns_candidates_from_memory_embeddings(temp_db):
    """Vector voice reads dense vectors from memory_embeddings.

    The pre-fix behavior would return [] because the standalone
    binary_vectors table is never populated. Post-fix, the voice
    ranks candidates from memory_embeddings."""
    beam = BeamMemory(session_id="e5a", db_path=temp_db)
    # Seed two EM rows with embeddings.
    beam.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, importance) "
        "VALUES ('em-a', 'alpha content', 'test', datetime('now'), 0.5)"
    )
    beam.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, importance) "
        "VALUES ('em-b', 'bravo content', 'test', datetime('now'), 0.5)"
    )
    vec_a = _unit_vec(seed=1)
    vec_b = _unit_vec(seed=2)
    _seed_embedding(beam.conn, "em-a", vec_a)
    _seed_embedding(beam.conn, "em-b", vec_b)

    engine = PolyphonicRecallEngine(db_path=temp_db, conn=beam.conn)
    # Query embedding is exactly em-a → similarity ~1.0; em-b is unrelated.
    results = engine._vector_voice(vec_a)

    assert results, "vector voice returned empty after rewire"
    ids = [r.memory_id for r in results]
    assert "em-a" in ids, "expected EM hit em-a missing from vector voice"
    # em-a should rank above em-b (higher similarity).
    em_a_score = next(r.score for r in results if r.memory_id == "em-a")
    em_b_score = next((r.score for r in results if r.memory_id == "em-b"), -1)
    assert em_a_score > em_b_score, (
        f"em-a ({em_a_score}) did not outrank em-b ({em_b_score})"
    )
    # Voice attribution is correct.
    assert all(r.voice == "vector" for r in results)


def test_vector_voice_covers_both_wm_and_em_tiers(temp_db):
    """WM AND EM rows should both surface — single source of truth."""
    beam = BeamMemory(session_id="e5a-wmem", db_path=temp_db)
    beam.conn.execute(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance) "
        "VALUES ('wm-1', 'working row', 'test', datetime('now'), 'e5a-wmem', 0.5)"
    )
    beam.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, importance) "
        "VALUES ('em-1', 'episodic row', 'test', datetime('now'), 0.5)"
    )
    target_vec = _unit_vec(seed=42)
    _seed_embedding(beam.conn, "wm-1", target_vec)
    _seed_embedding(beam.conn, "em-1", target_vec)

    engine = PolyphonicRecallEngine(db_path=temp_db, conn=beam.conn)
    results = engine._vector_voice(target_vec)

    tiers = {r.metadata.get("tier") for r in results}
    ids = {r.memory_id for r in results}
    assert "working" in tiers, "WM tier missing from vector voice results"
    assert "episodic" in tiers, "EM tier missing from vector voice results"
    assert "wm-1" in ids
    assert "em-1" in ids


def test_vector_voice_skips_superseded_wm_rows(temp_db):
    """WM rows with superseded_by set must NOT surface — parity with
    the linear path's _wm_vec_search WHERE clause."""
    beam = BeamMemory(session_id="e5a-sup", db_path=temp_db)
    beam.conn.execute(
        "INSERT INTO working_memory (id, content, source, timestamp, "
        "session_id, importance, superseded_by) "
        "VALUES ('wm-old', 'stale', 'test', datetime('now'), 'e5a-sup', 0.5, 'wm-new')"
    )
    beam.conn.execute(
        "INSERT INTO working_memory (id, content, source, timestamp, "
        "session_id, importance) "
        "VALUES ('wm-new', 'fresh', 'test', datetime('now'), 'e5a-sup', 0.5)"
    )
    vec = _unit_vec(seed=7)
    _seed_embedding(beam.conn, "wm-old", vec)
    _seed_embedding(beam.conn, "wm-new", vec)

    engine = PolyphonicRecallEngine(db_path=temp_db, conn=beam.conn)
    results = engine._vector_voice(vec)
    ids = {r.memory_id for r in results}
    assert "wm-old" not in ids, "superseded WM row surfaced by vector voice"
    assert "wm-new" in ids, "non-superseded WM row missing from vector voice"


def test_vector_voice_skips_expired_wm_rows(temp_db):
    """WM rows with valid_until in the past must NOT surface."""
    beam = BeamMemory(session_id="e5a-exp", db_path=temp_db)
    beam.conn.execute(
        "INSERT INTO working_memory (id, content, source, timestamp, "
        "session_id, importance, valid_until) "
        "VALUES ('wm-exp', 'old', 'test', datetime('now'), 'e5a-exp', 0.5, "
        "datetime('now', '-1 day'))"
    )
    beam.conn.execute(
        "INSERT INTO working_memory (id, content, source, timestamp, "
        "session_id, importance) "
        "VALUES ('wm-live', 'fresh', 'test', datetime('now'), 'e5a-exp', 0.5)"
    )
    vec = _unit_vec(seed=11)
    _seed_embedding(beam.conn, "wm-exp", vec)
    _seed_embedding(beam.conn, "wm-live", vec)

    engine = PolyphonicRecallEngine(db_path=temp_db, conn=beam.conn)
    results = engine._vector_voice(vec)
    ids = {r.memory_id for r in results}
    assert "wm-exp" not in ids, "expired WM row surfaced by vector voice"
    assert "wm-live" in ids


# ---------------------------------------------------------------------------
# Contract: defensive fallbacks
# ---------------------------------------------------------------------------


def test_vector_voice_returns_empty_for_none_query_embedding(temp_db):
    """fastembed-unavailable callers pass query_embedding=None — voice
    must return [] without crashing. Preserves pre-fix behavior."""
    beam = BeamMemory(session_id="e5a-none", db_path=temp_db)
    engine = PolyphonicRecallEngine(db_path=temp_db, conn=beam.conn)
    assert engine._vector_voice(None) == []


def test_vector_voice_returns_empty_when_no_embeddings_stored(temp_db):
    """Fresh DB with no memory_embeddings rows → []. Critical regression
    guard: ensures we didn't accidentally re-create the silent fallback
    to the standalone binary_vectors table."""
    beam = BeamMemory(session_id="e5a-fresh", db_path=temp_db)
    engine = PolyphonicRecallEngine(db_path=temp_db, conn=beam.conn)
    vec = _unit_vec(seed=0)
    assert engine._vector_voice(vec) == []


def test_vector_voice_tolerates_bad_embedding_json(temp_db):
    """Malformed embedding_json should be skipped, not crash the voice."""
    beam = BeamMemory(session_id="e5a-bad", db_path=temp_db)
    beam.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, importance) "
        "VALUES ('em-bad', 'x', 'test', datetime('now'), 0.5)"
    )
    beam.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, importance) "
        "VALUES ('em-good', 'y', 'test', datetime('now'), 0.5)"
    )
    # Bad row: invalid JSON
    beam.conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json, model) "
        "VALUES ('em-bad', 'not-json', 'test-model')"
    )
    # Good row
    good_vec = _unit_vec(seed=99)
    _seed_embedding(beam.conn, "em-good", good_vec)

    engine = PolyphonicRecallEngine(db_path=temp_db, conn=beam.conn)
    results = engine._vector_voice(good_vec)
    ids = {r.memory_id for r in results}
    assert "em-good" in ids
    assert "em-bad" not in ids


# ---------------------------------------------------------------------------
# End-to-end: polyphonic recall now has all 4 voices contributing
# ---------------------------------------------------------------------------


def test_polyphonic_recall_includes_vector_voice_in_rrf(temp_db):
    """Full polyphonic recall path: with memory_embeddings populated,
    the combined result includes a vector voice score for at least one
    memory id. This is the headline contract: post-fix the engine is
    genuinely 4-voice in production-shaped queries."""
    beam = BeamMemory(session_id="e5a-rrf", db_path=temp_db)
    beam.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, importance) "
        "VALUES ('em-x', 'target content', 'test', datetime('now'), 0.5)"
    )
    target_vec = _unit_vec(seed=123)
    _seed_embedding(beam.conn, "em-x", target_vec)

    engine = PolyphonicRecallEngine(db_path=temp_db, conn=beam.conn)
    results = engine.recall(
        query="target content",
        query_embedding=target_vec,
        top_k=10,
    )

    # At least one result should have a non-empty voice_scores dict
    # containing "vector". This is the inverse of the pre-fix regression
    # where vector_scores was always empty.
    has_vector_signal = any(
        "vector" in r.voice_scores for r in results
    )
    assert has_vector_signal, (
        "no result carries a vector voice score after rewire — "
        "vector voice still silent in the combine step"
    )


def test_polyphonic_vector_score_outranks_unrelated_query(temp_db):
    """Two rows: one semantically identical to query, one orthogonal.
    Polyphonic must rank the identical row above. Pre-fix this would
    have failed: with vector voice silent, only FTS/temporal/graph
    contribute, and "target content" only matches FTS-equally."""
    beam = BeamMemory(session_id="e5a-rank", db_path=temp_db)
    beam.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, importance) "
        "VALUES ('em-target', 'common phrase A', 'test', datetime('now'), 0.5)"
    )
    beam.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, importance) "
        "VALUES ('em-other', 'common phrase B', 'test', datetime('now'), 0.5)"
    )
    target_vec = _unit_vec(seed=200)
    orthogonal = np.zeros(384, dtype=np.float32)
    orthogonal[0] = 1.0  # Mostly orthogonal to seed=200's random vec
    orthogonal = orthogonal / np.linalg.norm(orthogonal)
    _seed_embedding(beam.conn, "em-target", target_vec)
    _seed_embedding(beam.conn, "em-other", orthogonal)

    engine = PolyphonicRecallEngine(db_path=temp_db, conn=beam.conn)
    results = engine.recall(
        query="common phrase",
        query_embedding=target_vec,
        top_k=10,
    )

    # em-target should appear at or above em-other in ranking.
    ids = [r.memory_id for r in results]
    if "em-target" in ids and "em-other" in ids:
        assert ids.index("em-target") <= ids.index("em-other"), (
            f"vector-similar row did not outrank orthogonal: {ids}"
        )
    else:
        # Both might not survive diversity rerank; minimal contract:
        # em-target must be present.
        assert "em-target" in ids, (
            f"vector-similar row absent from results: {ids}"
        )


# ---------------------------------------------------------------------------
# Plumbing: engine no longer requires BinaryVectorStore at all
# ---------------------------------------------------------------------------


def test_engine_does_not_construct_binary_vector_store(temp_db):
    """The BinaryVectorStore class still exists for backward compat
    with anyone using it standalone, but the engine should not
    construct one. Verifies the dead-code path is gone."""
    beam = BeamMemory(session_id="e5a-noref", db_path=temp_db)
    engine = PolyphonicRecallEngine(db_path=temp_db, conn=beam.conn)
    assert not hasattr(engine, "vector_store"), (
        "engine still constructs BinaryVectorStore — rewire incomplete"
    )


def test_engine_get_stats_reports_embedded_row_count(temp_db):
    """get_stats() previously returned BinaryVectorStore.get_stats();
    post-rewire it should report the memory_embeddings count (the new
    vector-voice signal-of-life)."""
    beam = BeamMemory(session_id="e5a-stats", db_path=temp_db)
    beam.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, importance) "
        "VALUES ('em-s', 's', 'test', datetime('now'), 0.5)"
    )
    _seed_embedding(beam.conn, "em-s", _unit_vec(seed=5))

    engine = PolyphonicRecallEngine(db_path=temp_db, conn=beam.conn)
    stats = engine.get_stats()
    assert "vector_stats" in stats
    assert stats["vector_stats"].get("embedded_rows") == 1
