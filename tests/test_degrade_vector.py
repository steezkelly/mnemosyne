"""Regression tests for [C18.b]: degrade_episodic updates content text but
leaves stale dense embeddings. Pre-fix the embedding stored in vec_episodes
or memory_embeddings still represented the ORIGINAL content even after the
content was compressed/truncated, causing dense recall to score against
content that no longer exists in the row.

Two tests:
1. With embeddings provider available, degrade regenerates the embedding
   to match the new compressed content.
2. With embeddings provider unavailable, degrade invalidates (deletes)
   the stale embedding so dense recall doesn't return semantically
   misleading results.
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


def _content_to_vec(text: str, dim: int = 384) -> np.ndarray:
    """Deterministic content-encoding 'embedding'. Different content
    produces different vectors. Two scalars at the front carry length
    and first-char info so an assertion can detect change."""
    v = np.zeros(dim, dtype=np.float32)
    if not text:
        return v
    v[0] = float(len(text))
    v[1] = float(ord(text[0]))
    # Light hash spread so identical-length, identical-first-char strings
    # still produce different vectors (covers truncation that preserves
    # both signals).
    h = hash(text) & 0xFFFF
    v[2] = float(h % 256)
    v[3] = float((h >> 8) % 256)
    return v


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Patch the embeddings module: available() returns True, embed()
    returns content-deterministic vectors, and force the in-memory
    fallback path so we don't need sqlite-vec loaded."""
    from mnemosyne.core import embeddings as emb

    monkeypatch.setattr(emb, "available", lambda: True)
    monkeypatch.setattr(
        emb, "embed",
        lambda texts: np.stack([_content_to_vec(t) for t in texts]),
    )
    # Force the memory_embeddings fallback path; sqlite-vec presence
    # varies across test environments and the bug is identical for
    # both stores.
    monkeypatch.setattr(beam_module, "_vec_available", lambda conn: False)
    return emb


def _read_fallback_embedding(db_path, memory_id):
    """Return the serialized embedding stored in memory_embeddings for
    the given memory_id, or None if missing."""
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT embedding_json FROM memory_embeddings WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _read_binary_vector(db_path, memory_id):
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT binary_vector FROM episodic_memory WHERE id = ?",
            (memory_id,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


class TestDegradeEpisodicVectorRefresh:

    def test_tier_2_to_tier_3_regenerates_embedding(self, temp_db, fake_embeddings):
        """When tier 2→3 truncation changes content, the embedding stored
        in memory_embeddings must update to match the new content."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)

        # Long original content that will be truncated by tier 2→3 (TIER3_MAX_CHARS=300)
        original = ("ORIGINAL_DETAILED_CONTEXT " * 30).strip()
        assert len(original) > beam_module.TIER3_MAX_CHARS

        memory_id = beam.consolidate_to_episodic(
            summary=original,
            source_wm_ids=["fake-wm"],
            importance=0.6,
        )

        original_embedding = _read_fallback_embedding(temp_db, memory_id)
        assert original_embedding is not None, (
            "memory_embeddings should contain a row after consolidate_to_episodic"
        )

        # Backdate to make the row eligible for tier 2→3 and set tier=2 so it
        # hits the truncation path (skips the LLM-summarization tier 1→2 path
        # which is a no-op when local_llm is unavailable).
        old_ts = (datetime.now() - timedelta(days=beam_module.TIER3_DAYS + 1)).isoformat()
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "UPDATE episodic_memory SET tier = 2, created_at = ? WHERE id = ?",
            (old_ts, memory_id),
        )
        conn.commit()
        conn.close()

        result = beam.degrade_episodic(dry_run=False)
        assert result["tier2_to_tier3"] == 1, (
            f"Expected one tier 2→3 transition, got {result}"
        )

        conn = sqlite3.connect(str(temp_db))
        new_content = conn.execute(
            "SELECT content FROM episodic_memory WHERE id = ?", (memory_id,)
        ).fetchone()[0]
        conn.close()
        assert new_content != original, "tier 2→3 should have truncated the content"

        post_embedding = _read_fallback_embedding(temp_db, memory_id)
        assert post_embedding is not None, (
            "memory_embeddings row missing after degrade; expected regenerated, "
            "not deleted, when the embeddings provider is available"
        )
        assert post_embedding != original_embedding, (
            "memory_embeddings still holds the pre-degradation embedding — "
            "dense recall would score against original content while displaying "
            "truncated content. C18.b regeneration did not run."
        )

    def test_tier_2_to_tier_3_regenerates_binary_vector(self, temp_db, fake_embeddings):
        """The binary_vector column on episodic_memory must also update
        to match the new content."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)

        original = ("ORIGINAL_DETAILED_CONTEXT " * 30).strip()
        memory_id = beam.consolidate_to_episodic(
            summary=original,
            source_wm_ids=["fake-wm"],
            importance=0.6,
        )

        if beam_module._mib is None:
            pytest.skip("binary vectorization not available in this build")

        original_bv = _read_binary_vector(temp_db, memory_id)
        assert original_bv is not None

        old_ts = (datetime.now() - timedelta(days=beam_module.TIER3_DAYS + 1)).isoformat()
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "UPDATE episodic_memory SET tier = 2, created_at = ? WHERE id = ?",
            (old_ts, memory_id),
        )
        conn.commit()
        conn.close()

        beam.degrade_episodic(dry_run=False)

        post_bv = _read_binary_vector(temp_db, memory_id)
        assert post_bv is not None, (
            "binary_vector should be present (regenerated, not nulled) when "
            "the embedding provider is available"
        )
        assert post_bv != original_bv, (
            "binary_vector still holds pre-degradation bytes — same C18.b drift"
        )

    def test_tier_2_to_tier_3_invalidates_when_provider_unavailable(
        self, temp_db, monkeypatch
    ):
        """If embeddings provider is unavailable at degrade time, the stale
        embedding rows must be invalidated so dense recall can't return
        semantically misleading hits."""
        from mnemosyne.core import embeddings as emb

        # Phase 1: provider available — seed.
        monkeypatch.setattr(emb, "available", lambda: True)
        monkeypatch.setattr(
            emb, "embed",
            lambda texts: np.stack([_content_to_vec(t) for t in texts]),
        )
        monkeypatch.setattr(beam_module, "_vec_available", lambda conn: False)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        original = ("ORIGINAL_DETAILED_CONTEXT " * 30).strip()
        memory_id = beam.consolidate_to_episodic(
            summary=original,
            source_wm_ids=["fake-wm"],
            importance=0.6,
        )
        assert _read_fallback_embedding(temp_db, memory_id) is not None

        # Phase 2: provider goes unavailable BEFORE degrade.
        monkeypatch.setattr(emb, "available", lambda: False)

        old_ts = (datetime.now() - timedelta(days=beam_module.TIER3_DAYS + 1)).isoformat()
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "UPDATE episodic_memory SET tier = 2, created_at = ? WHERE id = ?",
            (old_ts, memory_id),
        )
        conn.commit()
        conn.close()

        beam.degrade_episodic(dry_run=False)

        post_embedding = _read_fallback_embedding(temp_db, memory_id)
        assert post_embedding is None, (
            "Stale memory_embeddings row remained after degrade with no embeddings "
            "provider. Should have been deleted to avoid ranking against content "
            "that no longer matches the row's text."
        )

        post_bv = _read_binary_vector(temp_db, memory_id)
        if beam_module._mib is not None:
            assert post_bv is None, (
                "binary_vector should be NULLed when the embedding provider is "
                "unavailable at degrade time"
            )
