"""Regression tests for E4 — remember_batch veracity threading.

Pre-E4: BeamMemory.remember_batch() didn't accept a veracity param. The
INSERT omitted the column, so every batch-ingested row defaulted to
'unknown'. The recall scorer's veracity multiplier (VERACITY_WEIGHTS in
veracity_consolidation.py) then collapsed to a constant 0.8 across the
entire batch corpus — globally scale-factored rather than rank-signal.

Post-E4:
  - remember_batch accepts a method-level `veracity` kwarg as the
    per-batch default
  - each item dict may carry its own `veracity` key to override
  - non-canonical labels are clamped to 'unknown' with a warning,
    matching the C12.b trust-boundary pattern in hermes_memory_provider
  - the column value is what recall reads, so scoring varies measurably
    between 'stated' and 'unknown' rows in the same DB

This blocks experiment Arms A and C of the BEAM-recovery experiment —
without per-row veracity, the recall scorer cannot differentiate
confident from unconfident facts.
"""

import logging
import sqlite3
import tempfile
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.veracity_consolidation import (
    VERACITY_ALLOWED,
    VERACITY_WEIGHTS,
    clamp_veracity,
)


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


def _veracity_for(db_path, memory_id):
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT veracity FROM working_memory WHERE id = ?",
            (memory_id,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


class TestClampVeracity:
    """The clamp helper is the trust-boundary primitive. Test it
    directly so future call sites can use it confidently."""

    def test_clamp_accepts_canonical_values(self):
        for label in VERACITY_ALLOWED:
            assert clamp_veracity(label) == label

    def test_clamp_normalizes_case_and_whitespace(self):
        assert clamp_veracity("STATED") == "stated"
        assert clamp_veracity("  Inferred ") == "inferred"
        assert clamp_veracity("Tool\n") == "tool"

    def test_clamp_unknown_label_returns_unknown(self):
        assert clamp_veracity("random_garbage") == "unknown"
        assert clamp_veracity("certain") == "unknown"  # not in the set
        assert clamp_veracity("state") == "unknown"    # truncated typo

    def test_clamp_none_returns_unknown(self):
        assert clamp_veracity(None) == "unknown"

    def test_clamp_empty_string_returns_unknown(self):
        assert clamp_veracity("") == "unknown"
        assert clamp_veracity("   ") == "unknown"

    def test_clamp_warns_on_unknown_label(self, caplog):
        with caplog.at_level(logging.WARNING):
            clamp_veracity("random_garbage")
        assert any(
            "veracity" in record.message.lower()
            and "random_garbage" in record.message
            for record in caplog.records
        ), f"no warning logged for unknown label: {caplog.records}"

    def test_clamp_does_not_warn_on_canonical_label(self, caplog):
        with caplog.at_level(logging.WARNING):
            for label in VERACITY_ALLOWED:
                clamp_veracity(label)
        assert not caplog.records, (
            f"clamp warned on canonical labels: {caplog.records}"
        )

    def test_veracity_allowed_matches_weights(self):
        """The allowlist and the weights table must stay in sync —
        a label without a weight collapses scoring to the .get default."""
        assert VERACITY_ALLOWED == set(VERACITY_WEIGHTS.keys())


class TestRememberBatchVeracity:

    def test_remember_batch_default_unknown(self, temp_db):
        """No veracity supplied anywhere → row's column is 'unknown'."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        ids = beam.remember_batch([
            {"content": "no-veracity item", "source": "test"},
        ])
        assert _veracity_for(temp_db, ids[0]) == "unknown"

    def test_remember_batch_method_default(self, temp_db):
        """Method-level veracity kwarg applies to every item that
        doesn't supply its own."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        ids = beam.remember_batch(
            [
                {"content": "item a", "source": "test"},
                {"content": "item b", "source": "test"},
            ],
            veracity="stated",
        )
        for mid in ids:
            assert _veracity_for(temp_db, mid) == "stated"

    def test_remember_batch_per_item_override(self, temp_db):
        """Per-item veracity overrides the method-level default."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        ids = beam.remember_batch(
            [
                {"content": "user said it", "source": "user", "veracity": "stated"},
                {"content": "llm inferred it", "source": "llm", "veracity": "inferred"},
                {"content": "tool reported it", "source": "tool", "veracity": "tool"},
                {"content": "no override here", "source": "test"},
            ],
            veracity="imported",  # method-level default for the no-override item
        )
        assert _veracity_for(temp_db, ids[0]) == "stated"
        assert _veracity_for(temp_db, ids[1]) == "inferred"
        assert _veracity_for(temp_db, ids[2]) == "tool"
        assert _veracity_for(temp_db, ids[3]) == "imported"

    def test_remember_batch_clamps_unknown_labels(self, temp_db, caplog):
        """Non-canonical labels get clamped to 'unknown' with a warning,
        mirroring C12.b at the provider trust boundary."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        with caplog.at_level(logging.WARNING):
            ids = beam.remember_batch([
                {"content": "STATED caps", "source": "test", "veracity": "STATED"},
                {"content": "junk label", "source": "test", "veracity": "random_garbage"},
                {"content": "typo", "source": "test", "veracity": "state"},
            ])

        # STATED normalizes to 'stated' (case-insensitive match).
        assert _veracity_for(temp_db, ids[0]) == "stated"
        # The other two clamp to 'unknown' and emit a warning.
        assert _veracity_for(temp_db, ids[1]) == "unknown"
        assert _veracity_for(temp_db, ids[2]) == "unknown"

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        # Two clamped labels → at least two warnings; one per non-canonical.
        clamp_warns = [
            w for w in warnings
            if "veracity" in w.message.lower()
            and ("random_garbage" in w.message or "state" in w.message.lower())
        ]
        assert len(clamp_warns) >= 2, (
            f"expected ≥2 clamp warnings, got {len(clamp_warns)}: {warnings}"
        )

    def test_remember_batch_clamps_method_default_too(self, temp_db, caplog):
        """The method-level veracity kwarg is also clamped — protects
        callers that pass an unvalidated string from the LLM layer."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        with caplog.at_level(logging.WARNING):
            ids = beam.remember_batch(
                [{"content": "uses bad default", "source": "test"}],
                veracity="totally_invalid_label",
            )
        assert _veracity_for(temp_db, ids[0]) == "unknown"
        assert any(
            "veracity" in r.message.lower() and "totally_invalid_label" in r.message
            for r in caplog.records
        )

    def test_remember_batch_recall_scoring_varies(self, temp_db):
        """[E4 experiment unblock] After per-row veracity is populated,
        recall scoring varies measurably between 'stated' and 'unknown'
        rows on the same content match. Pre-E4 this multiplier collapsed
        to a constant 0.8 because every batch row defaulted to 'unknown'.

        Uses two batches with shared rare token so FTS surfaces both
        rows; veracity is the only difference, so any score delta is
        attributable to the multiplier."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        rare_token = "quetzzzlcoatl"  # unlikely to collide with FTS noise
        beam.remember_batch(
            [{"content": f"{rare_token} stated-content", "source": "test"}],
            veracity="stated",
        )
        beam.remember_batch(
            [{"content": f"{rare_token} unknown-content", "source": "test"}],
            veracity="unknown",
        )

        results = beam.recall(rare_token, top_k=10)
        by_veracity = {r.get("veracity"): r.get("score", 0.0) for r in results}
        # Both rows should surface.
        assert "stated" in by_veracity, f"stated row not in results: {results}"
        assert "unknown" in by_veracity, f"unknown row not in results: {results}"
        # And the stated row should score strictly higher — stated=1.0
        # vs unknown=0.8 in VERACITY_WEIGHTS. Any equality means the
        # multiplier was bypassed (the pre-E4 bug shape).
        assert by_veracity["stated"] > by_veracity["unknown"], (
            f"veracity multiplier did not differentiate scores: "
            f"stated={by_veracity['stated']:.4f}, unknown={by_veracity['unknown']:.4f}. "
            f"Pre-E4 the column collapsed to 'unknown' for both rows."
        )

    def test_remember_batch_empty_list_no_op(self, temp_db):
        """Empty items list shouldn't crash or warn."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        ids = beam.remember_batch([])
        assert ids == []

    def test_remember_batch_signature_back_compat(self, temp_db):
        """Calling without the new veracity kwarg must still work —
        existing callers (BEAM benchmark adapter, importers) shouldn't
        need to update their call sites just because E4 added a kwarg."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        ids = beam.remember_batch([
            {"content": "legacy call shape", "source": "test"},
        ])
        assert len(ids) == 1
        # Default behavior: unknown, no warning.
        assert _veracity_for(temp_db, ids[0]) == "unknown"

    def test_remember_batch_force_veracity_ignores_per_item(self, temp_db, caplog):
        """[Codex adversarial] force_veracity=True locks the method
        default and IGNORES per-item veracity. Defends against the
        retrieval-poisoning path where an importer calls
        remember_batch(items, veracity='imported') but an item self-
        elevates with veracity='stated'."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        with caplog.at_level(logging.WARNING):
            ids = beam.remember_batch(
                [
                    {"content": "trusted content", "source": "test"},
                    {"content": "tries to elevate", "source": "test", "veracity": "stated"},
                    {"content": "tries to lie low", "source": "test", "veracity": "unknown"},
                ],
                veracity="imported",
                force_veracity=True,
            )
        # All three rows must carry the method-level 'imported' label,
        # NOT the per-item override.
        for mid in ids:
            assert _veracity_for(temp_db, mid) == "imported", (
                f"force_veracity=True did not enforce method default; "
                f"row {mid} got {_veracity_for(temp_db, mid)!r}"
            )
        # Each ignored per-item override should produce a WARNING so
        # the operator can audit attempts.
        ignored_warns = [
            r for r in caplog.records
            if "force_veracity" in r.message and "ignoring per-item" in r.message
        ]
        assert len(ignored_warns) == 2, (
            f"expected 2 ignored-override warnings (for stated + unknown), "
            f"got {len(ignored_warns)}: {[w.message for w in ignored_warns]}"
        )

    def test_remember_batch_force_veracity_default_false(self, temp_db):
        """Sanity: force_veracity defaults to False, preserving per-item
        override behavior. No silent flip in defaults."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        ids = beam.remember_batch(
            [
                {"content": "uses method default", "source": "test"},
                {"content": "overrides", "source": "test", "veracity": "stated"},
            ],
            veracity="imported",
        )
        assert _veracity_for(temp_db, ids[0]) == "imported"
        assert _veracity_for(temp_db, ids[1]) == "stated"

    def test_recall_full_veracity_ordering(self, temp_db):
        """[testing specialist] Pin the full canonical multiplier order.
        Pre-E4 only the stated > unknown pair was implicitly tested;
        a regression that swapped weights between inferred / tool /
        imported would slip through. E4 is the rank-signal unlocker
        for the experiment, so the full ordering deserves a guard."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        rare_token = "veraxxxordtest"
        # One row per canonical label, distinct content but same rare token.
        for label in ("stated", "inferred", "imported", "unknown", "tool"):
            beam.remember_batch(
                [{"content": f"{rare_token} {label}-tagged content", "source": "t"}],
                veracity=label,
            )

        results = beam.recall(rare_token, top_k=20)
        by_label = {r.get("veracity"): r.get("score", 0.0) for r in results}
        # All five canonical labels should appear.
        missing = {"stated", "inferred", "imported", "unknown", "tool"} - set(by_label.keys())
        assert not missing, f"recall missed labels: {missing}; got {by_label}"

        # Expected descending order per VERACITY_WEIGHTS:
        # stated=1.0 > unknown=0.8 > inferred=0.7 > imported=0.6 > tool=0.5
        ordered = sorted(by_label.items(), key=lambda kv: -kv[1])
        descending_labels = [label for label, _ in ordered]
        expected = ["stated", "unknown", "inferred", "imported", "tool"]
        assert descending_labels == expected, (
            f"veracity multiplier order regression: got {descending_labels} "
            f"expected {expected}. Raw scores: {by_label}"
        )

    def test_recall_handles_null_veracity_in_legacy_row(self, temp_db):
        """[testing specialist] Legacy rows (pre-veracity-column or
        hand-edited) may have NULL veracity. The recall multiplier's
        veracity_map.get(..., UNKNOWN_WEIGHT) fallback must handle it
        without crashing, applying the UNKNOWN_WEIGHT multiplier."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Insert a row directly via cursor with NULL veracity.
        conn = sqlite3.connect(str(temp_db))
        from datetime import datetime
        conn.execute(
            "INSERT INTO working_memory "
            "(id, content, source, timestamp, session_id, importance, veracity) "
            "VALUES (?, ?, ?, ?, ?, ?, NULL)",
            ("null-ver-1", "legacy null-veracity nullycontent", "test",
             datetime.now().isoformat(), "s1", 0.5),
        )
        conn.commit()
        conn.close()

        results = beam.recall("nullycontent", top_k=10)
        assert results, "row with NULL veracity not surfaced"
        # The defensive fallback should treat NULL as 'unknown' and
        # apply UNKNOWN_WEIGHT — score must be finite and > 0.
        for r in results:
            assert r.get("score", 0.0) > 0, (
                f"NULL-veracity row scored 0/non-finite: {r}"
            )

    def test_recall_handles_junk_veracity_in_row(self, temp_db):
        """[testing specialist] Defense-in-depth at recall: even if a
        non-canonical label landed in the DB via a non-clamping path
        (raw INSERT, schema migration, hand-edit), the recall scorer
        must fall back to UNKNOWN_WEIGHT and not crash."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        conn = sqlite3.connect(str(temp_db))
        from datetime import datetime
        conn.execute(
            "INSERT INTO working_memory "
            "(id, content, source, timestamp, session_id, importance, veracity) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("junk-ver-1", "junkverlabel contains rare token zzzqqqx", "test",
             datetime.now().isoformat(), "s1", 0.5, "totally_made_up_label"),
        )
        conn.commit()
        conn.close()

        results = beam.recall("zzzqqqx", top_k=10)
        assert results, "row with junk veracity not surfaced"
        for r in results:
            if r.get("id") == "junk-ver-1":
                # Junk veracity should fall through to UNKNOWN_WEIGHT;
                # score must be positive and finite.
                assert r.get("score", 0.0) > 0
                break
        else:
            pytest.fail(f"junk-ver-1 not in results: {results}")

    def test_export_import_preserves_veracity(self, temp_db):
        """[Codex adversarial] Backup round-trip must preserve
        per-row veracity. Pre-E4 fix, export omitted the column —
        restored rows collapsed to 'unknown' and lost their rank
        signal. Same shape as the E3 consolidated_at gap."""
        import tempfile as _tempfile

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        ids = beam.remember_batch(
            [
                {"content": "stated content", "source": "t"},
                {"content": "tool observation", "source": "t"},
            ],
            veracity="stated",
        )
        # Differentiate the second row.
        beam.conn.execute(
            "UPDATE working_memory SET veracity = 'tool' WHERE id = ?",
            (ids[1],),
        )
        beam.conn.commit()

        # Read originals.
        original = {mid: _veracity_for(temp_db, mid) for mid in ids}

        # Export → fresh DB → import.
        export = beam.export_to_dict()
        with _tempfile.TemporaryDirectory() as td:
            dest_path = Path(td) / "restored.db"
            beam_dest = BeamMemory(session_id="s1", db_path=dest_path)
            beam_dest.import_from_dict(export)

            for mid in ids:
                restored = _veracity_for(dest_path, mid)
                assert restored == original[mid], (
                    f"export/import lost veracity for {mid}: "
                    f"original={original[mid]!r}, restored={restored!r}"
                )

    def test_remember_method_also_clamps(self, temp_db, caplog):
        """[security specialist + Codex] BeamMemory.remember() now
        clamps veracity too — consistency with remember_batch and
        the C12.b pattern at the provider boundary."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        with caplog.at_level(logging.WARNING):
            beam.remember("content one", source="t", veracity="STATED")
            beam.remember("content two", source="t", veracity="random_garbage")

        # Case-folded canonical landed.
        rows = sqlite3.connect(str(temp_db)).execute(
            "SELECT content, veracity FROM working_memory ORDER BY content"
        ).fetchall()
        by_content = {c: v for c, v in rows}
        assert by_content["content one"] == "stated"
        # Non-canonical clamped.
        assert by_content["content two"] == "unknown"
        # Warning fired for the clamp.
        assert any(
            "random_garbage" in r.message for r in caplog.records
        ), f"no clamp warning for remember(): {caplog.records}"

    def test_clamp_truncates_long_raw_value_in_log(self, caplog):
        """[Codex / Claude adv] Bad veracity strings can be arbitrarily
        long (e.g. an LLM dumping the full prompt into the slot). The
        WARNING log must truncate to avoid log flood / content leak."""
        long_garbage = "x" * 500
        with caplog.at_level(logging.WARNING):
            result = clamp_veracity(long_garbage, context="test")
        assert result == "unknown"
        warn = next((r for r in caplog.records if r.levelname == "WARNING"), None)
        assert warn is not None
        # 80-char cap + truncation marker should appear, but full
        # 500-char value must NOT.
        assert long_garbage not in warn.message
        assert "[truncated]" in warn.message
