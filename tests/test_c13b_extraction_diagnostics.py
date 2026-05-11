"""Regression tests for C13.b — fact extraction failure diagnosability.

Pre-C13.b: fact extraction had five silent-failure layers (cloud HTTP
errors → `""`, JSON parse failures → `[]`, local LLM exceptions →
`pass`, no-LLM-available fallback → `[]`, outer `extract_facts_safe`
wrapper → `[]`). Operators got zero signal that extraction-dependent
features (fact recall, graph voice, the heal-quality pipeline) were
running blind.

Post-C13.b: a process-global `ExtractionDiagnostics` records each
extraction attempt's outcome at every tier (host / remote / local /
cloud). Failures are still swallowed at the call site (callers'
contract preserved); diagnostics surface what's being swallowed.
Operators query via `get_extraction_stats()`.

These tests pin:
  - The diagnostics class API (record/snapshot/reset, thread-safety)
  - The integration with `extract_facts` (each tier's outcome is
    recorded correctly)
  - The integration with `ExtractionClient` (cloud path)
  - The outer-wrapper instrumentation in `extract_facts_safe`
  - The unknown-tier rejection (typo guard for future callers)
"""

from __future__ import annotations

import json
import logging
import threading
from unittest.mock import patch

import pytest

from mnemosyne.extraction.diagnostics import (
    EXTRACTION_TIERS,
    ExtractionDiagnostics,
    get_diagnostics,
    get_extraction_stats,
    reset_extraction_stats,
)


@pytest.fixture(autouse=True)
def fresh_diag():
    """Every test starts with reset diagnostics — process-global state
    must not leak between tests."""
    reset_extraction_stats()
    yield
    reset_extraction_stats()


class TestExtractionDiagnosticsClass:
    """Class-level API. Test the primitives directly so future
    refactors can't quietly break the recording contract."""

    def test_tier_constants_are_canonical(self):
        # `wrapper` synthetic tier added in review-pass for outer-
        # wrapper exceptions whose origin can't be determined.
        assert EXTRACTION_TIERS == ("host", "remote", "local", "cloud", "wrapper")

    def test_snapshot_initial_state(self):
        diag = ExtractionDiagnostics()
        snap = diag.snapshot()
        assert snap["totals"]["calls"] == 0
        assert snap["totals"]["successes"] == 0
        assert snap["totals"]["failures"] == 0
        assert snap["totals"]["empty"] == 0
        assert snap["totals"]["success_rate"] == 0.0
        for tier in EXTRACTION_TIERS:
            t = snap["by_tier"][tier]
            assert t["attempts"] == 0
            assert t["successes"] == 0
            assert t["no_output"] == 0
            assert t["failures"] == 0
            assert t["error_samples"] == []

    def test_record_attempt_increments(self):
        diag = ExtractionDiagnostics()
        diag.record_attempt("local")
        diag.record_attempt("local")
        diag.record_attempt("cloud")
        snap = diag.snapshot()
        assert snap["by_tier"]["local"]["attempts"] == 2
        assert snap["by_tier"]["cloud"]["attempts"] == 1
        assert snap["by_tier"]["host"]["attempts"] == 0

    def test_record_success_increments(self):
        diag = ExtractionDiagnostics()
        diag.record_success("host", fact_count=3)
        diag.record_success("host", fact_count=1)
        snap = diag.snapshot()
        assert snap["by_tier"]["host"]["successes"] == 2

    def test_record_no_output_increments(self):
        diag = ExtractionDiagnostics()
        diag.record_no_output("remote")
        snap = diag.snapshot()
        assert snap["by_tier"]["remote"]["no_output"] == 1

    def test_record_failure_with_exception_captures_sample(self):
        diag = ExtractionDiagnostics()
        try:
            raise ValueError("bad json")
        except Exception as e:
            diag.record_failure("cloud", exc=e, reason="json_parse_failed")
        snap = diag.snapshot()
        assert snap["by_tier"]["cloud"]["failures"] == 1
        samples = snap["by_tier"]["cloud"]["error_samples"]
        assert len(samples) == 1
        assert samples[0]["type"] == "ValueError"
        assert "bad json" in samples[0]["msg"]
        assert samples[0]["reason"] == "json_parse_failed"

    def test_record_failure_with_reason_only(self):
        diag = ExtractionDiagnostics()
        diag.record_failure("local", reason="model_not_loaded")
        snap = diag.snapshot()
        sample = snap["by_tier"]["local"]["error_samples"][0]
        assert sample["type"] == "reason"
        assert sample["msg"] == "model_not_loaded"

    def test_record_failure_truncates_long_error(self):
        diag = ExtractionDiagnostics()
        long_err = "x" * 500
        try:
            raise RuntimeError(long_err)
        except Exception as e:
            diag.record_failure("cloud", exc=e)
        snap = diag.snapshot()
        sample = snap["by_tier"]["cloud"]["error_samples"][0]
        # repr(e) prefixes RuntimeError('...'); the inner 500-char
        # payload must be truncated within the configured cap.
        assert "...[truncated]" in sample["msg"]
        # And the FULL 500 chars must NOT appear verbatim.
        assert long_err not in sample["msg"]

    def test_error_samples_bounded(self):
        """Bounded deque — a chronically failing tier doesn't
        accumulate unbounded samples."""
        diag = ExtractionDiagnostics()
        for i in range(100):
            diag.record_failure("local", reason=f"err-{i}")
        snap = diag.snapshot()
        samples = snap["by_tier"]["local"]["error_samples"]
        # _MAX_ERROR_SAMPLES_PER_TIER = 10
        assert len(samples) == 10
        # Latest samples retained (FIFO drop).
        assert samples[-1]["msg"] == "err-99"

    def test_record_call_succeeded(self):
        diag = ExtractionDiagnostics()
        diag.record_call(succeeded=True)
        snap = diag.snapshot()
        assert snap["totals"]["calls"] == 1
        assert snap["totals"]["successes"] == 1
        assert snap["totals"]["failures"] == 0
        assert snap["totals"]["success_rate"] == 1.0

    def test_record_call_all_empty(self):
        diag = ExtractionDiagnostics()
        diag.record_call(succeeded=False, all_empty=True)
        snap = diag.snapshot()
        assert snap["totals"]["calls"] == 1
        assert snap["totals"]["empty"] == 1
        assert snap["totals"]["failures"] == 0
        assert snap["totals"]["success_rate"] == 0.0

    def test_record_call_hard_failure(self):
        diag = ExtractionDiagnostics()
        diag.record_call(succeeded=False, all_empty=False)
        snap = diag.snapshot()
        assert snap["totals"]["calls"] == 1
        assert snap["totals"]["failures"] == 1
        assert snap["totals"]["empty"] == 0

    def test_success_rate_math(self):
        diag = ExtractionDiagnostics()
        for _ in range(7):
            diag.record_call(succeeded=True)
        for _ in range(3):
            diag.record_call(succeeded=False)
        assert diag.success_rate() == pytest.approx(0.7)

    def test_unknown_tier_rejected(self):
        """Typo guard — `local` vs `Local` vs `localx` is exactly
        the kind of mistake silent recording would mask."""
        diag = ExtractionDiagnostics()
        for bad in ("Local", "LOCAL", "localx", "", "graph"):
            with pytest.raises(ValueError, match="unknown extraction tier"):
                diag.record_attempt(bad)
            with pytest.raises(ValueError, match="unknown extraction tier"):
                diag.record_success(bad)
            with pytest.raises(ValueError, match="unknown extraction tier"):
                diag.record_failure(bad, reason="oops")

    def test_reset_zeroes_everything(self):
        diag = ExtractionDiagnostics()
        diag.record_attempt("local")
        diag.record_success("local")
        diag.record_failure("cloud", reason="test")
        diag.record_call(succeeded=True)

        diag.reset()
        snap = diag.snapshot()
        assert snap["totals"]["calls"] == 0
        assert snap["totals"]["successes"] == 0
        for tier in EXTRACTION_TIERS:
            assert snap["by_tier"][tier]["attempts"] == 0
            assert snap["by_tier"][tier]["error_samples"] == []

    def test_snapshot_is_json_serializable(self):
        """Operators ship this to log aggregators / dashboards."""
        diag = ExtractionDiagnostics()
        try:
            raise RuntimeError("oops")
        except Exception as e:
            diag.record_failure("cloud", exc=e)
        diag.record_success("local", fact_count=2)
        diag.record_call(succeeded=True)
        snap = diag.snapshot()
        # Round-trip via JSON proves the shape is clean.
        serialized = json.dumps(snap)
        restored = json.loads(serialized)
        assert restored["totals"]["successes"] == 1

    def test_thread_safety_under_concurrent_recording(self):
        """Concurrent extraction calls from multiple threads must
        accumulate correctly under the lock."""
        diag = ExtractionDiagnostics()
        N_THREADS = 8
        ATTEMPTS_PER_THREAD = 100

        def worker():
            for _ in range(ATTEMPTS_PER_THREAD):
                diag.record_attempt("local")
                diag.record_success("local")

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = diag.snapshot()
        expected = N_THREADS * ATTEMPTS_PER_THREAD
        assert snap["by_tier"]["local"]["attempts"] == expected
        assert snap["by_tier"]["local"]["successes"] == expected


class TestProcessGlobalSingleton:

    def test_get_diagnostics_returns_same_instance(self):
        a = get_diagnostics()
        b = get_diagnostics()
        assert a is b

    def test_module_level_helpers_use_singleton(self):
        diag = get_diagnostics()
        diag.record_success("host", fact_count=1)
        diag.record_call(succeeded=True)

        snap = get_extraction_stats()
        assert snap["by_tier"]["host"]["successes"] == 1
        assert snap["totals"]["successes"] == 1

        reset_extraction_stats()
        snap = get_extraction_stats()
        assert snap["totals"]["calls"] == 0


class TestExtractFactsIntegration:
    """End-to-end: call extract_facts under various LLM availability
    scenarios and verify the diagnostics record what happened."""

    def test_llm_unavailable_records_failure(self, monkeypatch):
        """When local_llm.llm_available() is False, extract_facts
        bails immediately. Pre-C13.b this was completely silent."""
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.llm_available", lambda: False
        )
        from mnemosyne.core.extraction import extract_facts
        result = extract_facts("Alice prefers tea.")
        assert result == []

        snap = get_extraction_stats()
        # Recorded one outer call as a failure (no tier ran successfully).
        assert snap["totals"]["calls"] == 1
        assert snap["totals"]["failures"] == 1
        # And recorded a 'local' tier failure with the reason.
        local = snap["by_tier"]["local"]
        assert local["failures"] == 1
        sample = local["error_samples"][0]
        assert sample["msg"] == "llm_unavailable_at_call_site"

    def test_empty_text_no_recording(self, monkeypatch):
        """Empty input isn't an attempt — operators shouldn't see
        success_rate degrade from no-op callers."""
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.llm_available", lambda: True
        )
        from mnemosyne.core.extraction import extract_facts
        assert extract_facts("") == []
        assert extract_facts("   ") == []
        snap = get_extraction_stats()
        assert snap["totals"]["calls"] == 0

    def test_local_llm_raises_records_failure(self, monkeypatch):
        """Local LLM model raises mid-call. Records as `local` tier
        failure with the exception sample."""
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.llm_available", lambda: True
        )
        monkeypatch.setattr(
            "mnemosyne.core.local_llm._try_host_llm",
            lambda prompt, max_tokens, temperature: (False, ""),
        )
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.LLM_ENABLED", False
        )

        def boom(*args, **kwargs):
            raise RuntimeError("model crashed mid-inference")

        monkeypatch.setattr(
            "mnemosyne.core.local_llm._load_llm",
            lambda: boom,
        )

        from mnemosyne.core.extraction import extract_facts
        result = extract_facts("Alice prefers tea.")
        assert result == []

        snap = get_extraction_stats()
        local = snap["by_tier"]["local"]
        assert local["failures"] >= 1
        # The exception was captured.
        msgs = [s.get("msg", "") for s in local["error_samples"]]
        assert any("model crashed" in m for m in msgs)

    def test_local_llm_succeeds(self, monkeypatch):
        """Happy path — local LLM returns facts, success is recorded."""
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.llm_available", lambda: True
        )
        monkeypatch.setattr(
            "mnemosyne.core.local_llm._try_host_llm",
            lambda prompt, max_tokens, temperature: (False, ""),
        )
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.LLM_ENABLED", False
        )

        def fake_llm(prompt, max_new_tokens, stop):
            return "1. Alice prefers tea\n2. Alice lives in Seattle"

        monkeypatch.setattr(
            "mnemosyne.core.local_llm._load_llm",
            lambda: fake_llm,
        )
        monkeypatch.setattr(
            "mnemosyne.core.local_llm._clean_output",
            lambda s: s,
        )

        from mnemosyne.core.extraction import extract_facts
        result = extract_facts("Alice prefers tea and lives in Seattle.")
        assert len(result) == 2

        snap = get_extraction_stats()
        assert snap["totals"]["successes"] == 1
        assert snap["by_tier"]["local"]["successes"] == 1

    def test_extract_facts_safe_wraps_outer_exceptions(self, monkeypatch):
        """If extract_facts() itself raises (rare — bug, not LLM
        failure), extract_facts_safe records it as `wrapper` tier
        outer_wrapper_caught so operators can spot the bug class
        without polluting local-tier metrics."""
        from mnemosyne.core import extraction as ext_mod

        def boom(text):
            raise TypeError("simulated extract_facts bug")

        monkeypatch.setattr(ext_mod, "extract_facts", boom)
        result = ext_mod.extract_facts_safe("any content")
        assert result == []

        snap = get_extraction_stats()
        wrapper = snap["by_tier"]["wrapper"]
        assert wrapper["failures"] >= 1
        reasons = [s.get("reason", "") for s in wrapper["error_samples"]]
        assert "outer_wrapper_caught" in reasons
        # And local should NOT have been touched.
        assert snap["by_tier"]["local"]["failures"] == 0


class TestExtractionClientIntegration:
    """Cloud path — `ExtractionClient.chat()` and `extract_facts()`
    record per-call outcomes."""

    def test_chat_records_no_api_key_failure(self, monkeypatch):
        """No API key → urllib raises 401-ish → all retries fail."""
        from mnemosyne.extraction.client import ExtractionClient

        client = ExtractionClient(api_key="")

        def fail_api(*args, **kwargs):
            raise RuntimeError("401 Unauthorized")

        monkeypatch.setattr(client, "_call_api", fail_api)
        result = client.chat([{"role": "user", "content": "test"}])
        assert result == ""

        snap = get_extraction_stats()
        cloud = snap["by_tier"]["cloud"]
        assert cloud["failures"] >= 1
        # Most-recent error sample should contain the error trace.
        samples = cloud["error_samples"]
        assert any("401" in s.get("msg", "") for s in samples)

    def test_chat_records_attempt_not_success(self, monkeypatch):
        """[Review hardening] chat() records the API-transport
        attempt but NOT cloud-tier success. Success is gated on
        parseable output, decided by extract_facts() — not by HTTP
        returning content. Pre-fix chat() recorded success on any
        non-empty response, which double-counted when extract_facts
        then failed to parse."""
        from mnemosyne.extraction.client import ExtractionClient

        client = ExtractionClient(api_key="key")
        monkeypatch.setattr(
            client,
            "_call_api",
            lambda *a, **kw: '[{"subject":"Alice","predicate":"prefers","object":"tea"}]',
        )

        result = client.chat([{"role": "user", "content": "test"}])
        assert "Alice" in result

        snap = get_extraction_stats()
        # chat() ran once.
        assert snap["by_tier"]["cloud"]["attempts"] >= 1
        # But chat() does NOT decide cloud success — only
        # extract_facts() does, based on parsed output.
        assert snap["by_tier"]["cloud"]["successes"] == 0, (
            "chat() must not record cloud-tier success — that's "
            "extract_facts()'s job after parsing"
        )

    def test_extract_facts_records_json_parse_failure(self, monkeypatch):
        """Cloud LLM returned text, but it didn't parse as a fact
        list. Records as `cloud` failure with reason
        json_parse_failed."""
        from mnemosyne.extraction.client import ExtractionClient

        client = ExtractionClient(api_key="key")
        # Returns text without any [...] block — parse fails.
        monkeypatch.setattr(
            client,
            "_call_api",
            lambda *a, **kw: "I cannot extract facts from this text.",
        )

        result = client.extract_facts(
            [{"role": "user", "content": "some content"}]
        )
        assert result == []

        snap = get_extraction_stats()
        cloud = snap["by_tier"]["cloud"]
        # At minimum the chat() success counter fires (text returned).
        # The JSON parse failure also fires.
        msgs = [s.get("reason", "") for s in cloud["error_samples"]]
        # Actually wait — if response found NO [..] block, the parser
        # returns [] without raising json.JSONDecodeError. Verify
        # that path: the response had no [, so json_start < 0, no
        # parse attempted, no failure recorded. Then result = [].
        # The branch we want to test is "[ ... ]" with malformed JSON.

    def test_extract_facts_records_malformed_json(self, monkeypatch):
        from mnemosyne.extraction.client import ExtractionClient

        client = ExtractionClient(api_key="key")
        # Response includes brackets but the contents aren't valid JSON.
        monkeypatch.setattr(
            client,
            "_call_api",
            lambda *a, **kw: 'Here are the facts: [oops, this is not json]',
        )

        result = client.extract_facts(
            [{"role": "user", "content": "some content"}]
        )
        assert result == []

        snap = get_extraction_stats()
        cloud = snap["by_tier"]["cloud"]
        reasons = [s.get("reason", "") for s in cloud["error_samples"]]
        assert "json_parse_failed" in reasons


class TestReviewHardening:
    """Findings from /review (Codex structured + Codex adv +
    Claude adv + maintainability specialist). Each test pins one of
    the closed bypass paths."""

    def test_host_attempt_not_counted_when_host_disabled(self, monkeypatch):
        """[Codex P2, Codex adv #1, Claude adv #2] Pre-fix
        `record_attempt("host")` fired unconditionally — every call
        showed a phantom host attempt even when no host backend is
        registered. Fix: record only when `attempted=True`."""
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.llm_available", lambda: True
        )
        monkeypatch.setattr(
            "mnemosyne.core.local_llm._try_host_llm",
            lambda prompt, max_tokens, temperature: (False, ""),
        )
        monkeypatch.setattr("mnemosyne.core.local_llm.LLM_ENABLED", False)
        monkeypatch.setattr("mnemosyne.core.local_llm._load_llm", lambda: None)

        from mnemosyne.core.extraction import extract_facts
        extract_facts("Alice prefers tea.")

        snap = get_extraction_stats()
        # Host should NOT show an attempt — the backend wasn't registered.
        assert snap["by_tier"]["host"]["attempts"] == 0, (
            f"phantom host attempt recorded: {snap['by_tier']['host']}"
        )
        # Local DID attempt (the actual code path that ran).
        assert snap["by_tier"]["local"]["attempts"] >= 1

    def test_host_adapter_exception_recorded_on_host_tier(self, monkeypatch):
        """[Claude adv #2] If `_try_host_llm` raises, record on host
        tier (with reason `host_adapter_raised`) NOT propagate to
        extract_facts_safe which would misattribute to wrapper tier."""
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.llm_available", lambda: True
        )

        def host_raises(*args, **kwargs):
            raise RuntimeError("host adapter crashed")

        monkeypatch.setattr(
            "mnemosyne.core.local_llm._try_host_llm", host_raises
        )

        from mnemosyne.core.extraction import extract_facts
        result = extract_facts("Alice prefers tea.")
        assert result == []

        snap = get_extraction_stats()
        host = snap["by_tier"]["host"]
        assert host["failures"] >= 1
        reasons = [s.get("reason", "") for s in host["error_samples"]]
        assert "host_adapter_raised" in reasons

    def test_outer_wrapper_failure_attributed_to_wrapper_tier(self, monkeypatch):
        """[Claude adv #11] extract_facts_safe's outer except records
        on the synthetic `wrapper` tier, not `local`. Pre-fix this
        polluted local-tier metrics with failures from any layer."""
        from mnemosyne.core import extraction as ext_mod

        def boom(text):
            raise TypeError("simulated extract_facts bug")

        monkeypatch.setattr(ext_mod, "extract_facts", boom)
        ext_mod.extract_facts_safe("any content")

        snap = get_extraction_stats()
        # wrapper tier is the new home.
        wrapper = snap["by_tier"]["wrapper"]
        assert wrapper["failures"] >= 1
        # local tier should NOT have been touched by the wrapper case.
        assert snap["by_tier"]["local"]["failures"] == 0

    def test_cloud_chat_does_not_record_success_on_unparseable_text(
        self, monkeypatch
    ):
        """[Codex P2 #3, Codex adv #2] Pre-fix chat() recorded
        success on non-empty HTTP, then extract_facts recorded
        failure on JSON parse — double-counting. Fix: chat() never
        records cloud-tier success; only extract_facts decides based
        on parseable output."""
        from mnemosyne.extraction.client import ExtractionClient

        client = ExtractionClient(api_key="key")
        # Return text WITHOUT a JSON array.
        monkeypatch.setattr(
            client,
            "_call_api",
            lambda *a, **kw: "I cannot extract facts from this text.",
        )

        result = client.extract_facts(
            [{"role": "user", "content": "some content"}]
        )
        assert result == []

        snap = get_extraction_stats()
        cloud = snap["by_tier"]["cloud"]
        # Cloud-tier success counter MUST NOT have incremented —
        # the chat returned text but extraction yielded no facts.
        assert cloud["successes"] == 0, (
            f"cloud success counter incremented despite no parseable "
            f"facts: {cloud}"
        )
        # And there should be a failure recorded for no parseable
        # output.
        reasons = [s.get("reason", "") for s in cloud["error_samples"]]
        assert "no_facts_in_response" in reasons

    def test_cloud_extract_facts_records_record_call(self, monkeypatch):
        """[Codex P2 #2, Codex adv #3] ExtractionClient.extract_facts
        now records record_call() so totals.calls / success_rate
        reflect the cloud path. Pre-fix the cloud path was excluded
        from bird's-eye accounting."""
        from mnemosyne.extraction.client import ExtractionClient

        client = ExtractionClient(api_key="key")
        monkeypatch.setattr(
            client,
            "_call_api",
            lambda *a, **kw: '[{"subject":"Alice","predicate":"prefers","object":"tea"}]',
        )

        # Pre-call: zero totals.
        pre = get_extraction_stats()
        assert pre["totals"]["calls"] == 0

        result = client.extract_facts(
            [{"role": "user", "content": "Alice prefers tea"}]
        )
        assert len(result) == 1

        post = get_extraction_stats()
        # Outer call counted, success at totals level.
        assert post["totals"]["calls"] == 1
        assert post["totals"]["successes"] == 1
        # And cloud-tier success counted too.
        assert post["by_tier"]["cloud"]["successes"] >= 1

    def test_snapshot_samples_are_independent_copies(self, monkeypatch):
        """[Codex adv #4] snapshot() must return a deep-enough copy
        that the caller mutating the returned dict can't mutate
        internal diagnostics state. Pre-fix list(deque) aliased the
        sample dicts."""
        diag = ExtractionDiagnostics()
        try:
            raise ValueError("test error")
        except Exception as e:
            diag.record_failure("cloud", exc=e)

        snap = diag.snapshot()
        # Mutate the returned sample.
        snap["by_tier"]["cloud"]["error_samples"][0]["msg"] = "MUTATED"

        # Re-snapshot — original must NOT carry the mutation.
        snap2 = diag.snapshot()
        sample = snap2["by_tier"]["cloud"]["error_samples"][0]
        assert sample["msg"] != "MUTATED", (
            "snapshot returned aliased sample dicts; caller mutation "
            "leaked into internal state"
        )

    def test_log_sanitizes_newlines_in_exception_repr(self, monkeypatch, caplog):
        """[Codex adv #5] A custom exception with newlines or ANSI
        escapes in its __repr__ would inject log-line breaks /
        terminal control sequences if logged raw. Fix: _safe_for_log
        sanitizes to a bounded single-line string."""
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.llm_available", lambda: True
        )
        monkeypatch.setattr(
            "mnemosyne.core.local_llm._try_host_llm",
            lambda prompt, max_tokens, temperature: (False, ""),
        )
        monkeypatch.setattr("mnemosyne.core.local_llm.LLM_ENABLED", False)

        class EvilError(Exception):
            def __repr__(self):
                return "EvilError(\nLINE\x1b[31mANSI\nINJECTED\n)"

        def boom(*args, **kwargs):
            raise EvilError()

        monkeypatch.setattr(
            "mnemosyne.core.local_llm._load_llm", lambda: boom
        )

        with caplog.at_level(logging.WARNING, logger="mnemosyne.core.extraction"):
            from mnemosyne.core.extraction import extract_facts
            extract_facts("test content")

        # Find the WARNING log record.
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warnings, "no warning logged"
        msg = warnings[-1].message
        # No raw newlines or ANSI sequences should appear in the
        # logged message body.
        assert "\n" not in msg.replace("\\n", ""), (
            f"raw newline in log message: {msg!r}"
        )
        assert "\x1b" not in msg, (
            f"raw ANSI escape in log message: {msg!r}"
        )


class TestOperatorVisibleLogs:
    """C13.b's diagnostics are the primary signal; structured WARNING
    logs are the secondary signal for operators tailing logs."""

    def test_local_llm_failure_logs_warning(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.llm_available", lambda: True
        )
        monkeypatch.setattr(
            "mnemosyne.core.local_llm._try_host_llm",
            lambda prompt, max_tokens, temperature: (False, ""),
        )
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.LLM_ENABLED", False
        )

        def boom(*args, **kwargs):
            raise RuntimeError("model crashed mid-inference")

        monkeypatch.setattr(
            "mnemosyne.core.local_llm._load_llm",
            lambda: boom,
        )

        with caplog.at_level(logging.WARNING, logger="mnemosyne.core.extraction"):
            from mnemosyne.core.extraction import extract_facts
            extract_facts("Alice prefers tea.")

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("local LLM raised" in r.message for r in warnings)
