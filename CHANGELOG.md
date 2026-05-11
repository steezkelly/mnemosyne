# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Simple Versioning](https://github.com/AxDSan/mnemosyne) (MAJOR.MINOR).

## [Unreleased]

<<<<<<< HEAD
<<<<<<< HEAD
### Security

**C25 — DeltaSync hardening (table allowlist, opt-in column allowlist, qualified SQL, per-table checkpoints)**
- `compute_delta` / `apply_delta` / `sync_to` / `sync_from` validate the `table` kwarg via `_validate_table` with a **strict `type(table) is str`** check (not `isinstance`) against `ALLOWED_DELTA_TABLES = {"working_memory", "episodic_memory"}`. The strict check defeats a `str`-subclass bypass where `__eq__` / `__hash__` compare-equal to an allowlisted value while the f-string interpolates a different payload (caught by `/review` adversarial pass — verified exploit demonstrating subquery-in-FROM-position SQL injection).
- All SQL operations use `main.` schema qualification (`UPDATE "main"."working_memory" SET ...`) so a same-connection temp table named `working_memory` cannot shadow the real one. Pre-fix `PRAGMA table_info({table})` resolved unqualified to whichever schema came first (SQLite prefers temp), letting a hostile component with same-connection access redirect both schema reads and writes.
- Column filtering on apply is now **opt-in via static allowlist** (intersected with live schema), not opt-out via reserved set:
  - `_DELTA_UPDATABLE_COLUMNS` (UPDATE path): only `content`, `importance`, `metadata_json`, `veracity`, `memory_type`, `binary_vector`, `source`, `summary_of` may be mutated by a peer. Identity (`id`), scope (`session_id`, `scope`), lifecycle (`valid_until`, `superseded_by`, `created_at`, `timestamp`, `recall_count`, `last_recalled`, `consolidated_at`, `degraded_at`, `tier`), and authorship (`author_id`, `author_type`, `channel_id`) are all destination-controlled. Pre-fix `session_id` and `superseded_by` were schema columns NOT in the reserved-update set — a peer could re-route victim rows into the attacker's session or soft-delete arbitrary rows via UPDATE.
  - `_DELTA_INSERTABLE_COLUMNS` (INSERT path): only `id` + the same content/metadata fields + `timestamp` are accepted from the peer. Pre-fix INSERT reserved only `rowid`, letting a peer backdate `created_at`, forge `timestamp`, plant rows directly in the destination's session, or pre-tombstone.
- SQL identifiers (column names) are now quoted (`"col" = ?` / `INSERT INTO t ("c1", "c2") ...`) for defense-in-depth against schema poisoning. Static allowlist intersection with the live schema is the primary gate; identifier quoting is the secondary.
- **Checkpoints scoped by `(peer_id, table)`** — pre-fix a single per-peer checkpoint covered all tables, so peer syncing `working_memory` to rowid=100 then `compute_delta(peer, table="episodic_memory")` skipped episodic rows at rowid < 100 (rowid namespaces are table-local). New filename: `checkpoint_<peer>__<table>.json`. Legacy `checkpoint_<peer>.json` files load as the `working_memory` checkpoint for backward compat.
- `_allowed_columns` self-validates via `_validate_table` for defense-in-depth — direct calls bypassing the public methods can't reach the PRAGMA path with unvalidated input.
- Maintainer just wired streaming emit live in commit `b2a7fae` (issue #64), raising the practical relevance of the hardening: production callers now have real reasons to construct deltas across the wire.

### Changed

**DeltaSync stats output now includes `filtered_keys`**
- `apply_delta` return shape: `{"inserted": N, "updated": N, "skipped": N, "filtered_keys": N}`. New `filtered_keys` counter exposes peer-supplied keys that were rejected by the schema column allowlist. Operators can spot a misconfigured peer (typo'd column names) or a hostile peer (injection attempts) by watching this counter.

### Documentation

**`docs/api-reference.md`: corrected MemoryStream and DeltaSync examples** (originally PR #49 by @kohai-ut, rolled in here)
- `MemoryStream`: examples now use the real API — `emit(MemoryEvent(...))`, `on(event_type, callback)`, `on_any(callback)`, `listen()` iterator. Pre-fix the docs showed `push(...)`, `on_event(...)`, and direct iteration — none of which exist.
- `DeltaSync`: examples now show the peer_id-based call shape (`compute_delta(peer_id)`, `apply_delta(peer_id, delta)`, `sync_to(peer_id)`, `sync_from(peer_id, delta)`). Pre-fix the docs showed `compute_delta()` / `apply_delta(delta)` / `sync_to(other_mnemosyne)` — wrong signatures.
- New section on the `ALLOWED_DELTA_TABLES` allowlist + column filtering on apply (the security hardening above).
=======
### Added

**C4 — Recall path provenance diagnostics**
- New `mnemosyne.core.recall_diagnostics` module exposes a process-global `RecallDiagnostics` instance. Pre-C4 `BeamMemory.recall` had silent fallback layers per tier: WM (FTS + vec wrapped in `try/except`, falling through to substring scoring on recent items) and EM (vec + FTS, falling through to substring scoring on the most-recent 500 episodic rows when both produced nothing). Operators saw results but had no signal which path produced them — FTS-ranked good signal looked identical to substring-on-recent weak signal.
- `BeamMemory.recall()` now instruments each decision point: WM FTS hit count, WM vec-only hit count, WM fallback fired (boolean per call + scanned-row count), EM FTS hit count, EM vec-only hit count, EM fallback fired + scanned count. Plus an outer `record_call(truly_empty=...)` that distinguishes "fallback fired but returned weak hits" from "literally no results from any path."
- New module-level helpers: `get_recall_diagnostics()` returns a JSON-serializable snapshot; `reset_recall_diagnostics()` zeroes the counters (useful for tests + operators starting a fresh measurement window before a benchmark run).
- Snapshot exposes `totals.wm_fallback_rate` and `totals.em_fallback_rate` — the fraction of calls where the fallback layer fired. Operators monitoring a BEAM experiment alarm if these rise above an expected baseline (corpus + query distribution dependent).

### Notes for callers

- Diagnostics are **read-only signal** — they never alter recall behavior. The fallback still fires when FTS/vec produce nothing (legitimate no-match case); diagnostics expose WHEN and HOW OFTEN.
- Thread-safe: single `threading.Lock` gates all mutations.
- API: `from mnemosyne.core.recall_diagnostics import get_recall_diagnostics, reset_recall_diagnostics, get_diagnostics, RecallDiagnostics`.
- **For the BEAM experiment**: after running an arm, snapshot the diagnostics. If `wm_fallback_rate` or `em_fallback_rate` is high (>0.2 ish), recall scores in that arm are dominated by the weak-signal substring path. Arm-vs-arm comparisons need similar rates across arms to be interpretable.

### Counter semantics (post-/review hardening)

- **Per-row tier attribution.** Each kept row credits exactly one tier (FTS+vec overlap credited to FTS; vec-only rows to vec; fallback rows to wm_fallback/em_fallback). Sum across tiers per call = total kept rows for that call (excluding entity-aware expansion which is a separate signal source). Pre-fix counters recorded pre-filter candidate sets — rows that FTS/vec returned but got dropped by `wm_where`/`em_where` (session/scope/date/source/etc.) inflated the counters, making the provenance and fallback-rate signals look healthier than reality.
- **Kept rows, not scanned rows.** Both `wm_fallback` and `em_fallback` counters increment for rows that pass the substring relevance threshold (0.02) and end up in the result list — not for rows merely scanned by the fallback's `LIMIT 500` SELECT. Pre-fix the EM fallback always recorded `len(scanned_rows)` regardless of whether any survived the threshold.
- **`truly_empty` is post-filter strict.** True only when `len(final_results) == 0` AND zero kept rows across all tiers. Distinguishes "no signal anywhere" from "candidates existed but got filtered" (top_k=0 callers, post-filter dropouts, etc.).
- **`fallback_rate` clamped at 1.0.** Defends against a reset-mid-call race where pre-reset `record_fallback_used` calls could accumulate against a post-reset `_total_calls` counter, producing >1.0 ratios in dashboards.
>>>>>>> pr79
=======
### Added

**C13.b — Fact-extraction failure diagnostics**
- New `mnemosyne.extraction.diagnostics` module exposes a process-global `ExtractionDiagnostics` instance. Pre-C13.b fact extraction had five silent-failure layers (cloud HTTP errors → `""`, JSON parse failures → `[]`, local LLM exceptions → `pass`, no-LLM-available fallback → `[]`, outer `extract_facts_safe` wrapper → `[]`). Operators got zero signal that fact-recall and the graph voice were running blind.
- The diagnostics record each extraction attempt's outcome at every tier (`host` / `remote` / `local` / `cloud`). Counters per tier: `attempts`, `successes`, `no_output`, `failures`, plus bounded recent error samples. Bird's-eye totals: `calls`, `successes`, `failures`, `empty`, `success_rate`.
- `mnemosyne/core/extraction.py::extract_facts` instruments every tier transition (host attempted vs. succeeded, host fallback to local, remote LLM no-output, local LLM raised, model not loaded, etc.). Each branch records to the diagnostics so operators can see exactly what's being swallowed.
- `mnemosyne/extraction/client.py::ExtractionClient.chat` records `cloud` tier attempts, successes (non-empty response), `no_output` (empty response), and failures with the last exception. `extract_facts()` adds JSON-parse-failure recording so malformed model responses are distinguishable from "model had nothing to say."
- New module-level helpers: `get_extraction_stats()` returns a JSON-serializable snapshot; `reset_extraction_stats()` zeroes the counters (useful for tests + operators starting a fresh measurement window).
- WARNING-level log lines fire on the most actionable failure paths (`ExtractionClient.chat` all-models-failed, `extract_facts` local LLM raised, JSON parse failed, etc.). Diagnostics are the primary signal; logs are the secondary signal for log-tailing operators.

### Notes for callers

- Diagnostics are **read-only signal** — they never alter extraction behavior. Failures are still swallowed at the call site (`extract_facts_safe`, the outer `try/except: pass` in `beam.py::_extract_and_store_facts`); diagnostics surface what's being swallowed.
- Thread-safe: a single `threading.Lock` gates all mutations. Concurrent extraction calls from different threads accumulate correctly.
- The error-sample queue is bounded to 10 per tier; a chronically failing tier doesn't accumulate unbounded memory.
- Long error messages are truncated to 200 chars in the captured sample to bound log volume and prevent content leakage from upstream LLM errors that include the full prompt.
- API: `from mnemosyne.extraction import get_extraction_stats, reset_extraction_stats, get_diagnostics, ExtractionDiagnostics`.

### Counter semantics (post-/review hardening)

- **Tier attempt counters are conditional**, not unconditional. `record_attempt("host")` fires only when the host backend actually ran (`attempted=True` from `_try_host_llm`). Configurations with no host backend registered show zero host attempts, not phantom attempts.
- **Tier success counters reflect parseable output, not transport success.** `ExtractionClient.chat()` records cloud-tier attempts and transport-level outcomes (`no_output` for empty HTTP, `failure` for all-models-failed) but does NOT record cloud-tier success. `ExtractionClient.extract_facts()` records `cloud_success` only after the response parses into a fact list. This means cloud `success_rate` (`successes / attempts`) reflects extraction quality, not API health.
- **Outer-wrapper failures land on the synthetic `wrapper` tier.** `extract_facts_safe`'s outer `except` records to `wrapper` rather than `local` — pre-review the outer exceptions polluted local-tier metrics, misleading operators triaging local-LLM health. The `wrapper` tier explicitly means "tier of origin can't be determined."
- **Host adapter exceptions land on `host` tier.** If `_try_host_llm` itself raises, the failure records under `host` with reason `host_adapter_raised` (caught at the call site) instead of escaping to the outer wrapper.
- **Snapshot samples are independent copies.** `snapshot()` deep-copies the sample dicts so a caller mutating the returned snapshot can't mutate diagnostics' internal state.
- **Log lines sanitize exception repr.** `_safe_for_log` strips control characters / newlines / ANSI escapes from the logged exception representation. A hostile or malformed `__repr__` can't inject log-line breaks or terminal escape sequences.
>>>>>>> pr78

## [2.5] — 2026-05-10

### Added

**NAI-0 Algorithmic Sprint**
- `BeamMemory.format_context(results, format="bullet"|"json")` — structured context formatting
- `BeamMemory._sandwich_order()` — U-shaped attention ordering (high-first, medium-middle, high-last)
- `BeamMemory._fact_line()` — clean one-line fact format with date, source, confidence
- `BeamMemory._format_context_json()` / `_format_context_bullet()` — JSON and markdown output
- RRF (Reciprocal Rank Fusion) in `PolyphonicRecallEngine._combine_voices()` with k=60 constant
- Covering indexes: `idx_em_scope_imp`, `idx_wm_session_recall`, `idx_mem_emb_type`
- `tools/bench_nai0.py` — minimal 20-question benchmark for quick before/after measurement

**Self-Healing Quality Pipeline** (`scripts/heal_quality.py`, PR #67 by ether-btc)
- Detects degraded episodic memory entries (bullet-format, <300 chars) and repairs them via a 4-stage LLM-as-Judge closed loop: Extract → Generate → Judge → Repair
- Fault taxonomy: `truncated`, `generic`, `missing_facts`, `wrong_format`
- Judge scores 4 dimensions (factual density, format compliance, length sufficiency, grounding) each 0-100
- Repair strategies are fault-specific: context doubling, specificity enforcement, fact injection, format rewrite
- Loop with `MAX_RETRIES` (default 3) and automatic escalation to stronger model after 2 failures
- Quality provenance in `metadata_json`: `quality_score`, `judge_model`, `consolidated_at`, `fault_before_repair`, `retry_loop_count`
- Configurable via env: `MNEMOSYNE_HEAL_JUDGE_THRESHOLD`, `MNEMOSYNE_HEAL_MAX_RETRIES`, `MNEMOSYNE_HEAL_MIN_LEN`, `MNEMOSYNE_HEAL_BUDGET`, `MNEMOSYNE_HEAL_ESCALATE_AFTER`
- Works with any LLM backend (MiniMax M2.7 via mmx-cli, local GGUF, or remote OpenAI-compatible API)
- CLI: `python scripts/heal_quality.py [--detect-only] [--entry-id ID] [--dry-run]`

**Chunked LLM Summarization** (`mnemosyne/core/local_llm.py`)
- Splits large memory lists into context-window-sized chunks before summarization
- Two-pass: summarize each chunk individually, then consolidate chunk summaries
- Fixes truncation issues with smaller models (Qwen2.5-1.5B) on large sessions

### Changed
- `BeamMemory.recall()` default `top_k`: 5 → 40
- Polyphonic recall voice combination: weighted average → position-based RRF
- `mnemosyne/__init__.py`: version bump to 2.5.0

## [2.4] — 2026-05-07

### Added

**Hindsight Importer — migrate FROM Hindsight INTO Mnemosyne**
- New `HindsightImporter` class in `mnemosyne/core/importers/hindsight.py`
- Import from Hindsight JSON exports OR live Hindsight HTTP API (`/v1/default/banks/{bank}/memories/list`)
- Writes directly to `episodic_memory` (not working memory) — preserves original timestamps, fact types, session grouping, metadata, scope, and veracity
- Stable duplicate skipping via SHA256-based IDs (`hs_` prefix)
- Importance scoring derived from Hindsight `fact_type` (world=0.75, experience=0.65, observation=0.55) + proof_count bonus
- Full metadata preservation: hindsight_id, fact_type, context, dates, entities, chunk_id, tags, consolidation timestamps
- CLI: `mnemosyne import-hindsight <file.json|url> [bank]`
- Registered in provider registry alongside Mem0, Letta, Zep, Cognee, Honcho, SuperMemory
- 102 lines of regression tests: timestamp preservation, episodic-only import, stable duplicate skipping, FTS indexing, provider-registry usage

**Host LLM Adapter — route consolidation through Hermes' authenticated provider**
- New `mnemosyne/core/llm_backends.py` — tiny `LLMBackend` Protocol (one method: `complete()`), process-global registry, `CallableLLMBackend` dataclass for tests
- New `hermes_memory_provider/hermes_llm_adapter.py` — `HermesAuxLLMBackend` routes through `agent.auxiliary_client.call_llm(task="compression", ...)`
- `MnemosyneMemoryProvider.initialize()` registers the backend; `shutdown()` unregisters it with a brief drain for in-flight threads
- `summarize_memories()` and `extract_facts()` consult host first when `MNEMOSYNE_HOST_LLM_ENABLED=true`
- **Host-skips-remote rule (A3):** When host attempt produces no usable text, remote URL is skipped — falls straight to local GGUF. Prevents stale URL leaks.
- `llm_available()` returns `True` when host backend is registered, so Hermes-only users don't get short-circuited by `beam.sleep()`
- `on_session_end()` runs sleep in daemon thread with 15s join timeout; `shutdown()` drains 2s before unregistering
- Fact extraction uses `temperature=0.0` for determinism; consolidation stays at `0.3`
- 7 new tests covering registry round-trip, host-route precedence, A3 skip-remote rule, gate semantics, shutdown drain race, daemon exception logging, bullet-list output preservation
- Live end-to-end verified with `openai-codex` OAuth subscription through ChatGPT backend

### Why this matters

**Hindsight importer:** Before this, migrating FROM Hindsight required going through `remember()`, which assigned current timestamps and wrote to working memory. Historical memories lost their original context. Now Hindsight migrations preserve the full temporal record with zero data loss.

**Host LLM adapter:** Hermes users on OAuth-backed providers (ChatGPT/Codex subscriptions) could not use Mnemosyne's LLM-backed operations because `MNEMOSYNE_LLM_BASE_URL` expects an OpenAI-compatible API key endpoint, not OAuth. Now they can route through Hermes' already-authenticated auxiliary client with zero extra credentials.

---

## [2.3.1] — 2026-05-06

### Fixed

- **Auto-sleep consolidation blocks TUI agent**: `_maybe_auto_sleep()` now runs in a background thread with a 5-second timeout instead of synchronously. Local LLM summarization (ctransformers) can no longer hang the agent worker thread. (#23)
- `MNEMOSYNE_AUTO_SLEEP_ENABLED` env var now controls auto-sleep behavior. Default is `false` (disabled) for interactive safety. Set to `true` to re-enable.
- Config schema updated to reflect new default.

## [2.3] — 2026-05-05

### Added

**Tiered Episodic Degradation — long-term recall without unbounded growth**
- Three degradation tiers: Tier 1 (0-30d, full detail), Tier 2 (30-180d, LLM-compressed), Tier 3 (180d+, entity-extracted signal)
- Automatic tier promotion during `sleep()` — no manual maintenance
- Tier multipliers in recall scoring: cold memories need 4x stronger semantic match
- Configurable via `MNEMOSYNE_TIER2_DAYS`, `MNEMOSYNE_TIER3_DAYS`, `MNEMOSYNE_TIER*_WEIGHT`
- Mnemonics can now truthfully claim "remembers what you told it a year ago"

**Smart Compression — entity-aware tier 2→3 extraction**
- `_extract_key_signal()` scores sentences by entity density (proper nouns, acronyms, security terms, tech stack, urgency)
- Preserves facts buried anywhere in a long memory, not just the first sentence
- Configurable: `MNEMOSYNE_SMART_COMPRESS=1` (default on), `MNEMOSYNE_TIER3_MAX_CHARS=300`

**Memory Confidence — veracity signal for every memory**
- New `veracity` field: `stated`, `inferred`, `tool`, `imported`, `unknown`
- `remember(veracity="stated")` — set confidence at write time
- `recall(veracity="stated")` — filter by confidence level
- Recall applies veracity multiplier to scores (stated=1.0x, inferred=0.7x, tool=0.5x)
- `get_contaminated()` — surface non-stated memories for review
- Configurable weights via `MNEMOSYNE_*_WEIGHT` env vars

### Fixed
- `local_llm.summarize()` → `summarize_memories()` — would crash on LLM degradation path
- SQLite connection conflicts in batch degradation tests
- Removed hallucinated Phase 2 from roadmap

## [2.2] — 2026-05-02

### Added

**Cross-Provider Importers — migrate from any memory platform**
- New `mnemosyne/core/importers/` module with 6 provider importers
- **Mem0:** SDK pagination → REST → structured export fallback chain; preserves user/agent/app scoping
- **Letta (MemGPT):** AgentFile `.af` format parsing (JSON/YAML/TOML); memory blocks → working_memory, messages → episodic
- **Zep:** users → sessions → `memory.get()` per-session iteration; messages + summaries + facts extraction
- **Cognee:** `get_graph_data()` nodes/edges extraction; nodes → episodic memories, edges → triples
- **Honcho:** peers → sessions → `context()` + messages; peer identity preserved as author_id
- **SuperMemory:** `documents.list()` + `search.execute()`; container tags mapped to channel_id
- **Agentic importer:** generates ready-to-run Python migration scripts and AI agent instructions for all 6 providers

**CLI: `hermes mnemosyne import` extended**
- `--from <provider>` — import directly from Mem0, Letta, Zep, etc.
- `--list-providers` — show all supported providers with docs links
- `--generate-script` — generate a migration script for any provider
- `--agentic` — output instructions to give your AI agent for extraction
- `--dry-run` — validate and transform without writing

**Plugin tool updated**
- `mnemosyne_import` schema extended with `provider`, `api_key`, `user_id`, `agent_id`, `dry_run`, `channel_id` params

### Changed

- README: added "Migrate from other memory providers" section with examples

## [2.1] — 2026-05-02

### Added

**Multi-Agent Identity Layer**
- New columns `author_id`, `author_type`, `channel_id` on `working_memory` and `episodic_memory` with indexes
- `Mnemosyne(author_id=..., author_type=..., channel_id=...)` constructor params
- `remember()` auto-populates identity columns from session context
- `recall(author_id=..., author_type=..., channel_id=...)` filter params
- `get_stats(author_id=..., author_type=..., channel_id=...)` filter params
- Cross-session channel recall: when `channel_id` is provided, scope expands to include all memories in that channel regardless of session
- MCP server: per-connection instances replace module-level cache; identity via tool args or env vars (`MNEMOSYNE_AUTHOR_ID`, `MNEMOSYNE_AUTHOR_TYPE`, `MNEMOSYNE_CHANNEL_ID`)
- Hermes plugin `_get_memory()` reads identity from environment variables

### Changed
- MCP `_get_instance()` renamed to `_create_instance()` — creates fresh instances per connection
- Episodic memory SELECTs and recall-tracking UPDATEs use dynamic session/channel scope

## [2.0] — 2026-04-29

### Added

**Phase 1: Entity Sketching**
- Regex-based entity extraction (`@mentions`, `#hashtags`, quoted phrases, capitalized sequences)
- Pure-Python Levenshtein distance with O(min) space optimization
- Fuzzy entity matching with prefix/substring bonuses and configurable threshold
- `extract_entities=True` parameter on `remember()` — backward compatible, default False

**Phase 2: Structured Fact Extraction**
- LLM-driven fact extraction via `extract_facts()` and `extract_facts_safe()`
- Graceful fallback chain: remote OpenAI-compatible API → local ctransformers GGUF → skip
- Fact parsing with numbering/bullet cleanup, length filter, cap at 5 facts

**Phase 3: Temporal Recall**
- Exponential decay temporal scoring: `exp(-hours_delta / halflife)`
- `temporal_weight`, `query_time`, `temporal_halflife` parameters on `recall()`
- Environment variable `MNEMOSYNE_TEMPORAL_HALFLIFE_HOURS` for global default
- Temporal boost applied across all recall tiers (working, episodic, entity, fact)

**Phase 4: Configurable Hybrid Scoring**
- User-tunable scoring weights: `vec_weight`, `fts_weight`, `importance_weight`
- `_normalize_weights()` with env var fallback and sensible defaults (50/30/20)
- Per-query weight overrides without global state mutation

**Phase 5: Memory Banks**
- `BankManager` class for named namespace isolation
- Per-bank SQLite files under `banks/<name>/mnemosyne.db`
- Bank operations: create, delete, list, rename, exists check, stats
- `Mnemosyne(bank="work")` constructor parameter
- Bank name validation (alphanumeric + hyphens/underscores, max 64 chars)

**Phase 6: MCP Server**
- Model Context Protocol server with 6 tools
- stdio transport (Claude Desktop, etc.) and SSE transport (web clients)
- Per-bank instance caching
- CLI entry: `mnemosyne mcp`

**Phase 7: Hermes Agent Integration**
- 15 Hermes tools: remember, recall, stats, triple_add, triple_query, sleep, scratchpad_write/read/clear, invalidate, export, update, forget, import, diagnose
- 3 lifecycle hooks: `pre_llm_call` (context injection), `on_session_start`, `post_tool_call`
- AAAK compression for context injection
- Session-aware memory instances

**Phase 8: v2 Differentiation**
- `MemoryStream` — push (callbacks) and pull (iterator) event stream, thread-safe
- `DeltaSync` — checkpoint-based incremental synchronization between instances
- `MemoryCompressor` — dictionary-based, RLE, and semantic compression
- `PatternDetector` — temporal (hour/weekday), content (keyword, co-occurrence), sequence patterns
- `MnemosynePlugin` ABC with 4 lifecycle hooks
- `PluginManager` with auto-discovery from `~/.hermes/mnemosyne/plugins/`
- 3 built-in plugins: `LoggingPlugin`, `MetricsPlugin`, `FilterPlugin`

### Changed

- **CLI rewritten** — all commands now use v2 `Mnemosyne`/`BeamMemory` instead of stale v1 `MnemosyneCore`
- **SQLite WAL mode** — both `memory.py` and `beam.py` now use WAL journal mode with 5s busy timeout for better concurrency
- **FastEmbed cache** — model cache persists at `~/.hermes/cache/fastembed` instead of ephemeral `/tmp`
- **Legacy dual-write** — uses `INSERT OR REPLACE` for dedup safety

### Fixed

- `cli.py` DATA_DIR hardcoded to stale v1 path — now uses `MNEMOSYNE_DATA_DIR` env var
- Duplicate `_recency_decay()` definitions in `beam.py` merged into single function
- SQLite concurrency test failures — WAL mode + proper tearDown cleanup
- `plugin.yaml` declared only 9 of 15 tools — now declares all 15

### Tests

- 292 tests passing (up from unknown baseline)
- New test files: `test_entities.py`, `test_entity_integration.py`, `test_banks.py`, `test_mcp_tools.py`, `test_streaming.py`, `test_temporal_recall.py`
- All test tearDown methods handle WAL `-wal`/`-shm` files

---

## [1.13] — 2026-04-28

### Added

- **Temporal queries** — query the knowledge graph with time awareness (`temporal_halflife`, `temporal_weight`)
- **Memory bank isolation** — separate namespaces for different projects or contexts
- **Configurable hybrid scoring** — tune vector vs. FTS vs. importance weights per query
- **PII-safe diagnostic tool** (`mnemosyne_diagnose`) — inspect your memory without exposing sensitive data

### Fixed

- `sqlite-vec` LIMIT parameter handling
- Triples module-level helpers
- Embeddings fallback when `sqlite-vec` is absent
- Memory embeddings table auto-creation for sqlite-vec fallback

---

## [1.12] — 2026-04-26

### Added

- **Feature comparison matrix** vs. cloud providers (Honcho, Zep, Mem0, Hindsight)
- **DevOps policy** — comprehensive procedures for releases, security, and operations

### Changed

- Documentation cleanup — replaced placeholder files with proper repo docs

---

## [1.11] — 2026-04-25

### Added

- **Token-aware batch sizing** in consolidation — no more OOM on large memory sets
- **Remote API support** for LLM summarization in `sleep()`

### Fixed

- Consolidation edge cases with mixed local/remote LLM configs

---

## [1.10] — 2026-04-24

### Added

- **`mnemosyne_update` tool** — modify existing memories without full replacement
- **`mnemosyne_forget` tool** — targeted memory deletion
- **Global stats flag** — `hermes mnemosyne stats --global` for workspace-wide metrics

### Fixed

- Working memory scope handling across sessions (PR #11)
- Default scope set to 'global' for migrated memories
- Working memory stats and recall tracking consistency

---

## [1.9] — 2026-04-23

### Added

- **PyPI release** — `pip install mnemosyne-memory` works out of the box
- **CI/CD pipeline** — GitHub Actions for testing and release automation
- **`pyproject.toml`** — modern Python packaging
- **UPDATING.md** — migration guide for existing users

### Fixed

- Plugin `register()` export for Hermes plugin loader discovery
- Cross-session recall inconsistency (Issue #7, Bug 2)
- Subagent context write blocking (PR #8)

---

## [1.8] — 2026-04-22

### Added

- **Plugin auto-discovery** — `register()` method for Hermes plugin CLI
- **Bug report template** — official GitHub issue template

### Fixed

- 6 bugs from Issue #6 — edge cases in recall, scope handling, and tool registration

---

## [1.7] — 2026-04-22

### Added

- **PEP 668 PSA** — documentation for Ubuntu 24.04 / Debian 12 users hitting `externally-managed-environment`

### Fixed

- Provider `register_cli` using nested parser instead of subparser
- `sys.path` injection with graceful `ImportError` fallback

---

## [1.6] — 2026-04-21

### Added

- **Feature request template** — GitHub issue template for enhancements
- **Simple versioning** adopted — MAJOR.MINOR instead of semver

### Fixed

- `fastembed` dependency correction (was incorrectly listing `sentence-transformers`)
- Benchmarks restored to README with LongMemEval scores

---

## [1.5] — 2026-04-20

### Added

- **Export/import** — cross-machine memory migration (`mnemosyne_export` / `mnemosyne_import`)
- **One-command installer** — `curl | bash` setup for new users
- **MemoryProvider mode** — deploy Mnemosyne as a standalone memory provider via plugin system
- **Anchored table of contents** in README

### Changed

- README fully rewritten — professional, community-focused, removed bloat
- FluxSpeak branding removed from LICENSE and metadata (Mnemosyne is its own thing)

---

## [1.4] — 2026-04-19

### Added

- **Temporal validity** — memories can have expiration dates
- **Global scope** — memories visible across all sessions
- **Local LLM-based sleep()** — summarization without cloud APIs
- **Recall tracking** — knows what you already remembered
- **Recency decay** — older memories naturally fade in relevance

### Fixed

- Path type bug in memory override skill
- `plugin.yaml` moved to repo root for Hermes compatibility

---

## [1.3] — 2026-04-17

### Added

- **Memory override skill** — bake memory into pre_llm_call and session_start hooks
- **Critical deprecation notice** for legacy memory tool

---

## [1.2] — 2026-04-13

### Added

- **Scale limits** — tested and documented for 1M+ token capacity
- **Legacy DB migration script** — upgrade path from early schemas

### Changed

- Auto-logging of `tool_execution` disabled by default (privacy)

---

## [1.1] — 2026-04-10

### Added

- **BEAM architecture** — sqlite-vec + FTS5 + sleep consolidation
- **BEAM benchmarks** — dedicated benchmark suite with published results
- **Dense retrieval** via fastembed
- **AAAK compression** — compressed memory format for context injection
- **Temporal triples** — structured fact storage with subject/predicate/object

### Fixed

- Thread-local connection bug

---

## [1.0] — 2026-04-05

### Added

- **Initial release** — zero-dependency AI memory system
- **`remember()` / `recall()` / `sleep()`** — core memory cycle
- **SQLite + fastembed embeddings** — local vector search
- **Hermes plugin registration** — basic tool integration
- **AAAK compression** — early context compression for token limits

[2.4]: https://github.com/AxDSan/mnemosyne/releases/tag/v2.4
[2.0]: https://github.com/AxDSan/mnemosyne/releases/tag/v2.0
[1.13]: https://github.com/AxDSan/mnemosyne/releases/tag/v1.13
[1.0]: https://github.com/AxDSan/mnemosyne/releases/tag/v1.0
