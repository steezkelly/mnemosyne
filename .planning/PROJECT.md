# Mnemosyne Competitive Enhancement Project

## Vision

Transform Mnemosyne from a "reliable dumb layer" into a **competitively complete local-first memory system** that matches or exceeds Hindsight self-hosted capabilities while preserving its core identity: zero-container, in-process, SQLite-native, and ruthlessly simple.

**Core Value:** Prove that sophistication and simplicity are not mutually exclusive — that you can have entity resolution, structured fact extraction, multi-strategy recall, and cross-agent sharing without Docker, PostgreSQL, or 500MB PyTorch dependencies.

**Differentiation Strategy (Anti-Copying):**
- Hindsight uses **PostgreSQL + pgvector + Docker + PyTorch** — Mnemosyne stays **SQLite + sqlite-vec + pip + ONNX**
- Hindsight has **4-way parallel retrieval with rerankers** — Mnemosyne has **single-pass scoring with configurable weights**
- Hindsight does **LLM-driven fact normalization** — Mnemosyne does **raw-text-primary with derived structured views**
- Hindsight exposes **custom HTTP API** — Mnemosyne exposes **MCP (open standard)**
- Hindsight has **background consolidation workers** — Mnemosyne has **on-demand sleep with AAAK fallback**

---

## Constraints

1. **No new heavy dependencies** — No PostgreSQL, no Docker requirement, no PyTorch. ONNX/fastembed is the ceiling.
2. **Preserve SQLite-native identity** — All features must work in single-file SQLite. No external database servers.
3. **Graceful degradation** — Every new feature must fall back to keyword-only mode if optional deps are missing.
4. **Hermes integration first** — New features must integrate cleanly with the existing MemoryProvider ABC.
5. **No breaking changes** — Existing `remember()`/`recall()`/`sleep()` APIs remain unchanged. New features are additive.
6. **Honest documentation** — No fabricated APIs, no inflated benchmarks. Every claim verifiable in code.

---

## v1 Requirements (Must-Have for Competitive Parity)

### REQ-1: Entity Sketching System
Lightweight entity extraction and fuzzy matching without spaCy or heavy NLP libraries.

- Extract entity candidates using regex patterns + optional LLM (when available)
- Store entity mentions as TripleStore triples: `(subject, "mentions", "entity_name")`
- Fuzzy subject matching using SQLite `LIKE` + Levenshtein distance (pure Python)
- Entity co-occurrence tracking via TripleStore queries
- **Acceptance:** Two memories mentioning "Abdias" and "Abdias J" can be linked via entity recall

### REQ-2: Structured Memory Extract (Optional Flag)
LLM-driven fact extraction as a derived layer, not a replacement for raw text.

- `remember(content, extract=True)` — when LLM available, extract 2-5 structured facts
- Store extracted facts in TripleStore alongside raw memory
- Facts are **views**, not replacements — raw text remains canonical
- **Acceptance:** `recall("what does the user like?")` returns both raw memories and structured facts

### REQ-3: Temporal Boost in Recall
Add time-awareness to existing hybrid scoring without separate temporal retrieval pipeline.

- `recall(query, temporal_weight=0.2)` — boost memories with timestamps near query time
- Add `mentioned_at` field to temporal triples (auto-populated from memory timestamp)
- **Acceptance:** Recent memories score higher than old ones when temporal_weight > 0

### REQ-4: MCP Server for Cross-Agent Sharing
Model Context Protocol server for interoperability, not custom HTTP.

- Expose `remember`, `recall`, `sleep`, `scratchpad_read`, `scratchpad_write` as MCP tools
- Support stdio and SSE transports
- Any MCP-compatible client can connect (Claude Desktop, future Hermes, etc.)
- **Acceptance:** Claude Desktop can add Mnemosyne as an MCP tool and call `recall`

### REQ-5: Configurable Hybrid Scoring
Replace hardcoded 50/30/20 weights with user-tunable parameters.

- `recall(query, vec_weight=0.5, fts_weight=0.3, importance_weight=0.2, temporal_weight=0.0)`
- Weights normalized to sum to 1.0
- **Acceptance:** User can prioritize keyword search over semantic: `fts_weight=0.6, vec_weight=0.2`

### REQ-6: Memory Bank Isolation
Support multiple named memory banks within the same SQLite file.

- `Mnemosyne(bank="default")` — each bank is a namespace
- Banks share the same database but have isolated working/episodic tables
- Cross-bank recall is opt-in: `recall(query, banks=["project_a", "project_b"])`
- **Acceptance:** Two agents can use different banks without memory leakage

---

## v2 Requirements (Differentiation / Future-Proofing)

### REQ-7: Streaming Recall
Yield results as they are found, not batched.

- `recall_stream(query)` — generator that yields memories in confidence order
- Early results available without waiting for full hybrid scoring
- **Acceptance:** First result available in <5ms even for 10K corpus

### REQ-8: Memory Embeddings Compression
Aggressive quantization for massive corpora.

- Support `bit` vectors (48 bytes per 384-dim vector) as default instead of `int8`
- Automatic dimensionality reduction via PCA (optional, numpy-only)
- **Acceptance:** 100K memories fit in <100MB SQLite file

### REQ-9: Pattern-Based Consolidation
Rule-based consolidation without LLM dependency.

- Detect repeated patterns in working memory (same source, similar content)
- Auto-merge using AAAK compression when LLM unavailable
- **Acceptance:** 10 identical "reminder" memories consolidate to 1 automatically

### REQ-10: Export/Import with Differential Sync
Efficient cross-machine memory transfer.

- `export_delta(since_timestamp)` — only export memories changed since last sync
- `import_delta(delta_file)` — merge without full replacement
- **Acceptance:** Sync 1000 new memories in <1s

### REQ-11: Memory Health Dashboard
CLI tool for inspecting memory system state.

- `mnemosyne health` — shows stats, fragmentation, duplicate detection
- `mnemosyne doctor` — suggests optimizations, finds orphans
- **Acceptance:** Detects 5+ exact-duplicate memories and suggests consolidation

### REQ-12: Plugin Architecture for Custom Retrievers
Allow users to add custom retrieval strategies.

- Register custom scorer: `register_scorer(name, fn(query, memory) -> score)`
- Custom scorers participate in hybrid ranking
- **Acceptance:** User can add a "recency-only" scorer for time-sensitive queries

---

## Success Criteria

1. **Feature parity:** Every Hindsight self-hosted feature the user asked about has a Mnemosyne equivalent
2. **Performance:** Recall latency stays under 10ms for 10K corpus (measured, not claimed)
3. **Honesty:** Comparison page updated with real benchmarks, no fabricated APIs
4. **Adoption:** At least one Hermes user switches from Hindsight self-hosted to Mnemosyne
5. **No bloat:** Core package stays under 5MB installed size (excluding optional model downloads)

---

## Exclusions (What We Will NOT Do)

1. **No PostgreSQL migration** — SQLite-only forever
2. **No Docker requirement** — pip install remains the deployment model
3. **No PyTorch dependency** — ONNX/fastembed is the ceiling for embeddings
4. **No background workers** — All processing is on-demand or cron-triggered
5. **No cloud sync** — Export/import is the cross-machine model
6. **No custom HTTP API** — MCP is the network interface
7. **No LLM-as-required** — All features work without LLM (degraded but functional)

---

*Document version: 1.0*
*Author: Abdias J (AxDSan)*
*Date: 2026-04-28*
