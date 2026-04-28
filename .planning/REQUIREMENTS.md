# Requirements Document

## v1 Requirements (Competitive Parity)

### REQ-1: Entity Sketching System
**Priority:** P0
**Phase:** 1
**Status:** Pending

**Description:**
Lightweight entity extraction and fuzzy matching without spaCy or heavy NLP libraries. Extract entity candidates using regex patterns + optional LLM, store as TripleStore triples, enable fuzzy subject matching.

**Acceptance Criteria:**
- [ ] `remember("I met Abdias yesterday")` creates triple `(memory_id, "mentions", "Abdias")`
- [ ] `recall("Abdias")` returns memories mentioning "Abdias J" via fuzzy match
- [ ] Entity co-occurrence query: `triples.query(subject="*", predicate="mentions", object="Abdias")` returns all mentions
- [ ] Works without LLM (regex fallback), enhanced with LLM (better extraction)
- [ ] Pure Python implementation — no spaCy, no heavy NLP deps

**Dependencies:** TripleStore (existing), regex (stdlib), optional LLM
**Effort:** Medium (2-3 sessions)

---

### REQ-2: Structured Memory Extract
**Priority:** P0
**Phase:** 2
**Status:** Pending

**Description:**
Optional LLM-driven fact extraction as a derived layer, not replacement for raw text. When `extract=True`, extract 2-5 structured facts and store in TripleStore alongside raw memory.

**Acceptance Criteria:**
- [ ] `remember(content, extract=True)` extracts facts when LLM available
- [ ] Extracted facts stored as triples: `(memory_id, "fact", "extracted_fact_text")`
- [ ] `recall("what does the user like?")` searches both raw memories AND extracted facts
- [ ] Raw text remains canonical — facts are views, not replacements
- [ ] Graceful fallback: if LLM unavailable, stores empty fact set, no error

**Dependencies:** REQ-1 (Entity Sketching), LLM (optional)
**Effort:** Medium (2-3 sessions)

---

### REQ-3: Temporal Boost in Recall
**Priority:** P0
**Phase:** 3
**Status:** Pending

**Description:**
Add time-awareness to existing hybrid scoring. Boost memories with timestamps near query time without separate temporal retrieval pipeline.

**Acceptance Criteria:**
- [ ] `recall(query, temporal_weight=0.2)` accepts temporal weight parameter
- [ ] Memories with `timestamp` closer to `datetime.now()` get boost
- [ ] Temporal boost formula: `score *= (1.0 + temporal_weight * recency_decay)`
- [ ] Backward compatible: `recall(query)` works exactly as before
- [ ] Works with all vector types (float32, int8, bit) and FTS5 fallback

**Dependencies:** Existing BEAM architecture
**Effort:** Low (1-2 sessions)

---

### REQ-4: MCP Server for Cross-Agent Sharing
**Priority:** P0
**Phase:** 6
**Status:** Pending

**Description:**
Model Context Protocol server exposing Mnemosyne tools for cross-agent interoperability. Supports stdio and SSE transports.

**Acceptance Criteria:**
- [ ] `mnemosyne mcp` command starts MCP server
- [ ] Exposes tools: `remember`, `recall`, `sleep`, `scratchpad_read`, `scratchpad_write`
- [ ] Stdio transport works (for Claude Desktop, etc.)
- [ ] SSE transport works (for web clients)
- [ ] Tool schemas match MCP specification
- [ ] Any MCP client can discover and call Mnemosyne tools

**Dependencies:** All v1 features (needs stable API surface)
**Effort:** High (3-4 sessions)

---

### REQ-5: Configurable Hybrid Scoring
**Priority:** P0
**Phase:** 4
**Status:** Pending

**Description:**
Replace hardcoded 50/30/20 weights with user-tunable parameters. Weights normalized to sum to 1.0.

**Acceptance Criteria:**
- [ ] `recall(query, vec_weight=0.5, fts_weight=0.3, importance_weight=0.2, temporal_weight=0.0)`
- [ ] Weights automatically normalized if they don't sum to 1.0
- [ ] Invalid weights (negative, all zero) raise ValueError with helpful message
- [ ] Backward compatible: default weights match current behavior
- [ ] Documented in API reference with examples

**Dependencies:** REQ-3 (Temporal Boost)
**Effort:** Low (1 session)

---

### REQ-6: Memory Bank Isolation
**Priority:** P0
**Phase:** 5
**Status:** Pending

**Description:**
Support multiple named memory banks within the same SQLite file. Each bank is a namespace with isolated tables.

**Acceptance Criteria:**
- [ ] `Mnemosyne(bank="project_a")` creates/isolates bank namespace
- [ ] Banks share SQLite file but have isolated working/episodic tables
- [ ] `recall(query, banks=["project_a", "project_b"])` for cross-bank search
- [ ] Default bank is `"default"` — backward compatible
- [ ] Bank list/discovery API: `list_banks()`
- [ ] Bank deletion: `delete_bank("project_a")` with confirmation

**Dependencies:** Existing BEAM schema
**Effort:** Medium (2 sessions)

---

## v2 Requirements (Differentiation)

### REQ-7: Streaming Recall
**Priority:** P1
**Phase:** 8
**Status:** Pending

**Description:**
Yield results as they are found via generator, not batched. Early results available without waiting for full hybrid scoring.

**Acceptance Criteria:**
- [ ] `recall_stream(query)` returns generator yielding `(memory, score)` tuples
- [ ] First result available in <5ms even for 10K corpus
- [ ] Results yielded in confidence order (highest first)
- [ ] Consumer can stop early: `next(recall_stream(query))` for top-1
- [ ] Memory-efficient: doesn't hold full result set in RAM

**Dependencies:** REQ-5 (Configurable Scoring)
**Effort:** Medium (2 sessions)

---

### REQ-8: Memory Embeddings Compression
**Priority:** P1
**Phase:** 8
**Status:** Pending

**Description:**
Aggressive quantization for massive corpora. Support `bit` vectors as default, optional PCA dimensionality reduction.

**Acceptance Criteria:**
- [ ] `bit` vectors work as default (48 bytes per 384-dim vector)
- [ ] `VEC_TYPE=bit` env var sets default at initialization
- [ ] Optional PCA reduction: `embedding_dim=128` via config
- [ ] 100K memories fit in <100MB SQLite file
- [ ] Recall quality degradation <5% vs float32 (measured)

**Dependencies:** Existing sqlite-vec integration
**Effort:** Medium (2 sessions)

---

### REQ-9: Pattern-Based Consolidation
**Priority:** P1
**Phase:** 8
**Status:** Pending

**Description:**
Rule-based consolidation without LLM dependency. Detect repeated patterns, auto-merge using AAAK compression.

**Acceptance Criteria:**
- [ ] Detect 5+ memories with >80% text similarity in same source
- [ ] Auto-merge using AAAK compression when LLM unavailable
- [ ] Configurable threshold: `consolidation_similarity_threshold=0.8`
- [ ] Manual trigger: `consolidate_patterns()`
- [ ] Auto-trigger on sleep if pattern count exceeds threshold

**Dependencies:** AAAK compression (existing)
**Effort:** Medium (2 sessions)

---

### REQ-10: Export/Import with Differential Sync
**Priority:** P1
**Phase:** 8
**Status:** Pending

**Description:**
Efficient cross-machine memory transfer. Export only changed memories since last sync.

**Acceptance Criteria:**
- [ ] `export_delta(since_timestamp)` exports only memories with `created_at > since`
- [ ] `import_delta(delta_file)` merges without full replacement
- [ ] Conflict resolution: newer timestamp wins
- [ ] Sync 1000 new memories in <1s
- [ ] JSONL format for streaming compatibility

**Dependencies:** Existing export/import
**Effort:** Low-Medium (1-2 sessions)

---

### REQ-11: Memory Health Dashboard
**Priority:** P1
**Phase:** 8
**Status:** Pending

**Description:**
CLI tool for inspecting memory system state, detecting issues, suggesting optimizations.

**Acceptance Criteria:**
- [ ] `mnemosyne health` shows: total memories, bank sizes, duplicate count, fragmentation
- [ ] `mnemosyne doctor` suggests: consolidate duplicates, vacuum SQLite, adjust weights
- [ ] Detects exact-duplicate memories (>5 triggers suggestion)
- [ ] Detects orphaned embeddings (embedding without memory)
- [ ] Exit code 0 = healthy, 1 = warnings, 2 = critical issues

**Dependencies:** All v1 features
**Effort:** Low (1-2 sessions)

---

### REQ-12: Plugin Architecture for Custom Retrievers
**Priority:** P1
**Phase:** 8
**Status:** Pending

**Description:**
Allow users to register custom retrieval strategies that participate in hybrid ranking.

**Acceptance Criteria:**
- [ ] `register_scorer(name, fn(query, memory) -> float)` API
- [ ] Custom scorers participate in hybrid ranking with configurable weight
- [ ] Example scorers: recency-only, source-priority, entity-density
- [ ] Scorer discovery: `list_scorers()`
- [ ] Documentation with example custom scorer implementation

**Dependencies:** REQ-5 (Configurable Scoring)
**Effort:** Medium (2 sessions)

---

## Requirement Traceability Matrix

| Requirement | Phase | Status | Test File | Docs Page |
|---|---|---|---|---|
| REQ-1 | 1 | Pending | TBD | TBD |
| REQ-2 | 2 | Pending | TBD | TBD |
| REQ-3 | 3 | Pending | TBD | TBD |
| REQ-4 | 6 | Pending | TBD | TBD |
| REQ-5 | 4 | Pending | TBD | TBD |
| REQ-6 | 5 | Pending | TBD | TBD |
| REQ-7 | 8 | Pending | TBD | TBD |
| REQ-8 | 8 | Pending | TBD | TBD |
| REQ-9 | 8 | Pending | TBD | TBD |
| REQ-10 | 8 | Pending | TBD | TBD |
| REQ-11 | 8 | Pending | TBD | TBD |
| REQ-12 | 8 | Pending | TBD | TBD |

---

*Document version: 1.0*
*Date: 2026-04-28*
