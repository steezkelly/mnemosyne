# API Reference

## Module-Level Functions

The primary interface. Import directly from `mnemosyne`:

```python
from mnemosyne import remember, recall, get_stats, forget, update, sleep
from mnemosyne import get_context, scratchpad_write, scratchpad_read, scratchpad_clear
```

### `remember(content, *, source, importance, metadata, valid_until, scope)`

Store a memory. Writes to both BEAM working memory and the legacy table.

```python
remember(
    content="User prefers dark mode",
    source="preference",        # default: "conversation"
    importance=0.9,             # default: 0.5, range: 0.0–1.0
    metadata={"ui": "v2"},     # default: None
    valid_until="2026-12-31",  # default: None (never expires)
    scope="global",            # "session" (default) or "global"
)
# → returns memory_id (str)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `content` | `str` | required | The memory text to store |
| `source` | `str` | `"conversation"` | Origin label (e.g. "preference", "credential") |
| `importance` | `float` | `0.5` | Priority weight (0.0–1.0). Affects hybrid ranking. |
| `metadata` | `dict` | `None` | Arbitrary JSON-serializable metadata |
| `valid_until` | `str` | `None` | ISO 8601 datetime. Memory expires after this time. |
| `scope` | `str` | `"session"` | `"session"` = current session only, `"global"` = all sessions |

---

### `recall(query, *, top_k)`

Search memories using hybrid retrieval (vector + FTS5 + importance).

```python
results = recall("editor preferences", top_k=5)
for r in results:
    print(r["content"], r["score"])
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Search query text |
| `top_k` | `int` | `5` | Number of results to return |

**Returns:** `List[Dict]` — each dict contains `content`, `score`, `source`, `importance`, and other metadata.

---

### `forget(memory_id)`

Delete a memory by ID. Removes from both legacy and BEAM working memory.

```python
forget("a1b2c3d4e5f67890")
# → returns True if deleted, False if not found
```

---

### `update(memory_id, *, content, importance)`

Update an existing memory's content or importance.

```python
update("a1b2c3d4e5f67890", content="Updated preference", importance=0.7)
# → returns True if updated, False if not found
```

---

### `get_stats()`

Return memory system statistics.

```python
stats = get_stats()
# {
#   "total_memories": 42,
#   "total_sessions": 3,
#   "sources": {"preference": 10, "conversation": 32},
#   "mode": "beam",
#   "beam": {
#     "working_memory": {"count": 15, ...},
#     "episodic_memory": {"count": 8, ...}
#   }
# }
```

---

### `sleep(dry_run=False)`

Run the consolidation sleep cycle. Moves stale working memories into episodic memory via summarization.

```python
result = sleep()
# {"consolidated": 12, "method": "llm", "duration_ms": 340}
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dry_run` | `bool` | `False` | If True, preview what would be consolidated without making changes |

---

### `get_context(limit)`

Get recent memories from working memory for context injection.

```python
context = get_context(limit=10)
```

---

### Scratchpad Functions

```python
scratchpad_write("TODO: fix auth bug")    # → returns entry ID
entries = scratchpad_read()                # → List[Dict]
scratchpad_clear()                         # clears all entries
```

---

## Mnemosyne Class

For multi-session or custom database path usage:

```python
from mnemosyne import Mnemosyne

m = Mnemosyne(session_id="project-alpha", db_path="/path/to/custom.db")
m.remember("Project uses React 19", source="tech_stack", importance=0.8)
results = m.recall("frontend framework", top_k=3)
m.sleep()
```

### Constructor

```python
Mnemosyne(session_id="default", db_path=None)
```

### Methods

| Method | Signature | Description |
|---|---|---|
| `remember` | `(content, source, importance, metadata, valid_until, scope) → str` | Store a memory |
| `recall` | `(query, top_k=5) → List[Dict]` | Search memories |
| `get_context` | `(limit=10) → List[Dict]` | Get recent working memory entries |
| `get_stats` | `() → Dict` | Get statistics |
| `forget` | `(memory_id) → bool` | Delete a memory |
| `update` | `(memory_id, content, importance) → bool` | Update a memory |
| `invalidate` | `(memory_id, replacement_id=None) → bool` | Mark a memory as superseded |
| `sleep` | `(dry_run=False) → Dict` | Run consolidation |
| `scratchpad_write` | `(content) → str` | Write to scratchpad |
| `scratchpad_read` | `() → List[Dict]` | Read scratchpad |
| `scratchpad_clear` | `() → None` | Clear scratchpad |
| `consolidation_log` | `(limit=10) → List[Dict]` | View consolidation history |
| `export_to_file` | `(output_path) → Dict` | Export all data to JSON |
| `import_from_file` | `(input_path, force=False) → Dict` | Import data from JSON |

---

## BeamMemory Class

Direct access to the BEAM tier. Useful for advanced use cases:

```python
from mnemosyne.core.beam import BeamMemory

beam = BeamMemory(session_id="my_session")

# Working memory
beam.remember("Important context", importance=0.9)

# Episodic memory (manual insert)
beam.consolidate_to_episodic(
    summary="User likes Neovim",
    source_wm_ids=["wm1"],
    importance=0.8
)

# Search across tiers
results = beam.recall("editor preferences", top_k=5)

# Stats
wm_stats = beam.get_working_stats()
ep_stats = beam.get_episodic_stats()
```

---

## TripleStore Class

Temporal knowledge graph:

```python
from mnemosyne.core.triples import TripleStore

kg = TripleStore()

# Add a triple (auto-invalidates previous (subject, predicate) pair)
triple_id = kg.add(
    subject="Maya",
    predicate="assigned_to",
    object="auth-migration",
    valid_from="2026-01-15",       # default: today
    source="project_manager",       # default: "inferred"
    confidence=0.95                 # default: 1.0
)

# Query triples
results = kg.query(subject="Maya", as_of="2026-02-01")

# Invalidate a triple
kg.invalidate(triple_id)

# Export all triples
all_triples = kg.export_all()
```

### Methods

| Method | Description |
|---|---|
| `add(subject, predicate, object, valid_from, source, confidence) → int` | Add a temporal triple |
| `query(subject, predicate, object, as_of) → List[Dict]` | Query triples, optionally at a point in time |
| `invalidate(triple_id) → bool` | Mark a triple as no longer valid |
| `export_all() → List[Dict]` | Export all triples |
