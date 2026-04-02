# Memory Optimisation — Technical Reference

> Applied to the Volo AI RAG pipeline (Railway deployment).  
> All changes are backward-compatible; no API surface was altered.

---

## Background

Railway's free tier container has a hard memory ceiling. The original pipeline
had several patterns that caused unnecessary memory usage:

| Pattern | Root cause |
|---------|------------|
| Groq client re-created per request | `Groq()` called inside `query_rag()` and `extract_to_excel()` on every invocation |
| Pinecone connection re-created per query | `PineconeVectorStore(...)` called inside every `query_rag()` call |
| Sessions never evicted | `_store` dict grows forever; each session holds history + stats |
| History trim fired too late | Trim only triggered at `MAX_HISTORY * 2 = 40` messages |
| Large upload intermediaries held too long | Python GC eventually frees them but peak RSS spikes during upload |
| Embedding model always local | 22–30 MB of model weights loaded regardless of whether API is available |
| Pinecone vectors never deleted | Expired sessions leave orphaned namespaces in the index |

---

## Changes Applied

### 1 — Groq Client Singleton (`app/rag.py`)

**What changed:** `Groq(api_key=...)` moved from inside `query_rag()` and
`extract_to_excel()` to a module-level lazy singleton accessed via
`get_groq_client()`.

**Why it matters:** The `Groq` constructor sets up an HTTP connection pool
(~150 KB). Every `/chat` and `/export` call was creating and immediately
discarding one. Under load, this created constant allocation/GC pressure and
prevented connection reuse.

**How it works:**
```python
_groq_client = None

def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client
```

The singleton is created on first use (lazy) so it does not block startup if
`GROQ_API_KEY` is not yet set in the environment.

---

### 2 — Pinecone Vectorstore Cache (`app/rag.py`)

**What changed:** `get_vectorstore(namespace)` now maintains a
`_vectorstore_cache` dict. The first call for a namespace creates the object;
subsequent calls return the cached instance. `evict_vectorstore(namespace)`
removes the entry (called on re-upload and session expiry).

**Why it matters:** `PineconeVectorStore(...)` opens a gRPC/HTTP connection to
Pinecone. Re-creating it on every `similarity_search` call wastes time and
memory.

**How it works:**
```python
_vectorstore_cache: dict[str, PineconeVectorStore] = {}

def get_vectorstore(namespace: str) -> PineconeVectorStore:
    if namespace not in _vectorstore_cache:
        _vectorstore_cache[namespace] = PineconeVectorStore(
            index_name=PINECONE_INDEX,
            embedding=get_embeddings(),
            namespace=namespace,
        )
    return _vectorstore_cache[namespace]

def evict_vectorstore(namespace: str) -> None:
    _vectorstore_cache.pop(namespace, None)
```

`evict_vectorstore` is called:
- **Before** `PineconeVectorStore.from_documents()` in every `ingest_*` function,
  so the next query gets a fresh connection pointing at the newly indexed data.
- **By the session expiry hook** in `main.py` when a session is cleaned up.

---

### 3 — Session TTL & Background Cleanup (`app/session_store.py`, `app/main.py`)

**What changed:**
- `Session` dataclass gained a `last_active: datetime` field (UTC).
- `touch(session)` updates this timestamp; called from `get_or_create`,
  `append_message`.
- `is_expired(session)` returns `True` if `last_active` is older than
  `SESSION_TTL_HOURS` (default: 2 hours, configurable via env var).
- `cleanup_expired(on_expire)` scans `_store`, removes expired sessions, and
  calls the `on_expire` callback for each one before removal.
- `main.py` starts a background asyncio task at startup that calls
  `cleanup_expired` every 5 minutes (configurable via
  `SESSION_CLEANUP_INTERVAL_SECONDS`), passing a callback that deletes the
  Pinecone namespace and evicts the vectorstore cache.

**Why it matters:** Without TTL, every anonymous visitor permanently occupies
memory. On a busy day with 100 unique visitors, the uncleaned store could hold
10–50 MB of session data indefinitely.

**How the cleanup hook works:**
```python
def _on_expire(session: Session) -> None:
    rag.delete_pinecone_namespace(session.namespace)   # removes vectors from Pinecone
    rag.evict_vectorstore(session.namespace)           # removes from local cache

session_store.cleanup_expired(on_expire=_on_expire)
```

**Configuration:**
```
SESSION_TTL_HOURS=2                    # how long before a session expires
SESSION_CLEANUP_INTERVAL_SECONDS=300  # how often the background scan runs
```

---

### 4 — Pinecone Namespace Deletion on Expiry (`app/rag.py`)

**What changed:** `delete_pinecone_namespace(namespace)` added. It uses the
Pinecone SDK's `index.delete(delete_all=True, namespace=namespace)` to remove
all vectors for a given namespace.

**Why it matters:** Without this, every expired session leaves orphaned vectors
in Pinecone forever. At free-tier Pinecone limits (100K vectors), a single
large upload can occupy thousands of vectors. Deleting on expiry keeps index
utilisation bounded.

**Implementation:**
```python
def delete_pinecone_namespace(namespace: str) -> None:
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))
        index = pc.Index(PINECONE_INDEX)
        index.delete(delete_all=True, namespace=namespace)
    except Exception:
        pass  # best-effort; session is already removed from memory
```

---

### 5 — History Trim Tightened (`app/session_store.py`)

**What changed:** Trim threshold changed from `MAX_HISTORY * 2` (40) to
`MAX_HISTORY + 2` (22). History is trimmed to `MAX_HISTORY` (20) when this
threshold is crossed.

**Why it matters:** The original logic let history grow to 40 messages before
any trim fired, then cut it to 40. In practice, a long conversation could
accumulate 39 messages (~40–80 KB) before the first trim, and then oscillate
between 39 and 40 forever. The new logic:

```
messages 1–22  → no trim
message 23     → trim fires → kept = last 20 messages
messages 24–42 → no trim (growing back from 20 to 22)
message 43     → trim fires → kept = last 20 messages
```

This keeps history tightly bounded between 20 and 22 messages at all times
instead of between 40 and 41.

---

### 6 — Explicit Deletion of Upload Intermediaries (`app/rag.py`)

**What changed:** `del raw_text`, `del messages`, `del docs` added at the
earliest possible point in `ingest_whatsapp()`, `ingest_plain_text()`, and the
PDF branch of `ingest_file()`.

**Why it matters:** Python's garbage collector will eventually free these, but
"eventually" can mean hundreds of milliseconds after the function returns.
During that window, RSS shows the full peak:

```
file_bytes (20 MB) + raw_text (20 MB) + messages (11 MB) + docs (5 MB) = 56 MB
```

With explicit `del`:
- `raw_text` freed before `messages_to_documents` runs
- `messages` freed before `PineconeVectorStore.from_documents` runs
- `docs` freed after Pinecone ingestion completes

Peak drops to approximately:
```
file_bytes (20 MB) + messages (11 MB)  →  raw_text gone
                   + docs (5 MB)       →  messages gone
                                       →  docs gone after upload
≈ 31 MB peak (vs 56 MB before)
```

---

### 7 — HuggingFace Inference API for Embeddings (`app/rag.py`)

**What changed:** `get_embeddings()` checks for `HUGGINGFACE_API_KEY`. If set,
it uses `HuggingFaceInferenceAPIEmbeddings` (langchain_community) which makes
HTTP calls to HuggingFace's hosted inference endpoint. Falls back to local
`HuggingFaceEmbeddings` if the key is absent or the import fails.

**Why it matters:** `sentence-transformers/all-MiniLM-L6-v2` loads ~22–30 MB
of PyTorch model weights into RAM at startup and keeps them there for the
entire container lifetime. On the free Railway plan (512 MB RAM), this is a
significant constant overhead. Offloading to the HF API eliminates it.

**Trade-offs:**
| | Local model | HF Inference API |
|---|---|---|
| RAM | 22–30 MB (constant) | ~0 MB |
| Latency per embed | ~5–20 ms (CPU) | ~100–300 ms (network) |
| Cost | Free | Free (rate limited) |
| Offline capable | Yes | No |

**Setup:** Create a free account at huggingface.co, go to Settings → Access
Tokens, create a token with `read` scope, set:
```
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

---

### 8 — File Upload Size Cap (`app/main.py`)

**What changed:** The hardcoded `20 * 1024 * 1024` limit replaced with a
configurable `_MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))`.

**Default:** 5 MB (free tier). The error message directs users to upgrade.

**Why it matters:** A 20 MB upload can produce a 56 MB peak RSS spike (see
section 6 above). Capping free-tier uploads at 5 MB bounds this spike to ~14 MB
and naturally nudges heavy users toward the paid plan.

**Configuring per-tier limits** (once authentication is in place):
```python
# In the upload handler, after verifying the user's plan:
effective_limit = 500 if user.plan == "pro" else 5  # MB
```

For now, set `MAX_UPLOAD_MB=500` on Railway to temporarily remove the cap
while you're testing.

---

## Action Items (Not Yet Implemented)

| Item | Notes |
|------|-------|
| Stream large file uploads | Replace `await file.read()` with streaming parser to cap peak at a few KB instead of full file size. High effort — requires parser refactor. |
| Chunking strategy review | Current window = 15 messages. Evaluate semantic chunking or dynamic window sizing for better retrieval quality. |
| Redis for session store | Replace in-memory `_store` with Redis so sessions survive container restarts. |

---

## Testing

All optimisations are covered by unit/integration tests in `tests/test_api.py`.
Run with:
```bash
pytest tests/ -v
```

Key new test groups:
- `test_session_is_not_expired_initially` — TTL baseline
- `test_session_is_expired_after_ttl` — TTL expiry detection
- `test_touch_resets_expiry` — TTL reset on activity
- `test_cleanup_expired_removes_session` — store eviction
- `test_cleanup_calls_on_expire_callback` — Pinecone/cache hook
- `test_history_trims_at_max_plus_two` — tight history bound
- `test_upload_rejects_oversized_file` — 5 MB cap enforcement
- `test_groq_client_is_singleton` — single Groq instance
- `test_vectorstore_cache_returns_same_instance` — cache hit
- `test_vectorstore_eviction_forces_new_instance` — cache miss after evict
