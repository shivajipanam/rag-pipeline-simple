# ChatSearch — High Level System Design & Solution Document

> Personal reference document for interviews. Covers architecture, implementation decisions, challenges, learnings, and improvement paths.

---

## 1. What is the Application?

**ChatSearch** is a RAG (Retrieval-Augmented Generation) web application that lets users upload a WhatsApp chat export and then ask questions about it in plain English.

Real-world use cases:
- A restaurant manager uploads their WhatsApp group with suppliers and asks "What did the vegetable supplier say about the delivery last week?"
- Someone uploads a personal chat and asks "When did we last talk about the Paris trip?"
- A business owner uploads customer support chats and asks "Summarise all complaints about delivery in March"

Each answer includes **source context** — the exact sender name, date range, and message snippet that was used to generate the answer.

There is also an **Excel export** feature: users can ask something like "extract all daily sales figures" and download a structured `.xlsx` file built by the LLM from the chat content.

---

## 2. What is RAG?

RAG = **Retrieval-Augmented Generation**. It solves a fundamental LLM limitation: LLMs have a fixed context window and no access to your private data.

Instead of sending the entire chat (which could be millions of characters) to the LLM, RAG does this:

```
1. INDEX TIME (upload):
   Raw chat text → parse → chunk → embed → store in vector DB

2. QUERY TIME (chat):
   User question → embed question → find similar chunks → send chunks + question to LLM → answer
```

The LLM only sees the relevant chunks — not the whole chat. This makes it:
- Cost-efficient (fewer tokens)
- Accurate (LLM answers from real evidence)
- Scalable (works for chats with 100,000 messages)

---

## 3. System Architecture

```
User Browser
     │
     ▼
[ Frontend — Single HTML/CSS/JS file ]
     │
     ├── POST /session     → get session ID
     ├── POST /upload      → upload chat file
     ├── POST /chat        → send question, get answer + sources
     ├── POST /chat/clear  → reset conversation
     └── POST /export      → download Excel file
     │
     ▼
[ FastAPI Backend ]
     ├── app/main.py         ← API routes
     ├── app/session_store.py← In-memory session management
     ├── app/rag.py          ← RAG pipeline: ingest + query + export
     ├── app/parser.py       ← WhatsApp .txt format parser
     ├── app/models.py       ← Pydantic schemas
     └── run.py              ← Entry point, reads PORT from env
          │
          ├── Pinecone (external vector DB)
          │       └── Stores embeddings, namespaced per session
          │
          └── Groq API (external LLM)
                  └── Llama 3.1 8B Instant — generates answers
```

**External services required:**
- **Pinecone** — free tier, stores chat embeddings per session namespace
- **Groq** — free tier, runs Llama 3.1 8B to generate answers

---

## 4. Data Flow — Upload

1. User selects a `.txt` WhatsApp export and clicks Upload
2. Frontend sends `POST /upload` as `multipart/form-data` with the file + session_id
3. `app/main.py` validates: only `.txt` or `.pdf`, max 20MB
4. `session_store.get_or_create(session_id)` — creates or retrieves the session
5. `rag.ingest_file()` is called:
   - Detects if it's a WhatsApp export (heuristic: ≥2 lines matching date pattern in first 10 lines)
   - If WhatsApp: `parse_whatsapp_export()` → list of `ChatMessage` objects
   - If plain text/PDF: `RecursiveCharacterTextSplitter` with 500-char chunks
   - `messages_to_documents()` groups messages into 15-message windows → LangChain `Document` objects with metadata (senders, date range, chunk index)
   - `PineconeVectorStore.from_documents()` embeds all chunks with `all-MiniLM-L6-v2` and stores in Pinecone under `namespace=session_id`
6. Session is updated with `doc_id`, `upload_filename`, `upload_stats`
7. Response includes message count, senders list, date range — displayed in the UI stat grid

---

## 5. Data Flow — Chat Query

1. User types "What did Ravi say about the delivery?" and clicks Send
2. Frontend sends `POST /chat` with `{"session_id": "...", "message": "..."}`
3. `app/main.py`:
   - Validates session has an uploaded document
   - Appends user message to session history
   - Calls `rag.query_rag(question, namespace, history)`
4. `app/rag.py`:
   - `PineconeVectorStore.similarity_search(question, k=6)` — embeds the question and retrieves top 6 most similar chunks from Pinecone
   - Builds context string: each chunk prefixed with `[date_start → date_end] (senders)`
   - Builds messages list: system prompt + last 10 history turns + new user message with context
   - Calls Groq API (`llama-3.1-8b-instant`, temp=0.2, max_tokens=768)
   - Extracts source metadata from the retrieved chunks
5. Response includes `answer` + `sources` list
6. Session history is updated with the assistant reply
7. Frontend renders the answer in a chat bubble, with collapsible source cards underneath

---

## 6. Key Components

### 6.1 WhatsApp Parser (`app/parser.py`)

The hardest part of the project. WhatsApp exports come in two formats depending on the phone OS:

**iOS format:**
```
[15/03/2024, 09:45:32] Ravi (Manager): Good morning, sales were 52,000
```

**Android format:**
```
15/03/2024, 09:45 - Ravi (Manager): Good morning, sales were 52,000
```

The parser handles:
- **Both formats** via two regex patterns (`_PATTERN_A`, `_PATTERN_B`)
- **Multiline messages** — continuation lines (no timestamp prefix) are appended to the current message
- **Media placeholders** — `<Media omitted>`, `image omitted`, etc. — marked with `is_media=True`, not dropped
- **Zero-width characters** — WhatsApp inserts invisible Unicode chars (`\u200e`, `\ufeff`) that break regex — stripped with `_ZWC_RE`
- **Multiple date formats** — WhatsApp uses different date formats by locale (`DD/MM/YYYY`, `MM/DD/YYYY`, 2-digit vs 4-digit year)

**Bug I fixed:** The original filter logic:
```python
# BROKEN — incorrectly dropped media messages
messages = [m for m in messages if m.text and m.text != "[media]" or not m.is_media]
```
Due to Python operator precedence, media messages (`is_media=True`, `text="[media]"`) evaluated to `False and False or False` = `False` — they were dropped instead of kept. Fixed to:
```python
messages = [m for m in messages if m.text]
```

### 6.2 Chunking Strategy (`app/parser.py` — `messages_to_documents`)

Rather than splitting text arbitrarily (standard RAG approach), the chunker groups **15 consecutive messages** into one LangChain `Document`. Each document has rich metadata:

```python
{
    "doc_id": "uuid",
    "source_filename": "chat.txt",
    "chunk_index": 0,
    "date_start": "2024-03-15T09:00:01",
    "date_end": "2024-03-15T09:15:22",
    "senders": "Ravi (Manager), Owner",
    "message_count": 14
}
```

This is better than character-based chunking for chat data because:
- Each chunk is a coherent conversation window
- Metadata enables rich source attribution (who said it, when)
- 15 messages per chunk balances context density vs retrieval precision

### 6.3 Session Store (`app/session_store.py`)

Each browser session gets a UUID. That UUID is used as:
- The **session identifier** (returned to the frontend, stored in localStorage)
- The **Pinecone namespace** (isolates each user's vectors from all others)

This means different users' chat data is completely isolated in Pinecone without any authentication layer — just namespace separation.

The session stores:
- `history` — list of `{role, content}` dicts (Groq-compatible format)
- `doc_id` — UUID of the uploaded document
- `upload_filename` and `upload_stats` — shown in the UI

**History trimming:** To avoid hitting Groq's token limit, only the last 20 turns (40 messages) are kept. The system prompt is always at index 0.

**Current limitation:** Sessions are in-memory (`dict[str, Session]`). If the container restarts, all sessions are lost. The user would need to re-upload their file. In production, this should be Redis.

### 6.4 RAG Pipeline (`app/rag.py`)

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface`. This is a 22M parameter model that produces 384-dimensional dense vectors. It's fast, free to run locally, and good enough for semantic search over chat text.

**Vector store:** Pinecone. Each namespace is a logical partition within the same index. The index is configured with `dims=384, metric=cosine`.

**LLM:** Groq running `llama-3.1-8b-instant`. The system prompt instructs the model to:
- Answer only from the provided chat excerpts
- Mention sender names and dates in the answer
- Say clearly when the answer isn't in the context

**Excel export:** A separate Groq call with a different prompt that instructs the model to output only a JSON array of rows. The rows are parsed and converted to `.xlsx` via pandas + openpyxl. Handles cases where the model wraps output in markdown code fences.

### 6.5 LangChain Integration

The app uses LangChain as a glue layer for:
- `HuggingFaceEmbeddings` — wraps sentence-transformers model
- `PineconeVectorStore` — handles upsert and similarity search
- `RecursiveCharacterTextSplitter` — for non-WhatsApp files (PDFs, plain text)
- `Document` — standard container with `page_content` + `metadata`

**Important:** LangChain restructured its packages in v0.2+. Several breaking import changes had to be fixed:

| Old import | New import |
|---|---|
| `langchain.schema.Document` | `langchain_core.documents.Document` |
| `langchain.text_splitter.RecursiveCharacterTextSplitter` | `langchain_text_splitters.RecursiveCharacterTextSplitter` |
| `langchain.document_loaders.PyPDFLoader` | `langchain_community.document_loaders.PyPDFLoader` |

Also: `langchain-huggingface` 1.x introduced a `ModelProfile` import that doesn't exist in `langchain-core` 1.x. Pinned to `langchain-huggingface==0.1.2` which is stable.

---

## 7. Deployment Architecture

```
GitHub (main branch)
       │ push triggers
       ▼
Railway (auto-deploy)
       │ builds using Dockerfile
       ▼
Docker Image (~1.5 GB)
  ├── python:3.11-slim base
  ├── torch (CPU-only — saves 2.7GB vs CUDA build)
  ├── sentence-transformers, langchain-*, pinecone-client, groq
  └── app code + frontend
       │ runs
       ▼
Container
  └── python run.py → uvicorn on $PORT
```

**CPU-only PyTorch:** The Dockerfile installs torch from `https://download.pytorch.org/whl/cpu`. This is the same PyTorch but without CUDA kernels. For embedding generation (inference-only, no training), CPU is sufficient and saves ~2.7GB of image size.

**`Dockerfile.gpu`** is kept as a backup. To switch to GPU/full build: rename it to `Dockerfile` and push. One-line change to upgrade.

**Environment variables required on Railway:**
- `PINECONE_API_KEY`
- `GROQ_API_KEY`
- `PINECONE_INDEX_NAME` (defaults to `rag-workshop`)

---

## 8. Challenges Faced & How I Solved Them

### Challenge 1: Broken LangChain imports after version upgrade
**Symptom:** `ModuleNotFoundError: No module named 'langchain.schema'` on test run
**Root cause:** LangChain split into sub-packages in v0.2. Old imports no longer exist.
**Solution:** Updated three import paths (see table in section 6.5). Also pinned `langchain-huggingface==0.1.2` because 1.x introduced a `ModelProfile` dependency that didn't exist in `langchain-core` at the time.
**Learning:** In AI/ML projects, always pin exact package versions. The LangChain ecosystem moves extremely fast — minor version bumps break imports.

### Challenge 2: Media messages being dropped by the parser
**Symptom:** Test `test_parser_media_omitted` failed — expected `is_media=True` messages to be kept in the list, but they were being filtered out.
**Root cause:** Python operator precedence bug in the filter condition:
```python
# Evaluates as: (m.text and m.text != "[media]") or (not m.is_media)
# For media msg: (True and False) or False = False → DROPPED
messages = [m for m in messages if m.text and m.text != "[media]" or not m.is_media]
```
**Solution:** Simplified to `messages = [m for m in messages if m.text]` — keep everything with any text content, regardless of media flag.

### Challenge 3: Pydantic version conflict between system and user Python paths
**Symptom:** `SystemError: The installed pydantic-core version (2.23.4) is incompatible with the current pydantic version, which requires 2.41.5`
**Root cause:** Two conflicting pydantic installations — one in system Python (`C:\Python312\Lib`), one in user path (`AppData\Roaming\Python312`). The `langsmith` pytest plugin loaded the wrong one.
**Solution:** `pip install --user --upgrade pydantic pydantic-core` to align both paths.

### Challenge 4: LangChain-HuggingFace 1.x incompatible with LangChain-Core
**Symptom:** `ImportError: cannot import name 'ModelProfile' from 'langchain_core.language_models'`
**Root cause:** `langchain-huggingface 1.2.1` references `ModelProfile` which doesn't exist in any released `langchain-core` version yet.
**Solution:** Pinned `langchain-huggingface==0.1.2` which is the last stable version before the breaking change.

### Challenge 5: WhatsApp zero-width characters breaking regex
**Context:** WhatsApp inserts invisible Unicode direction-control characters (`\u200e`, `\u200f`, `\u202a-\u202e`, `\ufeff`) into exported text. These are invisible in text editors but break regex matching.
**Solution:** Strip them before parsing with: `_ZWC_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\ufeff]")` applied to every line.

---

## 9. What I Learned

### About RAG Architecture
- **Chunking strategy matters as much as the LLM.** Character-based chunking breaks mid-sentence and loses context. Message-window chunking for chat data preserves conversational flow and makes source attribution accurate.
- **Namespace isolation in Pinecone is elegant.** Instead of building user authentication and per-user indices, you can give each session a UUID namespace. Zero extra infrastructure, perfect isolation.
- **Retrieval quality determines answer quality.** If the wrong chunks are retrieved, no LLM can give a good answer. Getting embedding model + chunk size right is more important than which LLM you use.
- **Multi-turn conversation needs history trimming.** Passing the full history to the LLM quickly exhausts the context window. Keeping only the last N turns is the practical solution.

### About Production Readiness
- **In-memory sessions don't survive restarts.** The session store is a Python dict. Every Railway deploy (or container crash) loses all active sessions. Redis is the correct solution.
- **File ingestion is slow on first upload.** Embedding 1,000 messages takes ~5-10 seconds. Users need a loading indicator. For production, ingestion should be async with a progress endpoint.
- **Free tier Pinecone has limits.** The free index has a fixed number of vectors. If many users upload large chat files, the index fills up. Need a cleanup strategy (delete namespace when session expires).

### About LangChain
- **LangChain moves fast and breaks things.** Multiple import paths changed between minor versions. For production, pin everything with exact versions and only upgrade intentionally.
- **LangChain is good glue, but adds indirection.** `PineconeVectorStore.from_documents()` hides a lot of complexity (batching, retries). This is good for speed but makes debugging harder when something fails silently.

---

## 10. How to Improve This App

### Short-term (easy wins)
| Improvement | Effort | Impact |
|---|---|---|
| Add loading spinner during file upload | 1 hour | Better UX — upload takes 5-10s |
| Support iMessage exports | 2 hours | Wider user base |
| Add "Upload another file" button to replace current doc | 1 hour | Better flow |
| Show character/message count before uploading | 1 hour | Sets expectations |
| Add suggested prompts specific to the chat content | 2 hours | Helps users get started |

### Medium-term
| Improvement | Effort | Impact |
|---|---|---|
| Replace in-memory sessions with Redis | 1 day | Sessions survive restarts |
| Make ingestion async with a status endpoint | 2 days | No timeout on large files |
| Add Pinecone namespace cleanup on session expiry | 1 day | Prevents index exhaustion |
| Support group chat multi-file upload | 2 days | Useful for businesses |
| Add a "Search by date range" filter | 2 days | Precise historical lookups |

### Long-term / Architecture changes
| Improvement | Effort | Impact |
|---|---|---|
| Add user accounts (email/password) | 1 week | Persistent sessions across devices |
| Store uploaded files in S3 / cloud storage | 1 week | Survive container restarts |
| Switch from Pinecone free to paid for higher capacity | - | Scales to enterprise chat volumes |
| Add support for Telegram, Signal exports | 2 weeks | Major user base expansion |
| Semantic deduplication before indexing | 3 days | Ignore repeated/forwarded messages |
| Add a timeline view — "Show me all messages from March" | 1 week | Visual exploration alongside chat |

---

## 11. Interview Talking Points

**"What is RAG and why did you use it here?"**
> RAG — Retrieval-Augmented Generation — solves the problem of LLMs not having access to your private data. A WhatsApp export can have hundreds of thousands of messages — far too large to fit in an LLM's context window. So instead of sending everything, I first chunk and embed the messages into a vector database. At query time I embed the user's question, find the most similar message chunks using cosine similarity, and send only those chunks to the LLM as context. The LLM answers from real evidence, not from hallucination.

**"Walk me through the upload flow."**
> When a file is uploaded, the parser detects whether it's a WhatsApp export by checking if the first few lines match the timestamp pattern. If it is, it parses each line with regex into structured ChatMessage objects — handling both iOS and Android formats, multiline messages, and media placeholders. Then it groups every 15 consecutive messages into a LangChain Document with rich metadata: who sent them and when. Those documents are embedded using the MiniLM sentence transformer model and stored in Pinecone under a namespace that's unique to that user session.

**"How do you handle multi-turn conversation?"**
> Each session stores a history list of role/content pairs in the OpenAI/Groq format. When a user sends a new message, I append it to history, call the RAG pipeline, then append the assistant reply. I pass the last 10 turns of history in the Groq API call so the LLM has conversational context. History is capped at 20 turns to avoid exceeding the token limit.

**"How is user data isolated between sessions?"**
> Pinecone supports namespaces — logical partitions within a single index. Each session gets a UUID as its namespace. When I ingest a file, it goes into that namespace. When I query, I search only within that namespace. No authentication needed — the namespace UUID is essentially an unguessable token that acts as a session key.

**"What would you do differently?"**
> Three things. First, make ingestion async — right now uploading a large chat file blocks the HTTP request for up to 30 seconds. I'd return immediately with a job ID and let the client poll for completion. Second, replace the in-memory session store with Redis so sessions survive container restarts. Third, add a Pinecone cleanup job that deletes namespaces for expired sessions — the free tier has a vector limit and without cleanup it'll fill up over time.

**"How did you handle the WhatsApp format parsing?"**
> WhatsApp exports in two formats depending on the OS — iOS uses square bracket notation with seconds, Android uses a dash separator without seconds. I wrote two regex patterns and try both on each line. The tricky parts were multiline messages (continuation lines have no timestamp, so they get appended to the current message), media omissions (they show up as `<Media omitted>`), and zero-width Unicode characters that WhatsApp inserts invisibly — those break regex silently and took me a while to debug.

---

## 12. File Structure Reference

```
rag-pipeline-simple/
├── app/
│   ├── __init__.py
│   ├── main.py           ← FastAPI routes: /session, /upload, /chat, /export
│   ├── models.py         ← Pydantic schemas: ChatRequest, ChatResponse, Source, etc.
│   ├── parser.py         ← WhatsApp .txt parser (iOS + Android, multiline, media)
│   ├── rag.py            ← RAG pipeline: ingest, query, Excel export
│   └── session_store.py  ← In-memory session management, Pinecone namespace per session
├── docs/
│   ├── system-design.md  ← This document
│   └── linkedin-post.md  ← LinkedIn post
├── frontend/
│   └── index.html        ← Chat UI: upload card, chat bubbles, source cards, Excel modal
├── tests/
│   └── test_api.py       ← 16 tests, all passing, all external calls mocked
├── Dockerfile            ← CPU-only PyTorch (~1.5GB image)
├── Dockerfile.gpu        ← Full CUDA PyTorch (~7.8GB, for future GPU upgrade)
├── run.py                ← Entry point, reads PORT from env
├── railway.json          ← Railway deployment config
├── requirements.txt      ← Pinned dependencies (tested versions)
├── RAG_pipeline.ipynb    ← Original notebook (reference)
└── .env.example          ← Template for required env vars
```
