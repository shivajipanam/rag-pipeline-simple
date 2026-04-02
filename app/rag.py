"""
RAG pipeline — Volo AI (WhatsApp Chat Search).
Uses: HuggingFace embeddings + Pinecone (vector store) + Groq (Llama 3.1)

Memory optimisations applied in this file:
  - _groq_client singleton: Groq() HTTP client created once at first use and
    reused for all subsequent queries.  Previously a new client (~150 KB) was
    instantiated and discarded on every single /chat and /export call.
  - _vectorstore_cache: PineconeVectorStore objects are cached per namespace.
    Previously a new connection object was created on every similarity_search.
    evict_vectorstore() removes an entry so the next call creates a fresh one
    (used after re-upload or session expiry).
  - delete_pinecone_namespace(): called by the session cleanup hook in main.py
    to remove all vectors belonging to an expired session from Pinecone, freeing
    storage and keeping index size bounded.
  - Explicit del of large intermediary objects in ingest_*():
      raw_text  — decoded string (~file size duplicate)
      messages  — list of ChatMessage dataclasses (1.5–11 MB for large exports)
      docs      — list of LangChain Documents (1–5 MB)
    These are freed immediately after they are no longer needed, reducing peak
    RSS during upload rather than waiting for Python's GC cycle.
  - HuggingFace Inference API: if HUGGINGFACE_API_KEY is set, embeddings are
    computed via the HF API (no local model loaded, saves 22–30 MB constant
    memory).  Falls back to the local model if the key is absent.
"""
import io
import json
import os
import uuid

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from groq import Groq

from app.parser import parse_whatsapp_export, messages_to_documents, get_export_stats

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PINECONE_INDEX  = os.getenv("PINECONE_INDEX_NAME", "rag-workshop")
GROQ_MODEL      = "llama-3.1-8b-instant"

# ── Module-level singletons ──────────────────────────────────────────────────
_embeddings = None
_groq_client = None
_vectorstore_cache: dict[str, PineconeVectorStore] = {}


# ── Singleton accessors ──────────────────────────────────────────────────────

def get_embeddings():
    """
    Return the embedding model.

    If HUGGINGFACE_API_KEY is present, use the HuggingFace Inference API so the
    22–30 MB model weights are never loaded into the container.  Falls back to
    the local HuggingFaceEmbeddings if the key is absent or the API import fails.
    """
    global _embeddings
    if _embeddings is None:
        hf_api_key = os.environ.get("HUGGINGFACE_API_KEY")
        if hf_api_key:
            try:
                from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
                _embeddings = HuggingFaceInferenceAPIEmbeddings(
                    api_key=hf_api_key,
                    model_name=EMBEDDING_MODEL,
                )
            except Exception:
                _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        else:
            _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_groq_client() -> Groq:
    """
    Return the module-level Groq HTTP client singleton.
    Creating a new Groq() on every request wastes ~150 KB and re-establishes
    the underlying HTTP connection pool each time.
    """
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    return _groq_client


def get_vectorstore(namespace: str) -> PineconeVectorStore:
    """
    Return a cached PineconeVectorStore for the given namespace.
    Creates a new one on first access; subsequent calls return the same object.
    Call evict_vectorstore(namespace) to force a fresh connection (e.g. after
    re-upload or session expiry).
    """
    if namespace not in _vectorstore_cache:
        _vectorstore_cache[namespace] = PineconeVectorStore(
            index_name=PINECONE_INDEX,
            embedding=get_embeddings(),
            namespace=namespace,
        )
    return _vectorstore_cache[namespace]


def evict_vectorstore(namespace: str) -> None:
    """Remove a namespace from the vectorstore cache."""
    _vectorstore_cache.pop(namespace, None)


def delete_pinecone_namespace(namespace: str) -> None:
    """
    Delete all vectors stored under a Pinecone namespace.
    Called by the session expiry hook in main.py so that expired sessions
    do not leave orphaned vectors in the index indefinitely.
    This is a best-effort operation — errors are silently swallowed.
    """
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))
        index = pc.Index(PINECONE_INDEX)
        index.delete(delete_all=True, namespace=namespace)
    except Exception:
        pass


# ── Ingestion ────────────────────────────────────────────────────────────────

def ingest_whatsapp(file_bytes: bytes, filename: str, namespace: str) -> dict:
    """
    Parse a WhatsApp .txt export, chunk, embed, and store in Pinecone.

    Intermediary objects (raw_text, messages, docs) are explicitly deleted
    after each stage to release peak memory as early as possible.
    """
    raw_text = file_bytes.decode("utf-8", errors="replace")
    messages = parse_whatsapp_export(raw_text)
    del raw_text

    if not messages:
        raise ValueError(
            "No messages could be parsed. Make sure this is a WhatsApp export (.txt)."
        )

    doc_id = str(uuid.uuid4())
    docs = messages_to_documents(messages, filename, doc_id, chunk_size=15)
    stats = get_export_stats(messages)    # compute stats BEFORE deleting messages
    chunks_indexed = len(docs)
    del messages                          # free ChatMessage list

    evict_vectorstore(namespace)
    PineconeVectorStore.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        index_name=PINECONE_INDEX,
        namespace=namespace,
    )
    del docs                              # free Document list after Pinecone ingestion

    return {"doc_id": doc_id, "chunks_indexed": chunks_indexed, "stats": stats}


def ingest_plain_text(file_bytes: bytes, filename: str, namespace: str) -> dict:
    """Fallback ingestion for non-WhatsApp .txt or .pdf files."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    raw_text = file_bytes.decode("utf-8", errors="replace")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc_id = str(uuid.uuid4())
    chunks = splitter.create_documents(
        [raw_text],
        metadatas=[{"doc_id": doc_id, "source_filename": filename}],
    )
    del raw_text                          # free decoded string after chunking

    evict_vectorstore(namespace)
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        index_name=PINECONE_INDEX,
        namespace=namespace,
    )
    n = len(chunks)
    del chunks

    return {
        "doc_id": doc_id,
        "chunks_indexed": n,
        "stats": {"total_messages": n, "senders": {}, "date_range_start": "", "date_range_end": ""},
    }


def _is_whatsapp_export(raw_text: str) -> bool:
    from app.parser import _PATTERN_A, _PATTERN_B
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()][:10]
    matches = sum(1 for l in lines if _PATTERN_A.match(l) or _PATTERN_B.match(l))
    return matches >= 2


def ingest_file(file_bytes: bytes, filename: str, namespace: str) -> dict:
    """Route to the right ingestion strategy based on file content."""
    if filename.lower().endswith(".txt"):
        raw = file_bytes.decode("utf-8", errors="replace")
        is_wa = _is_whatsapp_export(raw)
        del raw
        if is_wa:
            return ingest_whatsapp(file_bytes, filename, namespace)
        else:
            return ingest_plain_text(file_bytes, filename, namespace)
    else:
        # PDF
        import tempfile
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            raw_docs = loader.load()
        finally:
            os.unlink(tmp_path)

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(raw_docs)
        del raw_docs
        doc_id = str(uuid.uuid4())
        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["source_filename"] = filename

        evict_vectorstore(namespace)
        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            index_name=PINECONE_INDEX,
            namespace=namespace,
        )
        n = len(chunks)
        del chunks

        return {
            "doc_id": doc_id,
            "chunks_indexed": n,
            "stats": {"total_messages": n, "senders": {}, "date_range_start": "", "date_range_end": ""},
        }


# ── Query ────────────────────────────────────────────────────────────────────

def query_rag(
    question: str,
    namespace: str,
    history: list[dict] | None = None,
    top_k: int = 6,
) -> dict:
    vectorstore = get_vectorstore(namespace)
    results = vectorstore.similarity_search(question, k=top_k)

    if not results:
        return {
            "answer": "I couldn't find any relevant messages. Please upload a WhatsApp chat export first.",
            "sources": [],
        }

    context_parts = []
    for doc in results:
        meta = doc.metadata
        header = f"[{meta.get('date_start', '')[:10]} → {meta.get('date_end', '')[:10]}] ({meta.get('senders', 'unknown')})"
        context_parts.append(f"{header}\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant that answers questions about WhatsApp chat conversations. "
            "Use ONLY the chat excerpts provided in the user message as context. "
            "When relevant, mention the sender's name and approximate date in your answer. "
            "If the answer is not in the context, say so clearly."
        ),
    }

    user_content = f"Chat context:\n{context}\n\nQuestion: {question}"
    messages = [system_msg]
    if history:
        messages.extend(history[-10:])
    messages.append({"role": "user", "content": user_content})

    completion = get_groq_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=768,
    )

    answer = completion.choices[0].message.content.strip()

    sources = []
    seen = set()
    for doc in results:
        meta = doc.metadata
        key = meta.get("chunk_index", "")
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "sender": meta.get("senders", "unknown"),
            "date_start": meta.get("date_start", "")[:10],
            "date_end": meta.get("date_end", "")[:10],
            "snippet": doc.page_content[:300],
        })

    return {"answer": answer, "sources": sources}


# ── Excel export ─────────────────────────────────────────────────────────────

def extract_to_excel(question: str, namespace: str, top_k: int = 12) -> bytes:
    vectorstore = get_vectorstore(namespace)
    results = vectorstore.similarity_search(question, k=top_k)

    if not results:
        raise ValueError("No relevant messages found to extract data from.")

    context = "\n\n---\n\n".join(doc.page_content for doc in results)

    extraction_prompt = f"""You are a data extraction assistant. Extract ALL structured/tabular information from the chat excerpts below into a JSON array.

Rules:
- Each row should be a JSON object with consistent keys across all rows.
- Include date, sender, and all numerical/categorical values you find.
- For sales/financial data include: date, sender, item/category, amount/value, notes.
- If a field is missing for a row, use null.
- Output ONLY valid JSON — an array of objects. No explanations, no markdown code fences.

Chat excerpts:
{context}

User request: {question}

JSON array:"""

    completion = get_groq_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": extraction_prompt}],
        temperature=0.0,
        max_tokens=2048,
    )

    raw_json = completion.choices[0].message.content.strip()

    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
    raw_json = raw_json.strip()

    try:
        rows = json.loads(raw_json)
        if not isinstance(rows, list):
            rows = [rows]
    except json.JSONDecodeError:
        raise ValueError(
            "Could not parse structured data from these messages. "
            "Try a more specific query like 'extract all sales figures with dates'."
        )

    df = pd.DataFrame(rows)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Extracted Data")
        ws = writer.sheets["Extracted Data"]
        for col in ws.columns:
            max_len = max((len(str(cell.value or "")) for cell in col), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 50)

    return buffer.getvalue()
