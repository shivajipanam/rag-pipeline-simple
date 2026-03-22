"""
RAG pipeline — WhatsApp Chat Search edition.
Uses: HuggingFace embeddings (free) + Pinecone (vector store) + Groq (free Llama 3.1)
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
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "rag-workshop")
GROQ_MODEL = "llama-3.1-8b-instant"

_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_vectorstore(namespace: str) -> PineconeVectorStore:
    return PineconeVectorStore(
        index_name=PINECONE_INDEX,
        embedding=get_embeddings(),
        namespace=namespace,
    )


def ingest_whatsapp(file_bytes: bytes, filename: str, namespace: str) -> dict:
    """
    Parse a WhatsApp .txt export, chunk it, embed it, and store in Pinecone.
    Returns doc_id, chunk count, and chat stats.
    """
    raw_text = file_bytes.decode("utf-8", errors="replace")
    messages = parse_whatsapp_export(raw_text)

    if not messages:
        raise ValueError(
            "No messages could be parsed. Make sure this is a WhatsApp export (.txt)."
        )

    doc_id = str(uuid.uuid4())
    docs = messages_to_documents(messages, filename, doc_id, chunk_size=15)

    PineconeVectorStore.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        index_name=PINECONE_INDEX,
        namespace=namespace,
    )

    stats = get_export_stats(messages)
    return {"doc_id": doc_id, "chunks_indexed": len(docs), "stats": stats}


def ingest_plain_text(file_bytes: bytes, filename: str, namespace: str) -> dict:
    """Fallback ingestion for non-WhatsApp .txt or .pdf files."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    raw_text = file_bytes.decode("utf-8", errors="replace")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc_id = str(uuid.uuid4())
    chunks = splitter.create_documents(
        [raw_text],
        metadatas=[{"doc_id": doc_id, "source_filename": filename}],
    )

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        index_name=PINECONE_INDEX,
        namespace=namespace,
    )

    return {
        "doc_id": doc_id,
        "chunks_indexed": len(chunks),
        "stats": {"total_messages": len(chunks), "senders": {}, "date_range_start": "", "date_range_end": ""},
    }


def _is_whatsapp_export(raw_text: str) -> bool:
    """Heuristic: check if the first few non-empty lines look like a WhatsApp export."""
    from app.parser import _PATTERN_A, _PATTERN_B
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()][:10]
    matches = sum(1 for l in lines if _PATTERN_A.match(l) or _PATTERN_B.match(l))
    return matches >= 2


def ingest_file(file_bytes: bytes, filename: str, namespace: str) -> dict:
    """Route to the right ingestion strategy based on file content."""
    if filename.lower().endswith(".txt"):
        raw = file_bytes.decode("utf-8", errors="replace")
        if _is_whatsapp_export(raw):
            return ingest_whatsapp(file_bytes, filename, namespace)
        else:
            return ingest_plain_text(file_bytes, filename, namespace)
    else:
        # PDF — use plain text ingestion via PyPDF
        import tempfile
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

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
        doc_id = str(uuid.uuid4())
        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["source_filename"] = filename

        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            index_name=PINECONE_INDEX,
            namespace=namespace,
        )

        return {
            "doc_id": doc_id,
            "chunks_indexed": len(chunks),
            "stats": {"total_messages": len(chunks), "senders": {}, "date_range_start": "", "date_range_end": ""},
        }


def query_rag(
    question: str,
    namespace: str,
    history: list[dict] | None = None,
    top_k: int = 6,
) -> dict:
    """
    Retrieve relevant chunks from Pinecone and generate an answer with Groq.
    Includes conversation history for multi-turn chat.
    Returns answer, and rich source list [{sender, date, snippet}].
    """
    vectorstore = get_vectorstore(namespace)
    results = vectorstore.similarity_search(question, k=top_k)

    if not results:
        return {
            "answer": "I couldn't find any relevant messages. Please upload a WhatsApp chat export first.",
            "sources": [],
        }

    # Build context block with metadata so LLM can cite dates/senders
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

    # Build messages list: system + trimmed history + new user message
    messages = [system_msg]
    if history:
        messages.extend(history[-10:])  # last 5 turns
    messages.append({"role": "user", "content": user_content})

    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=768,
    )

    answer = completion.choices[0].message.content.strip()

    # Build rich sources list
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


def extract_to_excel(question: str, namespace: str, top_k: int = 12) -> bytes:
    """
    Retrieve relevant chat chunks and ask Groq to extract structured tabular data.
    Returns Excel file bytes.

    The LLM is instructed to output JSON array of rows; we convert that to .xlsx.
    Example query: "extract all daily sales figures"
    """
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

    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": extraction_prompt}],
        temperature=0.0,
        max_tokens=2048,
    )

    raw_json = completion.choices[0].message.content.strip()

    # Strip markdown code fences if model added them anyway
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
            f"Could not parse structured data from these messages. "
            f"Try a more specific query like 'extract all sales figures with dates'."
        )

    df = pd.DataFrame(rows)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Extracted Data")
        # Auto-fit column widths
        ws = writer.sheets["Extracted Data"]
        for col in ws.columns:
            max_len = max((len(str(cell.value or "")) for cell in col), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 50)

    return buffer.getvalue()
