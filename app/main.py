import os

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app import rag, session_store
from app.models import (
    ChatRequest, ChatResponse, ClearRequest,
    QueryRequest, QueryResponse, Source, UploadResponse,
)

app = FastAPI(title="ChatSearch — Ask Your WhatsApp Chats", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Root → serve frontend ──────────────────────────────────────────────────────

@app.get("/")
def root():
    index_path = os.path.join(_frontend_dir, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"message": "ChatSearch API running. See /docs for endpoints."}


# ── Session ────────────────────────────────────────────────────────────────────

@app.post("/session")
def new_session():
    """Create a fresh session and return its ID."""
    session = session_store.create_session()
    return {"session_id": session.session_id}


# ── Upload ─────────────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    session_id: str = Form(default=None),
):
    allowed = {".pdf", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 20 MB.")

    session = session_store.get_or_create(session_id)

    try:
        result = rag.ingest_file(contents, file.filename, session.namespace)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    session.doc_id = result["doc_id"]
    session.upload_filename = file.filename
    session.upload_stats = result["stats"]
    session_store.clear_history(session)  # fresh chat on new upload

    stats = result["stats"]
    senders_str = ", ".join(stats.get("senders", {}).keys()) or "unknown"
    return UploadResponse(
        session_id=session.session_id,
        doc_id=result["doc_id"],
        chunks_indexed=result["chunks_indexed"],
        message=(
            f"Indexed {stats.get('total_messages', result['chunks_indexed'])} messages "
            f"from {senders_str} "
            f"({stats.get('date_range_start', '')} – {stats.get('date_range_end', '')})"
        ),
        stats=stats,
    )


# ── Chat ───────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session = session_store.get_or_create(request.session_id)

    if not session.doc_id:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded for this session. Upload a chat export first.",
        )

    # Append user message to history
    session_store.append_message(session, "user", request.message)

    result = rag.query_rag(
        question=request.message,
        namespace=session.namespace,
        history=session.history[:-1],  # history before this message
    )

    # Append assistant reply to history
    session_store.append_message(session, "assistant", result["answer"])

    sources = [Source(**s) for s in result["sources"]]
    return ChatResponse(
        session_id=session.session_id,
        answer=result["answer"],
        sources=sources,
    )


@app.post("/chat/clear")
def clear_chat(request: ClearRequest):
    session = session_store.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    session_store.clear_history(session)
    return {"message": "Chat history cleared."}


# ── Export to Excel ────────────────────────────────────────────────────────────

@app.post("/export")
def export_excel(request: ChatRequest):
    """
    Extract structured data from the chat and return as an Excel file.
    Pass the extraction query as `message`, e.g. "extract all daily sales figures".
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Extraction query cannot be empty.")

    session = session_store.get_or_create(request.session_id)

    if not session.doc_id:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded for this session.",
        )

    try:
        excel_bytes = rag.extract_to_excel(request.message, session.namespace)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    filename = f"chat_export_{session.session_id[:8]}.xlsx"
    return Response(
        content=excel_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Legacy /query (kept for backward compat) ─────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = rag.query_rag(request.question, request.namespace)
    return QueryResponse(answer=result["answer"], sources=result["sources"])
