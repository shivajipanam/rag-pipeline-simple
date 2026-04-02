import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

from app import rag, session_store
from app.parser import preprocess_zip
from app.models import (
    ChatRequest, ChatResponse, ClearRequest,
    QueryRequest, QueryResponse, Source, UploadResponse,
)

# ── Upload size cap ───────────────────────────────────────────────────────────
# Free tier default: 5 MB. Override via MAX_UPLOAD_MB env var (e.g. set to 500
# for Pro users once authentication is in place).
_MAX_UPLOAD_MB    = int(os.getenv("MAX_UPLOAD_MB", "5"))
_MAX_UPLOAD_BYTES = _MAX_UPLOAD_MB * 1024 * 1024

# ── Background cleanup ────────────────────────────────────────────────────────
_CLEANUP_INTERVAL = int(os.getenv("SESSION_CLEANUP_INTERVAL_SECONDS", "300"))  # 5 min


async def _session_cleanup_loop() -> None:
    """
    Background task: every _CLEANUP_INTERVAL seconds, evict expired sessions
    and clean up their Pinecone namespaces + vectorstore cache entries.
    """
    while True:
        try:
            await asyncio.sleep(_CLEANUP_INTERVAL)

            def _on_expire(session: session_store.Session) -> None:
                rag.delete_pinecone_namespace(session.namespace)
                rag.evict_vectorstore(session.namespace)

            session_store.cleanup_expired(on_expire=_on_expire)
        except asyncio.CancelledError:
            break
        except Exception:
            pass  # never let cleanup crash the server


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_session_cleanup_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Volo AI — Ask Your Business Chats",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    index_path = os.path.join(_frontend_dir, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"message": "Volo AI API running. See /docs for endpoints."}


# ── Session ───────────────────────────────────────────────────────────────────
@app.post("/session")
def new_session():
    session = session_store.create_session()
    return {"session_id": session.session_id}


# ── Upload ────────────────────────────────────────────────────────────────────
@app.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    session_id: str = Form(default=None),
):
    allowed = {".pdf", ".txt", ".zip"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and ZIP files are supported.")

    contents = await file.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File too large. The free plan supports up to {_MAX_UPLOAD_MB} MB. "
                f"Upgrade to Pro for 500 MB."
            ),
        )

    session = session_store.get_or_create(session_id)

    filename = file.filename
    if ext == ".zip":
        try:
            contents, filename = preprocess_zip(contents)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    try:
        result = rag.ingest_file(contents, filename, session.namespace)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    session.doc_id = result["doc_id"]
    session.upload_filename = file.filename
    session.upload_stats = result["stats"]
    session_store.clear_history(session)

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


# ── Chat ──────────────────────────────────────────────────────────────────────
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

    session_store.append_message(session, "user", request.message)

    result = rag.query_rag(
        question=request.message,
        namespace=session.namespace,
        history=session.history[:-1],
    )

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


# ── Export to Excel ───────────────────────────────────────────────────────────
@app.post("/export")
def export_excel(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Extraction query cannot be empty.")

    session = session_store.get_or_create(request.session_id)

    if not session.doc_id:
        raise HTTPException(status_code=400, detail="No document uploaded for this session.")

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


# ── Contact Sales ─────────────────────────────────────────────────────────────
@app.post("/contact", response_class=PlainTextResponse)
async def contact(
    name:    str = Form(...),
    email:   str = Form(...),
    company: str = Form(default=""),
    message: str = Form(...),
):
    smtp_user = os.getenv("CONTACT_EMAIL_USER")
    smtp_pass = os.getenv("CONTACT_EMAIL_PASS")

    if not smtp_user or not smtp_pass:
        raise HTTPException(
            status_code=503,
            detail="Email service not configured. Please set CONTACT_EMAIL_USER and CONTACT_EMAIL_PASS.",
        )

    to_address = "shivajipanam@gmail.com"
    subject = f"Volo AI — Sales enquiry from {name}" + (f" ({company})" if company else "")
    body = (
        f"New sales enquiry via the Volo AI landing page.\n\n"
        f"Name:    {name}\n"
        f"Email:   {email}\n"
        f"Company: {company or '—'}\n\n"
        f"Message:\n{message}\n"
    )

    msg = MIMEMultipart()
    msg["From"]     = smtp_user
    msg["To"]       = to_address
    msg["Subject"]  = subject
    msg["Reply-To"] = email
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_address, msg.as_string())
    except smtplib.SMTPAuthenticationError:
        raise HTTPException(status_code=502, detail="Email authentication failed.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to send email: {str(e)}")

    return "ok"


# ── Legacy /query ─────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = rag.query_rag(request.question, request.namespace)
    return QueryResponse(answer=result["answer"], sources=result["sources"])
