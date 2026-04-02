"""
Test suite for ChatSearch (WhatsApp RAG app).
Run with: pytest tests/ -v

All external calls (Pinecone, Groq) are mocked — no API keys needed for tests.
"""
import io
import zipfile
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# ── Sample WhatsApp export (both formats) ────────────────────────────────────
SAMPLE_WA_IOS = """\
[15/03/2024, 09:00:01] Ravi (Manager): Good morning sir, yesterday's sales were 52,000
[15/03/2024, 09:00:45] Owner: Good, any issues?
[15/03/2024, 09:01:10] Ravi (Manager): No issues, lunch was very busy. Dinner did 28,000
[16/03/2024, 08:55:00] Ravi (Manager): Morning report: total 61,000 today
[16/03/2024, 08:56:00] Owner: Great, keep it up
"""

SAMPLE_WA_ANDROID = """\
15/03/2024, 09:00 - Ravi (Manager): Good morning sir, yesterday's sales were 52,000
15/03/2024, 09:00 - Owner: Good, any issues?
15/03/2024, 09:01 - Ravi (Manager): No issues. Dinner did 28,000
16/03/2024, 08:55 - Ravi (Manager): Morning report: total 61,000 today
"""

NOT_WHATSAPP = "This is just a plain text document with no chat format."


# ═══════════════════════════════════════════════════════════════════════════════
# Parser tests (no mocking needed — pure Python)
# ═══════════════════════════════════════════════════════════════════════════════

def test_parser_ios_format():
    from app.parser import parse_whatsapp_export, get_export_stats
    msgs = parse_whatsapp_export(SAMPLE_WA_IOS)
    assert len(msgs) == 5
    assert msgs[0].sender == "Ravi (Manager)"
    assert "52,000" in msgs[0].text
    assert msgs[0].date_str == "15/03/2024"

    stats = get_export_stats(msgs)
    assert stats["total_messages"] == 5
    assert "Ravi (Manager)" in stats["senders"]
    assert "Owner" in stats["senders"]


def test_parser_android_format():
    from app.parser import parse_whatsapp_export
    msgs = parse_whatsapp_export(SAMPLE_WA_ANDROID)
    assert len(msgs) == 4
    assert msgs[0].sender == "Ravi (Manager)"


def test_parser_multiline_message():
    wa_text = "[15/03/2024, 09:00:01] Ravi: First line\nSecond line\nThird line\n[15/03/2024, 09:01:00] Owner: Reply"
    from app.parser import parse_whatsapp_export
    msgs = parse_whatsapp_export(wa_text)
    assert len(msgs) == 2
    assert "Second line" in msgs[0].text
    assert "Third line" in msgs[0].text


def test_parser_media_omitted():
    wa_text = "[15/03/2024, 09:00:01] Ravi: <Media omitted>\n[15/03/2024, 09:01:00] Owner: Hello"
    from app.parser import parse_whatsapp_export
    msgs = parse_whatsapp_export(wa_text)
    # Media message is marked but not dropped — check it's flagged
    media_msgs = [m for m in msgs if m.is_media]
    assert len(media_msgs) == 1


def test_whatsapp_detection():
    from app.rag import _is_whatsapp_export
    assert _is_whatsapp_export(SAMPLE_WA_IOS) is True
    assert _is_whatsapp_export(SAMPLE_WA_ANDROID) is True
    assert _is_whatsapp_export(NOT_WHATSAPP) is False


# ═══════════════════════════════════════════════════════════════════════════════
# API tests (Pinecone + Groq mocked)
# ═══════════════════════════════════════════════════════════════════════════════

def _mock_ingest():
    return {"doc_id": "test-doc-id", "chunks_indexed": 3,
            "stats": {"total_messages": 5, "senders": {"Ravi": 3, "Owner": 2},
                      "date_range_start": "15 Mar 2024", "date_range_end": "16 Mar 2024"}}


def _mock_query():
    return {
        "answer": "The sales on 15 March were 52,000.",
        "sources": [{
            "sender": "Ravi (Manager)",
            "date_start": "2024-03-15",
            "date_end": "2024-03-15",
            "snippet": "15/03/2024 09:00 | Ravi (Manager): Good morning sir, yesterday's sales were 52,000",
        }],
    }


# ── Test: /session creates a session ID ──────────────────────────────────────
def test_create_session():
    res = client.post("/session")
    assert res.status_code == 200
    assert "session_id" in res.json()


# ── Test 1: /upload returns 200 + doc_id + session_id ───────────────────────
def test_upload_whatsapp_txt():
    with patch("app.main.rag.ingest_file", return_value=_mock_ingest()):
        res = client.post(
            "/upload",
            files={"file": ("chat.txt", io.BytesIO(SAMPLE_WA_IOS.encode()), "text/plain")},
        )
    assert res.status_code == 200
    body = res.json()
    assert "session_id" in body
    assert body["doc_id"] == "test-doc-id"
    assert body["chunks_indexed"] == 3
    assert "stats" in body


# ── Test 2: /upload rejects unsupported file type ───────────────────────────
def test_upload_rejects_docx():
    res = client.post(
        "/upload",
        files={"file": ("chat.docx", io.BytesIO(b"content"), "application/octet-stream")},
    )
    assert res.status_code == 400


# ── Test 3: /chat returns non-empty answer with sources ─────────────────────
def test_chat_returns_answer_with_sources():
    # First create a session with a doc
    with patch("app.main.rag.ingest_file", return_value=_mock_ingest()):
        up = client.post(
            "/upload",
            files={"file": ("chat.txt", io.BytesIO(SAMPLE_WA_IOS.encode()), "text/plain")},
        )
    sid = up.json()["session_id"]

    with patch("app.main.rag.query_rag", return_value=_mock_query()):
        res = client.post("/chat", json={"session_id": sid, "message": "What were the sales?"})

    assert res.status_code == 200
    body = res.json()
    assert len(body["answer"]) > 0
    assert isinstance(body["sources"], list)
    assert len(body["sources"]) > 0
    assert "sender" in body["sources"][0]
    assert "snippet" in body["sources"][0]


# ── Test 4: /chat rejects empty message ─────────────────────────────────────
def test_chat_rejects_empty_message():
    res = client.post("/chat", json={"session_id": "any", "message": "   "})
    assert res.status_code == 400


# ── Test 5: /chat/clear resets history ──────────────────────────────────────
def test_clear_chat_history():
    from app import session_store
    session = session_store.create_session()
    session.doc_id = "fake-doc"
    session_store.append_message(session, "user", "hello")
    session_store.append_message(session, "assistant", "hi")
    assert len(session.history) == 2

    res = client.post("/chat/clear", json={"session_id": session.session_id})
    assert res.status_code == 200
    assert len(session.history) == 0


# ── Test 6: /health returns ok ──────────────────────────────────────────────
def test_health():
    assert client.get("/health").json() == {"status": "ok"}


# ── Test 7: root serves HTML ─────────────────────────────────────────────────
def test_root_serves_html():
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert b"Volo AI" in res.content


# ── Test 8: /chat errors without uploaded doc ────────────────────────────────
def test_chat_without_upload_errors():
    res = client.post("/chat", json={"message": "What were the sales?"})
    # No session_id → creates new empty session → should 400
    assert res.status_code == 400
    assert "upload" in res.json()["detail"].lower()


# ── Test 9: messages_to_documents produces rich metadata ────────────────────
def test_documents_have_rich_metadata():
    from app.parser import parse_whatsapp_export, messages_to_documents
    msgs = parse_whatsapp_export(SAMPLE_WA_IOS)
    docs = messages_to_documents(msgs, "chat.txt", "doc-123")
    assert len(docs) > 0
    meta = docs[0].metadata
    assert "senders" in meta
    assert "date_start" in meta
    assert "source_filename" in meta
    assert meta["source_filename"] == "chat.txt"


# ── Test 10: /export requires uploaded doc ───────────────────────────────────
def test_export_without_upload_errors():
    res = client.post("/export", json={"message": "extract sales"})
    assert res.status_code == 400


# ── Test 11: /upload accepts a .zip containing a WhatsApp .txt ───────────────
def test_upload_whatsapp_zip():
    # Build an in-memory zip containing the sample WhatsApp export
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("WhatsApp Chat with Akhil.txt", SAMPLE_WA_IOS)
    zip_bytes = buf.getvalue()

    with patch("app.main.rag.ingest_file", return_value=_mock_ingest()):
        res = client.post(
            "/upload",
            files={"file": ("WhatsApp Chat with Akhil.zip", io.BytesIO(zip_bytes), "application/zip")},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["doc_id"] == "test-doc-id"
    assert body["chunks_indexed"] == 3


# ── Test 12: /upload rejects a .zip with no .txt inside ─────────────────────
def test_upload_zip_without_txt_errors():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("image.jpg", b"fake image data")
    zip_bytes = buf.getvalue()

    res = client.post(
        "/upload",
        files={"file": ("export.zip", io.BytesIO(zip_bytes), "application/zip")},
    )
    assert res.status_code == 422
    assert ".txt" in res.json()["detail"]


# ════════════════════════════════════════════════════════════════════════════
# Memory-optimisation tests (TDD — written before implementation)
# ════════════════════════════════════════════════════════════════════════════

# ── Session TTL ──────────────────────────────────────────────────────────────
def test_session_is_not_expired_initially():
    from app import session_store
    session = session_store.create_session()
    assert session_store.is_expired(session) is False


def test_session_is_expired_after_ttl():
    from app import session_store
    from datetime import datetime, timedelta, timezone
    session = session_store.create_session()
    session.last_active = datetime.now(timezone.utc) - timedelta(hours=session_store.SESSION_TTL_HOURS + 1)
    assert session_store.is_expired(session) is True


def test_touch_resets_expiry():
    from app import session_store
    from datetime import datetime, timedelta, timezone
    session = session_store.create_session()
    session.last_active = datetime.now(timezone.utc) - timedelta(hours=session_store.SESSION_TTL_HOURS + 1)
    assert session_store.is_expired(session) is True
    session_store.touch(session)
    assert session_store.is_expired(session) is False


def test_cleanup_expired_removes_session():
    from app import session_store
    from datetime import datetime, timedelta, timezone
    session = session_store.create_session()
    sid = session.session_id
    session.last_active = datetime.now(timezone.utc) - timedelta(hours=session_store.SESSION_TTL_HOURS + 1)
    removed = session_store.cleanup_expired()
    assert removed >= 1
    assert session_store.get_session(sid) is None


def test_cleanup_calls_on_expire_callback():
    from app import session_store
    from datetime import datetime, timedelta, timezone
    session = session_store.create_session()
    session.last_active = datetime.now(timezone.utc) - timedelta(hours=session_store.SESSION_TTL_HOURS + 1)
    fired = []
    session_store.cleanup_expired(on_expire=lambda s: fired.append(s.session_id))
    assert session.session_id in fired


def test_cleanup_keeps_active_sessions():
    from app import session_store
    session = session_store.create_session()
    sid = session.session_id
    session_store.cleanup_expired()
    assert session_store.get_session(sid) is not None


# ── History trim ─────────────────────────────────────────────────────────────
def test_history_trims_at_max_plus_two():
    from app import session_store
    session = session_store.create_session()
    for i in range(session_store.MAX_HISTORY + 3):
        session_store.append_message(session, "user", f"msg {i}")
    assert len(session.history) == session_store.MAX_HISTORY


def test_history_no_trim_below_limit():
    from app import session_store
    session = session_store.create_session()
    for i in range(session_store.MAX_HISTORY):
        session_store.append_message(session, "user", f"msg {i}")
    assert len(session.history) == session_store.MAX_HISTORY


# ── File size cap ────────────────────────────────────────────────────────────
def test_upload_rejects_oversized_file():
    import os
    max_mb = int(os.getenv("MAX_UPLOAD_MB", "5"))
    big = b"x" * (max_mb * 1024 * 1024 + 1)
    res = client.post(
        "/upload",
        files={"file": ("big.txt", io.BytesIO(big), "text/plain")},
    )
    assert res.status_code == 400
    assert "too large" in res.json()["detail"].lower()


# ── Groq client singleton ─────────────────────────────────────────────────────
def test_groq_client_is_singleton():
    import app.rag as rag_module
    with patch("app.rag.Groq") as mock_groq:
        mock_groq.return_value = MagicMock()
        old = rag_module._groq_client
        rag_module._groq_client = None
        try:
            c1 = rag_module.get_groq_client()
            c2 = rag_module.get_groq_client()
            assert c1 is c2
            assert mock_groq.call_count == 1
        finally:
            rag_module._groq_client = old


# ── Vectorstore cache ─────────────────────────────────────────────────────────
def test_vectorstore_cache_returns_same_instance():
    import app.rag as rag_module
    ns = "test-ns-cache-singleton"
    with patch("app.rag.PineconeVectorStore") as mock_vs:
        mock_vs.return_value = MagicMock()
        rag_module.evict_vectorstore(ns)
        vs1 = rag_module.get_vectorstore(ns)
        vs2 = rag_module.get_vectorstore(ns)
        assert vs1 is vs2
        assert mock_vs.call_count == 1
        rag_module.evict_vectorstore(ns)


def test_vectorstore_eviction_forces_new_instance():
    import app.rag as rag_module
    ns = "test-ns-evict-new"
    with patch("app.rag.PineconeVectorStore") as mock_vs:
        mock_vs.side_effect = [MagicMock(), MagicMock()]
        rag_module.evict_vectorstore(ns)
        vs1 = rag_module.get_vectorstore(ns)
        rag_module.evict_vectorstore(ns)
        vs2 = rag_module.get_vectorstore(ns)
        assert vs1 is not vs2
        assert mock_vs.call_count == 2
        rag_module.evict_vectorstore(ns)
