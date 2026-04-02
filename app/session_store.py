"""
In-memory session store with TTL-based expiry.

Each session tracks:
  - chat history (list of {role, content} dicts for Groq)
  - the Pinecone namespace where that session's documents live
  - upload metadata (filename, stats)
  - last_active timestamp for TTL-based eviction

For MVP this lives in process memory. For production, swap with Redis.

Memory optimisations applied:
  - SESSION_TTL_HOURS: sessions inactive longer than this are evicted by the
    background cleanup task in main.py.
  - History trim threshold changed from MAX_HISTORY * 2 (40 messages) to
    MAX_HISTORY + 2 (22 messages), keeping the working set tight at all times.
  - touch() is called on every interaction so the TTL resets on active sessions.
  - cleanup_expired() accepts an on_expire callback so callers (main.py) can
    trigger Pinecone namespace deletion and vectorstore cache eviction.
"""
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Callable


SESSION_TTL_HOURS: int = int(os.getenv("SESSION_TTL_HOURS", "2"))
MAX_HISTORY: int = 20          # maximum messages kept per session

_store: dict[str, "Session"] = {}


@dataclass
class Session:
    session_id: str
    namespace: str                        # Pinecone namespace = session_id
    history: list[dict] = field(default_factory=list)
    upload_filename: Optional[str] = None
    upload_stats: Optional[dict] = None   # from parser.get_export_stats()
    doc_id: Optional[str] = None
    last_active: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Public API ────────────────────────────────────────────────────────────────

def create_session() -> Session:
    sid = str(uuid.uuid4())
    session = Session(session_id=sid, namespace=sid)
    _store[sid] = session
    return session


def get_session(session_id: str) -> Optional[Session]:
    return _store.get(session_id)


def get_or_create(session_id: Optional[str]) -> Session:
    if session_id and session_id in _store:
        session = _store[session_id]
        touch(session)
        return session
    return create_session()


def touch(session: Session) -> None:
    """Reset the TTL clock — call on every user interaction."""
    session.last_active = _now()


def is_expired(session: Session) -> bool:
    """Return True if the session has been idle longer than SESSION_TTL_HOURS."""
    cutoff = _now() - timedelta(hours=SESSION_TTL_HOURS)
    return session.last_active < cutoff


def append_message(session: Session, role: str, content: str) -> None:
    session.history.append({"role": role, "content": content})
    touch(session)
    # Trim threshold: MAX_HISTORY + 2 (was MAX_HISTORY * 2 = 40).
    # Keeps history within MAX_HISTORY entries at all times rather than
    # letting it balloon to 40 before the first trim fires.
    if len(session.history) > MAX_HISTORY + 2:
        session.history = session.history[-MAX_HISTORY:]


def clear_history(session: Session) -> None:
    session.history.clear()


def cleanup_expired(
    on_expire: Optional[Callable[["Session"], None]] = None,
) -> int:
    """
    Scan the store, remove all expired sessions, and return the count removed.

    ``on_expire(session)`` is called for each expired session before it is
    deleted from the store.  Use this hook to:
      - delete the Pinecone namespace  (rag.delete_pinecone_namespace)
      - evict the vectorstore cache    (rag.evict_vectorstore)

    Errors inside on_expire are silently swallowed so one bad session cannot
    block cleanup of the rest.
    """
    expired_ids = [sid for sid, s in list(_store.items()) if is_expired(s)]
    for sid in expired_ids:
        session = _store.pop(sid)
        if on_expire:
            try:
                on_expire(session)
            except Exception:
                pass
    return len(expired_ids)
