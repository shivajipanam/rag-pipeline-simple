"""
In-memory session store.
Each session tracks:
  - chat history (list of {role, content} dicts for Groq)
  - the Pinecone namespace where that session's documents live
  - upload metadata (filename, stats)

For MVP this lives in process memory. For production, swap with Redis.
"""
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Session:
    session_id: str
    namespace: str                      # Pinecone namespace = session_id
    history: list[dict] = field(default_factory=list)   # [{role, content}, ...]
    upload_filename: Optional[str] = None
    upload_stats: Optional[dict] = None  # from parser.get_export_stats()
    doc_id: Optional[str] = None


# Global store — keyed by session_id
_store: dict[str, Session] = {}

MAX_HISTORY = 20  # keep last N turns to avoid Groq token limits


def create_session() -> Session:
    sid = str(uuid.uuid4())
    session = Session(session_id=sid, namespace=sid)
    _store[sid] = session
    return session


def get_session(session_id: str) -> Optional[Session]:
    return _store.get(session_id)


def get_or_create(session_id: Optional[str]) -> Session:
    if session_id and session_id in _store:
        return _store[session_id]
    return create_session()


def append_message(session: Session, role: str, content: str) -> None:
    session.history.append({"role": role, "content": content})
    # Trim to keep last MAX_HISTORY messages (preserve system message if present)
    if len(session.history) > MAX_HISTORY * 2:
        session.history = session.history[-(MAX_HISTORY * 2):]


def clear_history(session: Session) -> None:
    session.history.clear()
