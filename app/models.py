from typing import Optional
from pydantic import BaseModel


class UploadResponse(BaseModel):
    session_id: str
    doc_id: str
    chunks_indexed: int
    message: str
    stats: dict  # {total_messages, senders, date_range_start, date_range_end}


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class Source(BaseModel):
    sender: str
    date_start: str
    date_end: str
    snippet: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[Source]


class ClearRequest(BaseModel):
    session_id: str


# Legacy — kept so /query still works
class QueryRequest(BaseModel):
    question: str
    namespace: str = "uploaded-docs"


class QueryResponse(BaseModel):
    answer: str
    sources: list
