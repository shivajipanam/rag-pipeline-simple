"""
WhatsApp chat export parser.

Supports both export formats WhatsApp produces:
  Format A (iOS):  [DD/MM/YYYY, HH:MM:SS] Sender Name: message
  Format B (Android): DD/MM/YYYY, HH:MM - Sender Name: message

Also handles:
  - Multiline messages (continuation lines without a timestamp prefix)
  - System messages (no sender colon) → skipped
  - Media placeholders: "<Media omitted>", "image omitted", "video omitted", etc.
  - Zero-width / special Unicode characters WhatsApp inserts
"""
import io
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from langchain_core.documents import Document

# ── Regex patterns ─────────────────────────────────────────────────────────────

# Format A: [15/03/2024, 09:45:32] John: hello
_PATTERN_A = re.compile(
    r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?)\]\s*([^:]+):\s*(.*)",
    re.UNICODE,
)

# Format B: 15/03/2024, 09:45 - John: hello
_PATTERN_B = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*([^:]+):\s*(.*)",
    re.UNICODE,
)

# Detect any line that starts with a date (used to split continuation lines)
_DATE_PREFIX_A = re.compile(r"^\[(\d{1,2}/\d{1,2}/\d{2,4})")
_DATE_PREFIX_B = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4})")

# Media / system noise to normalise
_MEDIA_RE = re.compile(
    r"<[Mm]edia omitted>|‎?image omitted|‎?video omitted|‎?audio omitted|‎?sticker omitted|‎?document omitted",
    re.IGNORECASE,
)

# Zero-width and direction-control chars WhatsApp inserts
_ZWC_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\ufeff]")


@dataclass
class ChatMessage:
    date_str: str          # raw date string from export e.g. "15/03/2024"
    time_str: str          # raw time string e.g. "09:45:32"
    sender: str
    text: str
    is_media: bool = False
    parsed_dt: Optional[datetime] = None


def preprocess_zip(zip_bytes: bytes) -> tuple[bytes, str]:
    """
    Extract a WhatsApp chat .txt file from a zip archive.

    WhatsApp exports are zip files containing a single .txt chat file
    (and optionally media files which are ignored here).

    Returns:
        (txt_bytes, filename) — the raw bytes of the .txt file and its name.

    Raises:
        ValueError if the zip contains no .txt file.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        txt_names = [name for name in zf.namelist() if name.lower().endswith(".txt")]
        if not txt_names:
            raise ValueError("No .txt file found inside the ZIP. Please export the WhatsApp chat as a .txt file.")
        # Pick the first .txt (WhatsApp exports contain exactly one)
        txt_name = txt_names[0]
        txt_bytes = zf.read(txt_name)
    return txt_bytes, txt_name


def _clean(text: str) -> str:
    text = _ZWC_RE.sub("", text)
    return text.strip()


def _parse_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    """Try several date formats WhatsApp uses across locales."""
    for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%y %H:%M:%S",
                "%d/%m/%Y %H:%M",    "%d/%m/%y %H:%M",
                "%m/%d/%Y %H:%M:%S", "%m/%d/%y %H:%M:%S",
                "%m/%d/%Y %H:%M",    "%m/%d/%y %H:%M"):
        try:
            return datetime.strptime(f"{date_str} {time_str}", fmt)
        except ValueError:
            continue
    return None


def parse_whatsapp_export(raw_text: str) -> list[ChatMessage]:
    """
    Parse a WhatsApp .txt export into a list of ChatMessage objects.
    Multiline messages are merged into a single entry.
    System messages (no sender) are discarded.
    """
    raw_text = _clean(raw_text)
    lines = raw_text.splitlines()

    messages: list[ChatMessage] = []
    current: Optional[ChatMessage] = None

    for line in lines:
        line = _clean(line)
        if not line:
            continue

        m = _PATTERN_A.match(line) or _PATTERN_B.match(line)
        if m:
            # Save previous message
            if current is not None:
                messages.append(current)

            date_str, time_str, sender, text = m.group(1), m.group(2), m.group(3).strip(), m.group(4)
            text = _clean(text)
            is_media = bool(_MEDIA_RE.search(text))
            if is_media:
                text = "[media]"

            current = ChatMessage(
                date_str=date_str,
                time_str=time_str,
                sender=sender,
                text=text,
                is_media=is_media,
                parsed_dt=_parse_datetime(date_str, time_str),
            )
        else:
            # Continuation line — append to current message
            if current is not None:
                continuation = _clean(line)
                if continuation:
                    current.text = current.text + "\n" + continuation

    if current is not None:
        messages.append(current)

    # Drop empty / pure-system messages (keep media messages marked with is_media=True)
    messages = [m for m in messages if m.text]
    return messages


def messages_to_documents(
    messages: list[ChatMessage],
    source_filename: str,
    doc_id: str,
    chunk_size: int = 20,
) -> list[Document]:
    """
    Group consecutive messages into LangChain Documents for indexing.
    Each document is a window of `chunk_size` messages with rich metadata.
    Preserves sender, date, and message text so the LLM can cite sources.
    """
    docs: list[Document] = []

    for i in range(0, len(messages), chunk_size):
        window = messages[i: i + chunk_size]
        if not window:
            continue

        # Build readable block: "15/03/2024 09:45 | John: hello there"
        lines = []
        for msg in window:
            if msg.is_media:
                continue
            ts = f"{msg.date_str} {msg.time_str}"
            lines.append(f"{ts} | {msg.sender}: {msg.text}")

        if not lines:
            continue

        page_content = "\n".join(lines)

        # Date range for this chunk
        dates = [m.parsed_dt for m in window if m.parsed_dt]
        senders = list(dict.fromkeys(m.sender for m in window))  # ordered unique

        docs.append(Document(
            page_content=page_content,
            metadata={
                "doc_id": doc_id,
                "source_filename": source_filename,
                "chunk_index": i // chunk_size,
                "date_start": dates[0].isoformat() if dates else "",
                "date_end": dates[-1].isoformat() if dates else "",
                "senders": ", ".join(senders),
                "message_count": len([m for m in window if not m.is_media]),
            },
        ))

    return docs


def get_export_stats(messages: list[ChatMessage]) -> dict:
    """Return summary stats about the parsed chat — shown to user after upload."""
    non_media = [m for m in messages if not m.is_media]
    senders = {}
    for m in non_media:
        senders[m.sender] = senders.get(m.sender, 0) + 1

    dates = sorted(
        m.parsed_dt for m in non_media if m.parsed_dt
    )

    return {
        "total_messages": len(non_media),
        "senders": senders,
        "date_range_start": dates[0].strftime("%d %b %Y") if dates else "unknown",
        "date_range_end": dates[-1].strftime("%d %b %Y") if dates else "unknown",
    }
