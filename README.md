# ChatSearch — Ask Your WhatsApp Chats

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg) ![Tests](https://img.shields.io/badge/Tests-16%2F16%20passing-brightgreen.svg)

Upload a WhatsApp chat export and ask questions about it in plain English.

> "When did we last talk about the trip to Paris?"
> "What did John say about the project deadline?"
> "Summarise our conversations from March"

Responses include **source context** — the sender name, date range, and the exact message snippet that was used to answer.

## Features

- Upload WhatsApp `.txt` exports (iOS and Android formats supported)
- Persistent chat session with conversation history
- Collapsible source cards under each AI answer
- Suggested prompts after upload
- Excel export — ask a question like "extract all daily sales figures" and download a `.xlsx`
- Works with any WhatsApp chat: personal, group, business

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI + uvicorn |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (sentence-transformers, CPU) |
| Vector store | Pinecone (free tier) |
| LLM | Groq — Llama 3.1 8B Instant (free tier) |
| Frontend | Vanilla HTML/CSS/JS (served by FastAPI) |
| Deploy | Railway (Docker, CPU-only PyTorch for small image) |

## Installation

```bash
git clone https://github.com/shivajipanam/rag-pipeline-simple.git
cd rag-pipeline-simple

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cp .env.example .env
# Fill in your API keys in .env
```

## Environment Variables

| Variable | Required | Where to get |
|----------|----------|--------------|
| `PINECONE_API_KEY` | Yes | [app.pinecone.io](https://app.pinecone.io) — free tier |
| `GROQ_API_KEY` | Yes | [console.groq.com](https://console.groq.com) — free tier |
| `PINECONE_INDEX_NAME` | No | Defaults to `rag-workshop` |

**Pinecone index setup** (one-time):
- Name: `rag-workshop`
- Dimensions: `384`
- Metric: `cosine`

## Running Locally

```bash
python run.py
# Open http://localhost:8000
```

## Running Tests

```bash
pytest tests/ -v
# 16/16 tests pass — all external calls mocked, no API keys needed
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/session` | Create a new chat session |
| POST | `/upload` | Upload a WhatsApp `.txt` or `.pdf` file |
| POST | `/chat` | Send a message, get an answer + sources |
| POST | `/chat/clear` | Clear chat history for a session |
| POST | `/export` | Extract structured data → download `.xlsx` |
| GET | `/health` | Health check |

## Deploy to Railway

1. Create a Pinecone index (see above)
2. Push to GitHub
3. Railway → New Project → Deploy from GitHub
4. Add environment variables: `PINECONE_API_KEY`, `GROQ_API_KEY`, `PINECONE_INDEX_NAME`
5. Deploy — Railway uses the Dockerfile automatically

## WhatsApp Export Format

The parser supports both export formats:
- **iOS**: `[DD/MM/YYYY, HH:MM:SS] Sender: message`
- **Android**: `DD/MM/YYYY, HH:MM - Sender: message`

Multiline messages, media placeholders, and system messages are handled automatically.

## License

MIT
