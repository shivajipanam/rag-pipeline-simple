# LinkedIn Post — ChatSearch (RAG Pipeline)

---

🚀 Built a RAG app that lets you chat with your WhatsApp history.

Upload your WhatsApp chat export and ask questions like:

→ "When did we last talk about the Paris trip?"
→ "What did Ravi say about the delivery issue?"
→ "Summarise all conversations from March"

Every answer shows the exact source — sender name, date, and the message snippet used to generate the response. You can also export structured data to Excel: ask "extract all daily sales figures" and download a .xlsx built directly from your chat.

The original use case I built this for: a restaurant manager who tracks daily sales, supplier updates, and team notes over WhatsApp. Instead of scrolling through months of messages, they can now just ask.

---

**The technical side — this is a full RAG pipeline:**

📄 **Custom parser for WhatsApp exports** — WhatsApp exports text in two formats depending on your phone OS (iOS vs Android), with different timestamp patterns, multiline messages, and invisible Unicode characters that break standard regex. I wrote a parser that handles all of it and extracts structured message objects with sender, timestamp, and content.

🧩 **Message-window chunking** — Instead of splitting by character count (standard RAG), I group 15 consecutive messages into one document. Each chunk carries metadata: who sent the messages and the exact date range. This preserves conversational context and makes source attribution accurate.

🔍 **Pinecone vector store with per-session namespacing** — Each user session gets a UUID that serves as their Pinecone namespace. This isolates every user's data without needing authentication. Upload → embed with `all-MiniLM-L6-v2` → store. Query → embed question → retrieve top 6 chunks → answer.

🤖 **Multi-turn conversation with Groq (Llama 3.1 8B)** — The LLM receives the retrieved chunks as context plus the last 10 turns of conversation history. It's instructed to answer only from the provided evidence and cite senders and dates. Temperature set to 0.2 for factual, grounded responses.

📊 **Excel extraction** — A second Groq call with a structured extraction prompt tells the LLM to output a JSON array of rows. That gets parsed and written to `.xlsx` with pandas. Useful for pulling out tabular data buried in chat — sales figures, order lists, schedules.

🐳 **Deployed on Railway with CPU-only PyTorch** — The sentence-transformers model needs PyTorch. The default CUDA build is 3.5GB. I use the CPU-only build (800MB), keeping the total Docker image under Railway's 4GB free tier limit. A `Dockerfile.gpu` is ready for when I upgrade.

---

**Stack:**
- Backend: Python + FastAPI
- Embeddings: sentence-transformers `all-MiniLM-L6-v2` (HuggingFace, free, runs locally)
- Vector DB: Pinecone (free tier, namespace-isolated per session)
- LLM: Groq — Llama 3.1 8B Instant (free tier)
- Frontend: Vanilla HTML/CSS/JS
- Deploy: Docker on Railway

**16 automated tests** — all external calls (Pinecone, Groq) mocked. Parser tested against both WhatsApp formats, multiline messages, media placeholders, and edge cases.

---

The most interesting design decision: using Pinecone namespaces as a session isolation mechanism. No user accounts, no auth middleware — just UUIDs. Each upload lives in its own namespace and is completely invisible to other sessions. Simple, elegant, production-safe for a demo scale app.

Next up: async ingestion for large files and Redis-backed sessions so they survive container restarts.

Repo: https://github.com/shivajipanam/rag-pipeline-simple

#Python #RAG #LLM #FastAPI #NLP #AI #MachineLearning #BuildInPublic #LangChain #Pinecone

---

*Copy the text above the `---` line for LinkedIn. Adjust tone as needed.*
