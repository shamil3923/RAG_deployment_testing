# Kaya RAG — React + FastAPI + Qwen (NVIDIA) + Chroma

A minimal but complete **Retrieval-Augmented Generation** web app, built as interview prep.
Ingest documents → they get chunked, embedded, and stored in a vector DB → ask questions →
the backend retrieves the most relevant chunks and feeds them to an LLM, which answers
**with inline citations**.

```
┌──────────────┐   /api/*    ┌─────────────────────────────┐
│ React (Vite) │ ──────────► │ FastAPI                     │
│  chat UI     │             │  ├─ rag.py  → Chroma (vectors)
│  ingest panel│ ◄────────── │  └─ llm.py  → Qwen via NVIDIA│
└──────────────┘   answer +  └─────────────────────────────┘
                   sources
```

## Stack
| Layer        | Choice                                   | Why |
|--------------|------------------------------------------|-----|
| Frontend     | React + Vite                             | Fast, simple SPA |
| Backend      | FastAPI                                  | Async, typed, auto docs at `/docs` |
| Vector store | ChromaDB (persistent, cosine)            | Real vector DB, zero-config |
| Embeddings   | `all-MiniLM-L6-v2` (Chroma built-in ONNX)| Local, no API key needed |
| LLM          | Qwen 3 via NVIDIA's OpenAI-compatible API| Swappable in one line |

## Prerequisites
- Python 3.12 (3.13/3.14 may lack ChromaDB wheels)
- Node 18+
- An NVIDIA API key in `backend/.env` (already filled in for the demo)

## Run it (two terminals)
```bash
# terminal 1 — backend  (http://localhost:8000, docs at /docs)
./run-backend.sh

# terminal 2 — frontend (http://localhost:5173)
./run-frontend.sh
```
Open http://localhost:5173, paste some text (or upload a `.txt`/`.pdf`) on the left,
then ask a question. Try the included sample: paste `backend/data/sample.txt`, then ask
*"What are Kaya's pricing tiers?"*

> First backend start downloads the ~80 MB embedding model once (cached afterward).

## Manual setup (if a script fails)
```bash
# backend
cd backend
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m uvicorn main:app --reload --port 8000

# frontend
cd frontend
npm install
npm run dev
```

## API
| Method | Path           | Body                          | Purpose |
|--------|----------------|-------------------------------|---------|
| GET    | `/health`      | —                             | status + chunk count |
| POST   | `/ingest`      | `{text, source?}`             | add pasted text |
| POST   | `/ingest-file` | multipart `file` (.txt/.pdf)  | add a file |
| POST   | `/chat`        | `{message}`                   | RAG answer + sources |
| POST   | `/reset`       | —                             | wipe the vector store |

## Where each concept lives (for explaining it live)
- **Chunking** → `backend/rag.py` `chunk_text()` (sliding window, char-based + overlap)
- **Embedding + storage** → `backend/rag.py` `add_document()` (Chroma `collection.add`)
- **Retrieval** → `backend/rag.py` `retrieve()` (`collection.query`, cosine similarity)
- **Prompt assembly** → `backend/llm.py` `build_user_prompt()` (numbered context passages)
- **Generation** → `backend/llm.py` `generate_answer()` (OpenAI client → NVIDIA base URL)
- **API wiring** → `backend/main.py`
- **UI** → `frontend/src/App.jsx`, API calls in `frontend/src/api.js`

## Swapping the model / provider
Everything is in `backend/.env`:
```
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_API_KEY=...
LLM_MODEL=qwen/qwen3-next-80b-a3b-instruct
```
Run `./.venv/bin/python backend/list_models.py` to list available Qwen models on your key.
To use OpenAI/Anthropic instead, change `NVIDIA_BASE_URL`, the key, and `LLM_MODEL` — the
code is unchanged because it uses the OpenAI-compatible client.

## ⚠️ Security note
The demo API key is committed in `backend/.env` (gitignored) for convenience.
**Rotate it after the interview** — it was shared in plain text.
