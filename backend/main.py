"""FastAPI app: ingest documents, then chat with RAG over them."""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import rag
import llm
import extract

app = FastAPI(title="Kaya RAG API")

# Allow the Vite dev server to call us during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- request/response models ----------
class IngestText(BaseModel):
    text: str
    source: str = "pasted-text"


class ChatRequest(BaseModel):
    message: str


# ---------- endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", **rag.stats()}


@app.post("/ingest")
def ingest_text(body: IngestText):
    n = rag.add_document(body.text, body.source)
    if n == 0:
        raise HTTPException(400, "No text to ingest.")
    return {"ingested_chunks": n, "source": body.source, **rag.stats()}


@app.post("/ingest-file")
async def ingest_file(file: UploadFile = File(...)):
    raw = await file.read()
    name = file.filename or "upload"
    try:
        text = extract.extract_text(raw, name)
    except Exception as e:  # extraction (e.g. vision) can fail on odd files
        raise HTTPException(400, f"Could not read the file: {e}")
    n = rag.add_document(text, name)
    if n == 0:
        raise HTTPException(400, "Could not extract any text from the file.")
    return {"ingested_chunks": n, "source": name, **rag.stats()}


@app.post("/chat")
def chat(body: ChatRequest):
    if not body.message.strip():
        raise HTTPException(400, "Empty message.")

    # 1. Hybrid retrieval (dense + BM25 + RRF) -> candidate pool.
    candidates = rag.retrieve(body.message)
    if not candidates:
        return {
            "answer": "No documents have been ingested yet. Add some text or a file first.",
            "sources": [],
        }

    # 2. Anti-hallucination gate: if nothing is semantically close, don't answer.
    if max(c["score"] for c in candidates) < config.RELEVANCE_THRESHOLD:
        return {
            "answer": "I don't have enough information in the knowledge base to answer that.",
            "sources": [],
        }

    # 3. LLM rerank the pool down to the best TOP_K, then answer over those.
    chunks = llm.rerank(body.message, candidates, config.TOP_K)
    answer = llm.generate_answer(body.message, chunks)
    sources = [
        {"n": i + 1, "source": c["source"], "score": c["score"], "preview": c["text"][:160]}
        for i, c in enumerate(chunks)
    ]
    return {"answer": answer, "sources": sources}


@app.post("/reset")
def reset():
    rag.reset()
    return {"status": "cleared", **rag.stats()}
