"""FastAPI app: ingest documents, then chat with RAG over them."""
import io

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader

import rag
import llm

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
    if name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = raw.decode("utf-8", errors="ignore")
    n = rag.add_document(text, name)
    if n == 0:
        raise HTTPException(400, "Could not extract any text from the file.")
    return {"ingested_chunks": n, "source": name, **rag.stats()}


@app.post("/chat")
def chat(body: ChatRequest):
    if not body.message.strip():
        raise HTTPException(400, "Empty message.")
    chunks = rag.retrieve(body.message)
    if not chunks:
        return {
            "answer": "No documents have been ingested yet. Add some text or a file first.",
            "sources": [],
        }
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
