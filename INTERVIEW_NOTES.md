# Interview cheat-sheet (1 hour)

## 60-second pitch of the architecture
"It's a RAG app. The **frontend** is React. The **backend** is FastAPI with two
responsibilities split into two files: `rag.py` owns the vector pipeline (chunk → embed →
store → retrieve) using ChromaDB, and `llm.py` owns generation using Qwen through NVIDIA's
OpenAI-compatible API. On a question we embed it, pull the top-k similar chunks from Chroma,
stuff them into the prompt as numbered context, and ask the model to answer **only** from
that context with citations. Embeddings run locally (MiniLM) so retrieval needs no API key."

## What is RAG, in one line
Retrieve relevant text from your own data, then **augment** the LLM prompt with it so the
model answers from facts instead of hallucinating.

## The 3 stages (point at the code)
1. **Ingestion** — `rag.add_document`: split text into overlapping chunks, embed each,
   store vectors + metadata in Chroma.
2. **Retrieval** — `rag.retrieve`: embed the question, cosine-search Chroma for top-k.
3. **Generation** — `llm.generate_answer`: build a prompt with the chunks, call the LLM.

## Likely questions + crisp answers
- **Why chunk?** Embeddings capture a limited span of meaning and the LLM has a context
  limit. Smaller chunks = more precise retrieval. **Overlap** avoids cutting a fact in half.
- **Why overlap (120 chars)?** So a sentence split across a boundary still appears whole in
  at least one chunk.
- **Why a vector DB vs keyword search?** Semantic match — "cost" finds "pricing" even with
  no shared words. Keyword (BM25) is exact-term. Best systems do **hybrid** + a reranker.
- **What's an embedding?** A vector where semantically similar text is geometrically close.
  We compare with **cosine similarity**.
- **How do you stop hallucination?** System prompt says answer only from context + "say you
  don't know"; we also return the **sources** so answers are auditable.
- **How would you scale this?** Swap Chroma for a hosted vector DB (Pinecone/pgvector/
  Weaviate), batch embeddings, add caching, add a reranker, stream tokens to the UI.
- **Chunk size trade-off?** Too small → context fragments, lost meaning. Too big → noisy
  retrieval + wasted tokens. 500–1000 chars is a sane default; tune per data.
- **How is the LLM integrated?** Standard OpenAI client pointed at NVIDIA's `base_url`.
  Provider-agnostic — swapping to OpenAI/Anthropic is a config change.

## Fast live extensions if they ask for "one more feature"
- **Streaming**: `client.chat.completions.create(..., stream=True)` + SSE / chunked response.
- **Top-k slider**: pass `top_k` from the UI into `/chat` → `rag.retrieve`.
- **Conversation memory**: keep prior turns, prepend to the messages list.
- **Reranking**: after retrieve, sort chunks with a cross-encoder before prompting.
- **Citations as links**: you already return `sources`; render them as footnotes (UI shows
  them in a collapsible "sources" panel already).
- **Metadata filter**: `collection.query(where={"source": "..."})` to scope a search.

## Gotchas you already solved (mention if relevant)
- Use **Python 3.12**, not 3.13/3.14 — ChromaDB deps lack wheels on the newest Python.
- File uploads need **`python-multipart`**.
- NVIDIA model IDs matter — `qwen/qwen2.5-7b-instruct` 404s; valid ones are listed by
  `list_models.py` (e.g. `qwen/qwen3-next-80b-a3b-instruct`).
- `TOKENIZERS_PARALLELISM=false` silences a harmless fork warning.

## If you must rebuild from scratch under time pressure (order of operations)
1. `python3.12 -m venv .venv && pip install fastapi uvicorn openai chromadb python-dotenv python-multipart`
2. `rag.py`: Chroma client + `chunk_text` + `add_document` + `retrieve`.
3. `llm.py`: OpenAI client → NVIDIA base_url + `generate_answer`.
4. `main.py`: `/ingest` and `/chat`. Test with `curl` before touching the UI.
5. `npm create vite@latest frontend -- --template react`, then a chat box + fetch.
6. Add Vite proxy `/api → :8000` to dodge CORS, or set CORS middleware (this repo does both).
