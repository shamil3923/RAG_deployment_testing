"""RAG core: chunk -> embed (NVIDIA nv-embedqa) -> store in Chroma -> hybrid retrieve.

Chunking is sentence/paragraph aware so passages stay semantically whole.
Ingestion is idempotent: chunk IDs are derived from a hash of (source + text),
so re-uploading the same document upserts instead of creating duplicates.
Retrieval is hybrid: dense (vector) + sparse (BM25), fused with Reciprocal Rank Fusion.
"""
import hashlib
import re

import chromadb
from rank_bm25 import BM25Okapi

import config
from embeddings import PassageEmbeddingFunction, embed_query

_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
_embed_fn = PassageEmbeddingFunction()
_collection = _client.get_or_create_collection(
    name=config.COLLECTION_NAME,
    embedding_function=_embed_fn,
    metadata={"hnsw:space": "cosine"},
)

# In-memory BM25 index + a corpus lookup, rebuilt when the chunk count changes.
_bm25: BM25Okapi | None = None
_bm25_ids: list[str] = []
_corpus: dict[str, dict] = {}
_indexed_count = -1


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _ensure_index() -> None:
    """(Re)build the BM25 index and corpus lookup if the collection changed."""
    global _bm25, _bm25_ids, _corpus, _indexed_count
    count = _collection.count()
    if _bm25 is not None and count == _indexed_count:
        return
    data = _collection.get(include=["documents", "metadatas"])
    _bm25_ids = data["ids"]
    docs = data["documents"]
    metas = data["metadatas"]
    _corpus = {
        cid: {"text": d, "source": (m or {}).get("source")}
        for cid, d, m in zip(_bm25_ids, docs, metas)
    }
    _bm25 = BM25Okapi([_tokenize(d) for d in docs]) if docs else None
    _indexed_count = count


def _rrf(rankings: list[list[str]]) -> list[str]:
    """Reciprocal Rank Fusion: combine several ranked id-lists into one order."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (config.RRF_K + rank + 1)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def chunk_text(text: str, size: int = None, overlap: int = None) -> list[str]:
    """Sentence/paragraph-aware chunker that packs units up to ~size chars."""
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []

    # Split on sentence ends and blank lines, keeping non-empty units.
    units = [u.strip() for u in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if u and u.strip()]

    chunks: list[str] = []
    cur = ""
    for u in units:
        if len(u) > size:
            # A single oversized unit (e.g. a giant line): hard-split it.
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(u), size - overlap):
                chunks.append(u[i:i + size])
            continue
        if len(cur) + len(u) + 1 <= size:
            cur = f"{cur} {u}".strip()
        else:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""  # carry overlap into next chunk
            cur = f"{tail} {u}".strip() if tail else u
    if cur:
        chunks.append(cur)
    return chunks


def _chunk_id(source: str, text: str) -> str:
    return hashlib.sha1(f"{source}::{text}".encode("utf-8")).hexdigest()


def add_document(text: str, source: str) -> int:
    """Chunk a document and upsert it into the vector store. Returns chunk count."""
    chunks = chunk_text(text)
    if not chunks:
        return 0
    ids = [_chunk_id(source, c) for c in chunks]
    metadatas = [{"source": source, "chunk": i} for i in range(len(chunks))]
    # upsert (not add) so identical chunks don't pile up on re-upload.
    _collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


def retrieve(question: str, candidates: int = None) -> list[dict]:
    """Hybrid retrieval: dense + BM25 fused with RRF. Returns a candidate pool.

    Each item carries `score` = dense cosine similarity (0.0 if it was only a
    sparse/keyword hit), which the caller uses for the relevance gate.
    """
    n = _collection.count()
    if n == 0:
        return []
    candidates = min(candidates or config.RETRIEVE_CANDIDATES, n)
    _ensure_index()

    # Dense: encode the question with input_type="query" (asymmetric retrieval).
    qvec = embed_query(question)
    dres = _collection.query(query_embeddings=[qvec], n_results=candidates)
    dense_ids = dres["ids"][0]
    dense_sim = {cid: round(1 - dist, 4) for cid, dist in zip(dense_ids, dres["distances"][0])}

    # Sparse: BM25 over the same chunks.
    if _bm25 is not None:
        scores = _bm25.get_scores(_tokenize(question))
        order = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[:candidates]
        sparse_ids = [_bm25_ids[k] for k in order if scores[k] > 0]
    else:
        sparse_ids = []

    fused = _rrf([dense_ids, sparse_ids])[:candidates]
    out = []
    for cid in fused:
        rec = _corpus.get(cid)
        if rec:
            out.append({"text": rec["text"], "source": rec["source"], "score": dense_sim.get(cid, 0.0)})
    return out


def stats() -> dict:
    return {"chunks": _collection.count()}


def reset() -> None:
    """Wipe the collection (handy for live demos)."""
    global _collection
    _client.delete_collection(config.COLLECTION_NAME)
    _collection = _client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=_embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
