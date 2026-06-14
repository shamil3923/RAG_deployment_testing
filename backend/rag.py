"""RAG core: chunk -> embed -> store in Chroma -> retrieve by similarity.

Embeddings use Chroma's built-in DefaultEmbeddingFunction (all-MiniLM-L6-v2 via ONNX),
so retrieval works with zero extra API keys. Generation uses the NVIDIA Qwen model.
"""
import uuid

import chromadb
from chromadb.utils import embedding_functions

import config

_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
_embed_fn = embedding_functions.DefaultEmbeddingFunction()
_collection = _client.get_or_create_collection(
    name=config.COLLECTION_NAME,
    embedding_function=_embed_fn,
    metadata={"hnsw:space": "cosine"},
)


def chunk_text(text: str, size: int = None, overlap: int = None) -> list[str]:
    """Simple sliding-window chunker over characters."""
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    text = text.strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def add_document(text: str, source: str) -> int:
    """Chunk a document and add it to the vector store. Returns chunk count."""
    chunks = chunk_text(text)
    if not chunks:
        return 0
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": source, "chunk": i} for i in range(len(chunks))]
    _collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


def retrieve(question: str, top_k: int = None) -> list[dict]:
    """Return the top_k most similar chunks for a question."""
    top_k = top_k or config.TOP_K
    if _collection.count() == 0:
        return []
    res = _collection.query(query_texts=[question], n_results=top_k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    return [
        {"text": d, "source": m.get("source"), "score": round(1 - dist, 4)}
        for d, m, dist in zip(docs, metas, dists)
    ]


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
