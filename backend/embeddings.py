"""NVIDIA-hosted embeddings (nv-embedqa-e5-v5, 1024-dim).

This model is *asymmetric*: documents must be encoded with input_type="passage"
and search queries with input_type="query". We give Chroma a passage-encoding
embedding function for ingestion and encode queries ourselves in rag.retrieve().
"""
from chromadb import Documents, EmbeddingFunction, Embeddings

import config
import llm  # reuse the single OpenAI client pointed at NVIDIA


def _embed(texts: list[str], input_type: str) -> list[list[float]]:
    """Embed a list of texts, batching to stay within request limits."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), config.EMBED_BATCH):
        batch = texts[i:i + config.EMBED_BATCH]
        resp = llm.client.embeddings.create(
            model=config.EMBED_MODEL,
            input=batch,
            # NVIDIA-specific fields passed straight through to the request body.
            extra_body={"input_type": input_type, "truncate": "END"},
        )
        vectors.extend(d.embedding for d in resp.data)
    return vectors


def embed_query(text: str) -> list[float]:
    return _embed([text], "query")[0]


class PassageEmbeddingFunction(EmbeddingFunction):
    """Chroma calls this when documents are added -> encode them as passages."""

    def __call__(self, input: Documents) -> Embeddings:
        return _embed(list(input), "passage")
