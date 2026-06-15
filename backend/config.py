"""Central config. All knobs live here so you can change them fast during the interview."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM (NVIDIA-hosted, OpenAI-compatible API) ---
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.3-70b-instruct")
# Vision model used to read images and scanned PDF pages.
VISION_MODEL = os.getenv("VISION_MODEL", "meta/llama-3.2-11b-vision-instruct")

# --- Embeddings (NVIDIA-hosted, retrieval-tuned, 1024-dim) ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "32"))     # texts per embedding request

# --- RAG ---
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
# v2 = new embedding model (1024-dim); kept separate from the old 384-dim "docs".
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docs_v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))      # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # overlap between chunks
TOP_K = int(os.getenv("TOP_K", "5"))                   # final chunks sent to the LLM

# --- Retrieval (hybrid + rerank + grounding guard) ---
RETRIEVE_CANDIDATES = int(os.getenv("RETRIEVE_CANDIDATES", "20"))  # pool before rerank
RRF_K = int(os.getenv("RRF_K", "60"))                  # reciprocal-rank-fusion constant
# If the best dense similarity is below this, refuse instead of answering (anti-hallucination).
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.25"))
