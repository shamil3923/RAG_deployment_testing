"""Central config. All knobs live here so you can change them fast during the interview."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM (NVIDIA-hosted Qwen, OpenAI-compatible API) ---
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3-next-80b-a3b-instruct")

# --- RAG ---
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))      # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))  # overlap between chunks
TOP_K = int(os.getenv("TOP_K", "4"))                  # chunks retrieved per query
