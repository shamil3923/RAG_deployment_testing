"""Thin wrapper around the LLM. Swapping providers = change base_url/model in config.py."""
from openai import OpenAI

import config

# timeout so a stalled/unavailable model surfaces an error fast instead of hanging the UI;
# max_retries=0 so we fail on the first timeout rather than silently retrying 3x.
client = OpenAI(
    base_url=config.NVIDIA_BASE_URL,
    api_key=config.NVIDIA_API_KEY,
    timeout=30.0,
    max_retries=0,
)

SYSTEM_PROMPT = """You are a helpful assistant for a question-answering website.
Answer the user's question using ONLY the context passages provided below.
- If the answer is not in the context, say "I don't have enough information to answer that."
- Be concise and factual.
- Cite the passages you used like [1], [2] inline.
"""


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks))
    return f"Context passages:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Single non-streaming completion. Returns the answer text."""
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, chunks)},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()
