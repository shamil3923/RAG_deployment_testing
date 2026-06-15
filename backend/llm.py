"""Thin wrapper around the LLM. Swapping providers = change base_url/model in config.py."""
import re

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


def rerank(question: str, candidates: list[dict], top_n: int) -> list[dict]:
    """LLM cross-encoder-style rerank: reorder candidates by relevance, keep top_n.

    Cheap fallback to the fused order if the model output can't be parsed.
    """
    if len(candidates) <= top_n:
        return candidates
    listing = "\n".join(f"[{i + 1}] {c['text'][:500]}" for i, c in enumerate(candidates))
    prompt = (
        f"Question: {question}\n\nPassages:\n{listing}\n\n"
        "List the passage numbers most relevant to answering the question, most "
        "relevant first, as a comma-separated list (e.g. 3,1,7). Include only "
        "passages that genuinely help; omit irrelevant ones."
    )
    try:
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "You rank passages by relevance. Reply with only comma-separated numbers."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=80,
        )
        nums = [int(x) for x in re.findall(r"\d+", resp.choices[0].message.content)]
    except Exception:
        nums = []

    ordered, seen = [], set()
    for num in nums:
        idx = num - 1
        if 0 <= idx < len(candidates) and idx not in seen:
            seen.add(idx)
            ordered.append(candidates[idx])
    # Fill any gaps with the remaining candidates in their fused order.
    for j, c in enumerate(candidates):
        if j not in seen:
            ordered.append(c)
    return ordered[:top_n]


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
