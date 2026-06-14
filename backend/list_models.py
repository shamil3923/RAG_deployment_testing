"""Utility: print the chat models available on your NVIDIA key. Run: python list_models.py"""
from openai import OpenAI
import config

c = OpenAI(base_url=config.NVIDIA_BASE_URL, api_key=config.NVIDIA_API_KEY)
ids = sorted(m.id for m in c.models.list().data)
print(f"{len(ids)} models available. Qwen models:")
for i in ids:
    if "qwen" in i.lower():
        print(" ", i)
