from __future__ import annotations

import os
from pathlib import Path

SCHEMA_VERSION = "1.0"
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CONTACT = os.getenv("CONTACT_EMAIL", "you@example.com")
USER_AGENT = f"GraphOneTrialBot/1.0 (research trial; contact: {CONTACT})"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
FRESH_HOURS = 24

# Multi-tier LLM fallback chain. All three expose OpenAI-compatible chat endpoints,
# so a single client handles them. Tiers without an API key are skipped.
# max_chars is the per-tier input budget (~4 chars/token) used by the truncation strategy.
TIERS = [
    {
        "name": "gemini",
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "key_env": "GEMINI_API_KEY",
        "max_chars": 24000,
        "concurrency": 4,
    },
    {
        "name": "groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "key_env": "GROQ_API_KEY",
        "max_chars": 16000,
        "concurrency": 3,
    },
    {
        "name": "deepseek",
        "url": "https://api.deepseek.com/chat/completions",
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "key_env": "DEEPSEEK_API_KEY",
        "max_chars": 24000,
        "concurrency": 4,
    },
]
