"""Deterministic entity resolution (Phase IV). No LLM involved, so results are reproducible.

Pipeline: normalise -> exact alias/canonical match -> dynamic dedupe of unseen names ->
guarded fuzzy match -> otherwise register as a new canonical entity.
Every decision is appended to a mapping log (raw -> canonical, method, score).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

try:
    from rapidfuzz import fuzz

    def _ratio(a: str, b: str) -> float:
        return fuzz.ratio(a, b) / 100
except ImportError:  # stdlib fallback keeps the module usable without rapidfuzz
    from difflib import SequenceMatcher

    def _ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

FUZZY_THRESHOLD = 0.93
_LEGAL = {"inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "co", "company",
          "gmbh", "plc", "pbc", "sas", "bv", "lp", "pte", "sa", "ag"}

SEED: dict[str, list[str]] = {
    "OpenAI": ["Open AI", "OpenAI Inc", "OpenAI, Inc.", "OpenAI LP", "OpenAI Global"],
    "Anthropic": ["Anthropic PBC", "Anthropic AI"],
    "Google DeepMind": ["DeepMind", "Deep Mind", "Google Deep Mind"],
    "Mistral AI": ["Mistral", "Mistral.ai"],
    "Cohere": ["Cohere Inc", "Cohere AI"],
    "Hugging Face": ["HuggingFace", "Hugging Face Inc"],
    "Stability AI": ["Stability", "Stability AI Ltd"],
    "Midjourney": ["Midjourney Inc"],
    "Perplexity AI": ["Perplexity", "Perplexity.ai"],
    "Character.AI": ["Character AI", "CharacterAI", "Character"],
    "Inflection AI": ["Inflection"],
    "Adept AI": ["Adept", "Adept AI Labs"],
    "Runway": ["RunwayML", "Runway ML"],
    "Scale AI": ["Scale", "Scale.ai"],
    "Databricks": ["Databricks Inc"],
    "Anyscale": ["Anyscale Inc"],
    "Together AI": ["Together", "Together.ai"],
    "Replicate": ["Replicate Inc"],
    "Weights & Biases": ["Weights and Biases", "W&B", "wandb"],
    "Pinecone": ["Pinecone Systems", "Pinecone.io"],
    "Weaviate": ["Weaviate B.V."],
    "Qdrant": ["Qdrant Solutions"],
    "LangChain": ["Lang Chain", "LangChain Inc"],
    "LlamaIndex": ["Llama Index", "Llama-Index"],
    "Cursor": ["Anysphere", "Anysphere Inc"],
    "Cognition": ["Cognition AI", "Cognition Labs"],
    "Replit": ["Replit Inc"],
    "Glean": ["Glean Technologies"],
    "Jasper": ["Jasper AI"],
    "Writer": ["Writer Inc", "Writer.com"],
    "Synthesia": ["Synthesia Ltd"],
    "ElevenLabs": ["Eleven Labs", "11Labs", "Eleven Labs Inc"],
    "Suno": ["Suno AI", "Suno Inc"],
    "Luma AI": ["Luma", "Luma Labs"],
    "Pika": ["Pika Labs", "Pika Art"],
    "Groq": ["Groq Inc"],
    "Cerebras": ["Cerebras Systems"],
    "SambaNova": ["SambaNova Systems", "Samba Nova"],
    "Lambda": ["Lambda Labs", "Lambda Inc"],
    "CoreWeave": ["Core Weave", "CoreWeave Inc"],
    "Modal": ["Modal Labs"],
    "Fireworks AI": ["Fireworks", "Fireworks.ai"],
    "Sierra": ["Sierra AI"],
    "Harvey": ["Harvey AI"],
    "Hebbia": ["Hebbia Inc"],
    "Abridge": ["Abridge AI"],
    "Hippocratic AI": ["Hippocratic"],
    "xAI": ["x.AI", "X AI", "xAI Corp"],
    "Reka AI": ["Reka", "Reka.ai"],
    "AI21 Labs": ["AI21", "AI 21 Labs"],
    "DeepSeek": ["Deep Seek", "DeepSeek AI"],
    "Moonshot AI": ["Moonshot", "Kimi"],
    "Safe Superintelligence": ["SSI", "Safe Superintelligence Inc"],
    "Thinking Machines Lab": ["Thinking Machines"],
    "Figure AI": ["Figure", "Figure Robotics"],
    "Mercor": ["Mercor Inc"],
    "Decagon": ["Decagon AI"],
}


def tokens(name: str) -> list[str]:
    s = unicodedata.normalize("NFKC", name or "").lower().replace("&", " and ")
    toks = re.sub(r"[^a-z0-9]+", " ", s).split()
    if toks and toks[0] == "the" and len(toks) > 1:
        toks = toks[1:]
    while len(toks) > 1 and toks[-1] in _LEGAL:
        toks.pop()
    return toks


def compact(name: str) -> str:
    return "".join(tokens(name))


def display_clean(name: str) -> str:
    """Readable fallback canonical for unseen names: strip legal suffix + collapse whitespace."""
    s = re.sub(r"\s+", " ", (name or "").strip())
    s = re.sub(r"[,\s]+(inc|incorporated|llc|ltd|limited|corp|corporation|gmbh|plc|pbc)\.?$", "", s, flags=re.I)
    return s.strip(" ,.") or (name or "").strip()


@dataclass
class Resolution:
    raw: str
    canonical: str
    method: str  # exact | alias | dynamic | fuzzy | new
    score: float


class EntityResolver:
    def __init__(self, seed: dict[str, list[str]] | None = None):
        seed = SEED if seed is None else seed
        self._index: dict[str, str] = {}  # compact key -> canonical
        self._seed_keys: dict[str, str] = {}
        for canon, aliases in seed.items():
            self._index[compact(canon)] = canon
            for a in aliases:
                self._index.setdefault(compact(a), canon)
        self._seed_keys = dict(self._index)
        self._dynamic: dict[str, str] = {}  # compact key -> first-seen display name
        self.log: list[dict] = []
        self._logged: set[tuple[str, str]] = set()

    def resolve(self, raw: str, record_type: str = "", source_url: str = "") -> Resolution:
        res = self._resolve(raw)
        key = (raw, record_type)
        if key not in self._logged:
            self._logged.add(key)
            self.log.append({
                "raw_name": raw, "canonical_name": res.canonical, "method": res.method,
                "score": round(res.score, 3), "record_type": record_type, "source_url": source_url,
            })
        return res

    def _resolve(self, raw: str) -> Resolution:
        key = compact(raw)
        if not key:
            return Resolution(raw, (raw or "").strip(), "new", 0.0)
        if key in self._index:
            canon = self._index[key]
            return Resolution(raw, canon, "exact" if raw.strip() == canon else "alias", 1.0)
        if key in self._dynamic:
            return Resolution(raw, self._dynamic[key], "dynamic", 1.0)
        best, best_score = None, 0.0
        for k, canon in list(self._index.items()) + list(self._dynamic.items()):
            if abs(len(k) - len(key)) > 2 or k[0] != key[0]:
                continue  # cheap guards against false positives (Cohere vs Coherent etc.)
            sc = _ratio(key, k)
            if sc > best_score:
                best, best_score = canon, sc
        if best is not None and best_score >= FUZZY_THRESHOLD:
            return Resolution(raw, best, "fuzzy", best_score)
        canon = display_clean(raw)
        self._dynamic[key] = canon
        return Resolution(raw, canon, "new", 0.0)
