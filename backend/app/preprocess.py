from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "lexicons.json"

FILIPINO_CUES = (
    "ako", "mo", "ka", "ko", "ang", "ng", "sa", "na", "hindi", "huwag", "ayoko",
    "babae", "lalaki", "salamat", "saktan", "malandi", "please stop na",
)


@lru_cache(maxsize=1)
def load_lexicons() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def clean_text(text: str) -> str:
    text = (text or "").replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _apply_mapping(text: str, mapping: dict[str, str]) -> str:
    updated = text
    for term in sorted(mapping, key=len, reverse=True):
        updated = re.sub(
            rf"(?<!\w){re.escape(term)}(?!\w)",
            f" {mapping[term]} ",
            updated,
            flags=re.IGNORECASE,
        )
    return updated


def normalize_text(text: str) -> str:
    lex = load_lexicons()
    normalized = clean_text(text)
    for emoji, meaning in lex["emoji"].items():
        normalized = normalized.replace(emoji, f" {meaning} ")
    for emoticon, meaning in lex["emoticons"].items():
        normalized = normalized.replace(emoticon, f" {meaning} ")
    normalized = normalized.lower()
    for table in ("gay_lingo", "slang", "abbreviations", "tagalog"):
        normalized = _apply_mapping(normalized, lex[table])
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+|[^\s\w]", text, flags=re.UNICODE)


def detect_language(text: str) -> str:
    lowered = text.lower()
    if any(re.search(rf"\b{re.escape(cue)}\b", lowered) for cue in FILIPINO_CUES):
        return "Taglish"
    return "English"


def preprocess_messages(messages: list[dict]) -> list[dict]:
    processed = []
    for message in messages:
        original = message["text"]
        cleaned = clean_text(original)
        normalized = normalize_text(original)
        processed.append(
            {
                "speaker": message["speaker"],
                "original": original,
                "cleaned": cleaned,
                "normalized": normalized,
                "tokens": tokenize(cleaned),
                "language": detect_language(original),
            }
        )
    return processed


def conversation_document(processed: list[dict]) -> str:
    lines = [f"{item['speaker']}: {item['normalized']}" for item in processed]
    return " [SEP] ".join(lines)
