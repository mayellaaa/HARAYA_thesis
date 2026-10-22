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


def _apply_mappings_once(text: str, mappings: dict[str, str]) -> str:
    terms = sorted(mappings, key=len, reverse=True)
    pattern = re.compile(
        rf"(?<!\w)(?:{'|'.join(re.escape(term) for term in terms)})(?!\w)",
        flags=re.IGNORECASE,
    )
    return pattern.sub(lambda match: f" {mappings[match.group(0).casefold()]} ", text)


def _replace_symbols(text: str, mappings: dict[str, str]) -> str:
    terms = sorted(mappings, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(term) for term in terms))
    return pattern.sub(lambda match: f" {mappings[match.group(0)]} ", text)


def normalize_text(text: str) -> str:
    lex = load_lexicons()
    normalized = clean_text(text)
    normalized = _replace_symbols(normalized, lex["emoji"])
    normalized = _replace_symbols(normalized, lex["emoticons"])
    normalized = normalized.lower()
    mappings = {
        term.casefold(): meaning
        for table in ("sexual_slang", "gay_lingo", "slang", "abbreviations", "tagalog")
        for term, meaning in lex[table].items()
    }
    normalized = _apply_mappings_once(normalized, mappings)
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
