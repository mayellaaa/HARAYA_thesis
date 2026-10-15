from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

_tokenizer = None
_model = None
_classifier = None
_load_error: str | None = None
_classifier_error: str | None = None


def mbert_status() -> dict[str, Any]:
    return {
        "encoder_loaded": _model is not None,
        "classifier_loaded": _classifier is not None,
        "encoder_error": _load_error,
        "classifier_error": _classifier_error,
        "model_id": os.getenv("HARAYA_MBERT_ID", "google-bert/bert-base-multilingual-cased"),
        "classifier_id": os.getenv("HARAYA_CLASSIFIER_ID", "") or None,
    }


def load_mbert() -> bool:
    """Load mBERT as a contextual encoder. Classification is optional evidence only."""
    global _tokenizer, _model, _load_error
    if _model is not None:
        return True
    model_id = os.getenv("HARAYA_MBERT_ID", "google-bert/bert-base-multilingual-cased")
    try:
        from transformers import AutoModel, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModel.from_pretrained(model_id)
        _model.eval()
        _load_error = None
        logger.info("Loaded mBERT encoder: %s", model_id)
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        _load_error = str(exc)
        logger.warning("mBERT encoder unavailable: %s", exc)
        return False


def load_optional_classifier() -> None:
    global _classifier, _classifier_error
    classifier_id = os.getenv("HARAYA_CLASSIFIER_ID", "").strip()
    if not classifier_id or _classifier is not None:
        return
    try:
        from transformers import pipeline

        _classifier = pipeline(
            "text-classification",
            model=classifier_id,
            tokenizer=classifier_id,
            top_k=None,
            function_to_apply="softmax",
        )
        _classifier_error = None
        logger.info("Loaded optional HARAYA classifier as contextual evidence: %s", classifier_id)
    except Exception as exc:  # pragma: no cover
        _classifier_error = str(exc)
        logger.warning("Optional classifier unavailable: %s", exc)


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed(texts: list[str]) -> torch.Tensor | None:
    if _model is None or _tokenizer is None or not texts:
        return None
    encoded = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = _model(**encoded)
        vectors = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
        vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
    return vectors


def cosine_scores(conversation: str, descriptions: dict[str, str]) -> dict[str, float]:
    if not conversation or not descriptions:
        return {key: 0.0 for key in descriptions}
    vectors = embed([conversation, *descriptions.values()])
    if vectors is None:
        return {key: 0.0 for key in descriptions}
    conv = vectors[0]
    scores = {}
    for index, key in enumerate(descriptions):
        scores[key] = float(torch.dot(conv, vectors[index + 1]).item())
    return scores


def classify_context(text: str) -> list[dict[str, Any]]:
    if _classifier is None or not text:
        return []
    raw = _classifier(text[:2000])
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]
    results = []
    for item in raw or []:
        if isinstance(item, dict) and "label" in item:
            results.append({"label": item["label"], "score": float(item.get("score") or 0)})
    return sorted(results, key=lambda item: item["score"], reverse=True)
