from __future__ import annotations

from . import mbert, scoring
from .indicators import detect_indicators
from .preprocess import conversation_document, detect_language, preprocess_messages
from .scoring import load_rules
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    CategoryScore,
    IndicatorEvidence,
    ProcessedMessage,
)

DISCLAIMER = (
    "HARAYA 2.0 is a research prototype. Outputs are not legal advice or a finding "
    "under the Safe Spaces Act. Indicator weights remain draft until legal-expert validation."
)


def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    messages = [item.model_dump() for item in request.messages]
    processed = preprocess_messages(messages)
    document = conversation_document(processed)
    language = detect_language(" ".join(item["original"] for item in processed))

    detected = detect_indicators(processed, document)
    mbert_distribution = mbert.classify_context(document)
    category_scores = scoring.score_categories(detected)
    prediction, winner = scoring.final_prediction(category_scores)
    confidence = scoring.confidence_score(winner, mbert_distribution, prediction)

    rules = load_rules()
    indicator_payload = []
    for indicator_id, spec in rules["indicators"].items():
        info = detected.get(indicator_id, {"present": False, "source": "contextual_cues", "evidence": []})
        indicator_payload.append(
            IndicatorEvidence(
                id=indicator_id,
                label=spec["label"],
                present=bool(info.get("present")),
                source=info.get("source", "contextual_cues"),
                evidence=info.get("evidence") or [],
                similarity=info.get("similarity"),
            )
        )

    return AnalyzeResponse(
        messages=request.messages,
        processed_messages=[ProcessedMessage(**item) for item in processed],
        conversation_text=document,
        language=language,
        source=request.source,
        detected=prediction != "Non-Harassment",
        prediction=prediction,
        confidence=confidence,
        rule_score=winner["score"] if winner else 0,
        rule_max_score=winner["max_score"] if winner else 0,
        legal_indicators=indicator_payload,
        category_scores=[CategoryScore(**item) for item in category_scores],
        legal_explanation=scoring.legal_explanation(prediction, winner, detected),
        mbert={
            **mbert.mbert_status(),
            "role": "contextual evidence only; the rule-based legal scoring engine makes the decision",
            "distribution": mbert_distribution,
        },
        model="HARAYA 2.0 mBERT + rule-based legal scoring engine",
        disclaimer=DISCLAIMER,
    )
