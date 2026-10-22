from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "legal_rules.json"


@lru_cache(maxsize=1)
def load_rules() -> dict[str, Any]:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def decision_for(score: int, thresholds: dict[str, int]) -> str:
    if score >= thresholds["classified"]:
        return "Classified"
    if score >= thresholds["possible"]:
        return "Possible"
    return "Insufficient evidence"


def score_categories(detected: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rules = load_rules()
    results: list[dict[str, Any]] = []
    present = {key for key, value in detected.items() if value.get("present")}
    for label, spec in rules["categories"].items():
        if label == "Non-Harassment":
            continue
        core_groups = spec["core_indicator_groups"]
        eligible = bool(core_groups) and all(
            any(indicator in present for indicator in group)
            for group in core_groups
        )
        matched = [key for key in spec["indicators"] if eligible and key in present]
        score = sum(spec["indicators"][key] for key in matched)
        results.append(
            {
                "label": label,
                "score": score,
                "max_score": spec["max_score"],
                "decision": decision_for(score, spec["thresholds"]),
                "matched_indicators": [rules["indicators"][key]["label"] for key in matched],
                "legal_reference": spec["legal_reference"],
            }
        )
    results.sort(key=lambda item: (item["decision"] != "Classified", -item["score"], item["label"]))
    return results


# More specific Safe Spaces Act mappings are preferred when several categories clear the legal threshold.
CATEGORY_PRIORITY = [
    "Physical Threat",
    "Unauthorized Sharing of Personal Information",
    "Cyberstalking",
    "Unwanted Sexual Remarks/Comments",
    "Misogynistic Remarks/Comments",
    "Homophobic Remarks/Comments",
    "Transphobic Remarks/Comments",
    "Sexist Remarks/Comments",
    "Psychological / Emotional Threat",
    "Incessant Messaging",
]


def _rank(item: dict[str, Any]) -> tuple[int, int]:
    try:
        priority = CATEGORY_PRIORITY.index(item["label"])
    except ValueError:
        priority = len(CATEGORY_PRIORITY)
    return (priority, -item["score"])


def final_prediction(category_scores: list[dict[str, Any]]) -> tuple[str, dict[str, Any] | None]:
    classified = [item for item in category_scores if item["decision"] == "Classified"]
    if classified:
        winner = sorted(classified, key=_rank)[0]
        return winner["label"], winner
    possible = [item for item in category_scores if item["decision"] == "Possible"]
    if possible:
        winner = sorted(possible, key=_rank)[0]
        return winner["label"], winner
    return "Non-Harassment", None


def confidence_score(
    winner: dict[str, Any] | None,
    mbert_distribution: list[dict[str, Any]],
    prediction: str,
) -> float:
    if winner is None:
        top = mbert_distribution[0]["score"] if mbert_distribution else 0.55
        return round(min(0.95, max(0.4, 0.55 + (0.2 if prediction == "Non-Harassment" else 0) * top)), 4)

    coverage = winner["score"] / max(winner["max_score"], 1)
    agreement = 0.0
    for item in mbert_distribution:
        if item["label"] == prediction:
            agreement = item["score"]
            break
    mixed = 0.62 * coverage + 0.28 * agreement + 0.1
    if winner["decision"] == "Possible":
        mixed *= 0.85
    return round(min(0.97, max(0.42, mixed)), 4)


def legal_explanation(
    prediction: str,
    winner: dict[str, Any] | None,
    detected: dict[str, dict[str, Any]],
) -> str:
    rules = load_rules()
    if prediction == "Non-Harassment" or winner is None:
        return (
            "The conversation-level legal indicators do not satisfy the operational "
            "elements required for a gender-based online harassment category under the "
            "draft Safe Spaces Act mapping."
        )
    present_labels = [
        rules["indicators"][key]["label"]
        for key, value in detected.items()
        if value.get("present") and key in rules["indicators"]
    ]
    return (
        f"The conversation was evaluated as a whole against the legal elements for "
        f"{prediction}. Rule score {winner['score']} / {winner['max_score']} "
        f"({winner['decision']}). Supporting indicators: {', '.join(present_labels) or 'none'}. "
        f"{winner['legal_reference']}"
    )
