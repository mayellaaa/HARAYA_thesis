from __future__ import annotations

import re
from typing import Any

from . import mbert

NEGATION = re.compile(
    r"\b(?:don't|do not|didn't|did not|won't|will not|never|not|can't|cannot|"
    r"hindi|di|wag|huwag|ayoko|ayaw)\b",
    re.IGNORECASE,
)

REFUSAL_PATTERNS = [
    r"\bplease stop\b",
    r"\bstop messaging\b",
    r"\bstop na\b",
    r"\btumigil ka\b",
    r"\bhuwag mo\b.{0,40}\b(i-?message|text|contact|kausap)",
    r"\bayoko na\b",
    r"\bi (already )?blocked you\b",
    r"\bblocked you\b",
    r"\bdo not (message|contact|text)\b",
    r"\bleave me alone\b",
    r"\bthis is harassment\b",
]

CUE_PATTERNS: dict[str, list[str]] = {
    "physical_harm": [
        r"\b(kill|hurt|harm|punch|hit|attack|saktan|patayin|papatayin|susuntukin|sasaktan)\b.{0,20}\b(you|ka|kita)\b",
        r"\b(you|ka|kita)\b.{0,20}\b(kill|hurt|harm|punch|saktan|patayin)\b",
        r"\bkill yourself\b",
        r"\bsasaktan kita\b",
        r"\bsusuntukin kita\b",
    ],
    "sexual_context": [
        r"\bnudes?\b",
        r"\bsexy\b",
        r"\bnaked\b",
        r"\bhubad\b",
        r"\bprivate photo\b",
        r"\bsend me your (picture|photo|body)\b",
        r"\bsexually suggestive\b",
        r"\bmalandi\b",
    ],
    "misogynistic_language": [
        r"\b(bitch|slut|whore|puta)\b",
        r"\bmalandi ka\b",
        r"\bwomen belong\b",
        r"\bwomen are inferior\b",
        r"\bbabae ka lang\b",
    ],
    "sexist_language": [
        r"\bgirls can't\b",
        r"\bwomen can't\b",
        r"\bmen are better than women\b",
        r"\bwomen are too emotional\b",
        r"\bmen should lead\b",
        r"\bhindi bagay sa babae\b",
        r"\blalaki lang ang bagay\b",
        r"\bwomen like you should stay quiet\b",
        r"\bbabae ka so\b",
    ],
    "homophobic_language": [
        r"\bfag\b",
        r"\bdyke\b",
        r"\bbakla ka\b",
        r"\bno homo\b",
        r"\bgay people should not\b",
        r"\bdon't respect gay\b",
    ],
    "transphobic_language": [
        r"\btranny\b",
        r"\bnot a real (woman|man)\b",
        r"\btrans(gender)? people are not valid\b",
        r"\bdon't respect (your )?gender identity\b",
    ],
    "monitoring_or_pursuit": [
        r"\bstalk(ing)? you\b",
        r"\bwatching you\b",
        r"\btrack(ing)? you\b",
        r"\bi know where you live\b",
        r"\balam ko kung saan ka nakatira\b",
        r"\bmonitor(ing)? (your )?(posts|profile|activity|accounts?)\b",
        r"\bfollow(ing)? (your|every)\b",
        r"\bsinusubaybayan kita\b",
        r"\bminomonitor ko\b",
        r"\bwatching\b",
    ],
    "multiple_accounts": [
        r"\banother account\b",
        r"\bibang account\b",
        r"\bmake another account\b",
        r"\bnew account\b",
        r"\bfrom another account\b",
    ],
    "intimidation": [
        r"\byou will regret\b",
        r"\bmake you regret\b",
        r"\bruin your (life|reputation)\b",
        r"\bembarrass you\b",
        r"\bipapahiya kita\b",
        r"\bgets\?\b",
        r"\bor else\b",
    ],
    "fear_or_coercion": [
        r"\bnatatakot\b",
        r"\bi am (scared|afraid|frightened)\b",
        r"\bnakakatakot\b",
        r"\bthis is harassment\b",
        r"\bleave me alone\b",
    ],
    "actual_disclosure": [
        r"\bposted your (number|address|information)\b",
        r"\bshared your (number|address|information|private)\b",
        r"\bi posted\b.{0,40}\b(number|address|information)\b",
    ],
    "threatened_disclosure": [
        r"\b(will|isi-share|ipapakalat).{0,40}\b(personal information|private information|number|address)\b",
        r"\bshare your (private|personal) information\b",
        r"\bdoxx?\b",
        r"\bpost your (number|address|personal information)\b",
    ],
}

GENDER_CUES = [
    r"\bwom[ae]n\b",
    r"\bgirl(s)?\b",
    r"\bbabae\b",
    r"\blalaki\b",
    r"\bgay\b",
    r"\btrans\b",
    r"\bgender\b",
    r"\bmalandi\b",
]


def _negated(text: str, match: re.Match[str]) -> bool:
    window = text[max(0, match.start() - 40) : match.start()]
    return bool(NEGATION.search(window))


def _matches(text: str, patterns: list[str]) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            if _negated(text, match):
                continue
            snippet = re.sub(r"\s+", " ", match.group(0)).strip()
            if snippet and snippet not in found:
                found.append(snippet)
    return found


def detect_structural_indicators(processed: list[dict]) -> dict[str, dict[str, Any]]:
    speakers = [item["speaker"] for item in processed]
    texts = [item["normalized"] for item in processed]
    originals = [item["original"] for item in processed]
    joined = " ".join(texts)

    counts: dict[str, int] = {}
    for speaker in speakers:
        counts[speaker] = counts.get(speaker, 0) + 1
    majority_speaker = max(counts, key=counts.get) if counts else "User A"
    majority_count = counts.get(majority_speaker, 0)

    refusal_indexes: list[int] = []
    for index, text in enumerate(texts):
        if _matches(text, REFUSAL_PATTERNS):
            refusal_indexes.append(index)

    continued = False
    continued_evidence: list[str] = []
    if refusal_indexes:
        first_refusal = refusal_indexes[0]
        later = [
            originals[index]
            for index in range(first_refusal + 1, len(processed))
            if processed[index]["speaker"] != processed[first_refusal]["speaker"]
        ]
        continued = len(later) >= 1
        continued_evidence = later[:3]

    fear_from_target = False
    fear_evidence: list[str] = []
    for item in processed:
        hits = _matches(item["normalized"], CUE_PATTERNS["fear_or_coercion"])
        if hits:
            fear_from_target = True
            fear_evidence.extend(hits)

    return {
        "electronic_communication": {
            "present": True,
            "source": "system",
            "evidence": ["Conversation submitted as electronic messages."],
        },
        "repeated_communication": {
            "present": majority_count >= 3 or len(processed) >= 4,
            "source": "conversation_structure",
            "evidence": [f"{majority_speaker} sent {majority_count} of {len(processed)} messages."],
        },
        "continued_after_refusal": {
            "present": continued,
            "source": "conversation_structure",
            "evidence": continued_evidence or (["Refusal detected but no later contact."] if refusal_indexes else []),
        },
        "unwanted_contact": {
            "present": bool(refusal_indexes) or continued,
            "source": "conversation_structure",
            "evidence": [originals[index] for index in refusal_indexes[:3]],
        },
        "fear_or_coercion": {
            "present": fear_from_target or bool(_matches(joined, CUE_PATTERNS["fear_or_coercion"])),
            "source": "conversation_structure",
            "evidence": fear_evidence,
        },
    }


def detect_cue_indicators(processed: list[dict]) -> dict[str, dict[str, Any]]:
    joined_original = " ".join(item["original"] for item in processed)
    joined_normalized = " ".join(item["normalized"] for item in processed)
    combined = f"{joined_original}\n{joined_normalized}"
    detected: dict[str, dict[str, Any]] = {}
    for indicator, patterns in CUE_PATTERNS.items():
        if indicator == "fear_or_coercion":
            continue
        hits = _matches(combined, patterns)
        detected[indicator] = {
            "present": bool(hits),
            "source": "contextual_cues",
            "evidence": hits[:5],
        }

    gender_hits = _matches(combined, GENDER_CUES)
    bias_present = any(
        detected.get(key, {}).get("present")
        for key in (
            "misogynistic_language",
            "sexist_language",
            "homophobic_language",
            "transphobic_language",
            "sexual_context",
        )
    )
    detected["gender_based_targeting"] = {
        "present": bool(gender_hits) or bias_present,
        "source": "contextual_cues",
        "evidence": gender_hits[:5],
    }
    return detected


def merge_mbert_indicators(
    detected: dict[str, dict[str, Any]],
    similarities: dict[str, float],
    threshold: float = 0.42,
) -> dict[str, dict[str, Any]]:
    for indicator, score in similarities.items():
        current = detected.setdefault(
            indicator,
            {"present": False, "source": "mbert_similarity", "evidence": []},
        )
        current["similarity"] = round(score, 4)
        if score >= threshold and not current["present"]:
            current["present"] = True
            current["source"] = "mbert_similarity"
            current["evidence"] = [f"mBERT contextual similarity {score:.2f} to the legal indicator definition."]
    return detected


def _combine(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    evidence = list(dict.fromkeys([*(left.get("evidence") or []), *(right.get("evidence") or [])]))
    present = bool(left.get("present") or right.get("present"))
    source = left.get("source") if left.get("present") else right.get("source")
    combined = {**right, **left, "present": present, "evidence": evidence, "source": source or "contextual_cues"}
    if "similarity" in right:
        combined["similarity"] = right["similarity"]
    return combined


def detect_indicators(processed: list[dict], conversation_text: str) -> dict[str, dict[str, Any]]:
    detected = detect_structural_indicators(processed)
    for key, value in detect_cue_indicators(processed).items():
        detected[key] = _combine(detected[key], value) if key in detected else value

    descriptions = {
        "physical_harm": "A speaker threatens to physically hurt, punch, attack, or kill the other person.",
        "sexual_context": "A speaker makes unwanted sexual comments or asks for sexual photos.",
        "gender_based_targeting": "The speaker targets the person because they are a woman, man, gay, or transgender.",
        "misogynistic_language": "The speaker uses hateful or degrading language toward women.",
        "sexist_language": "The speaker says women are inferior or should not lead.",
        "homophobic_language": "The speaker attacks someone for being gay.",
        "transphobic_language": "The speaker denies or mocks a transgender identity.",
        "monitoring_or_pursuit": "The speaker watches, follows, tracks, or monitors the other person online.",
        "multiple_accounts": "The speaker will create another account to keep contacting the person.",
        "intimidation": "The speaker tries to frighten or control the other person with threats.",
        "fear_or_coercion": "The other person is afraid, distressed, or pressured.",
        "actual_disclosure": "Private personal information was already shared without consent.",
        "threatened_disclosure": "The speaker threatens to share private personal information.",
        "repeated_communication": "The speaker keeps sending many messages.",
        "continued_after_refusal": "The speaker keeps contacting someone after being told to stop.",
        "unwanted_contact": "Messages continue even though they are unwanted.",
    }
    similarities = mbert.cosine_scores(conversation_text, descriptions)
    return merge_mbert_indicators(detected, similarities)
