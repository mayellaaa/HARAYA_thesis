from __future__ import annotations

import json
import os
import re

import httpx

from .schemas import OcrBubble, OcrResponse

MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]

PROMPT = """You transcribe screenshots of conversations and reconstruct their individual message bubbles.
Transcription only — do not classify harassment.

Read only visible text. Do not invent, translate, or correct wording.
Return JSON: {"bubbles":[{"side":"left"|"right","text":"..."}]} in top-to-bottom order.
Preserve Filipino, Taglish, slang, emoji, and emoticons exactly.
If nothing is readable, return {"bubbles":[]}.
"""


def transcribe_screenshot(data_url: str) -> OcrResponse:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return OcrResponse(bubbles=[], error="GEMINI_API_KEY is not set.")

    match = re.match(r"^data:(image/[\w.+-]+);base64,(.+)$", data_url)
    if not match:
        return OcrResponse(bubbles=[], error="Image must be a data URL.")
    mime, payload = match.group(1), match.group(2)

    last_error = "Gemini request failed."
    for model in MODELS:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            f"?key={api_key}"
        )
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": PROMPT},
                        {"inline_data": {"mime_type": mime, "data": payload}},
                    ]
                }
            ],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        try:
            response = httpx.post(url, json=body, timeout=30)
            response.raise_for_status()
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(text)
            bubbles = []
            for item in parsed.get("bubbles") or []:
                side = "left" if item.get("side") == "left" else "right"
                speaker = "User B" if side == "left" else "User A"
                text_value = str(item.get("text") or "").strip()
                if text_value:
                    bubbles.append(OcrBubble(speaker=speaker, text=text_value, side=side))
            return OcrResponse(bubbles=bubbles)
        except Exception as exc:
            last_error = str(exc)
            continue
    return OcrResponse(bubbles=[], error=last_error)
