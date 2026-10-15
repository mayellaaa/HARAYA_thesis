from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    id: str | None = None
    speaker: str = Field(pattern=r"^User [AB]$")
    role: Literal["sender", "receiver", "participant", "unknown"] = "unknown"
    text: str = Field(min_length=1)
    order: int | None = None


class AnalyzeRequest(BaseModel):
    messages: list[Message]
    source: str = "Manual Input"


class IndicatorEvidence(BaseModel):
    id: str
    label: str
    present: bool
    source: str
    evidence: list[str] = Field(default_factory=list)
    similarity: float | None = None


class CategoryScore(BaseModel):
    label: str
    score: int
    max_score: int
    decision: Literal["Insufficient evidence", "Possible", "Classified"]
    matched_indicators: list[str]
    legal_reference: str


class ProcessedMessage(BaseModel):
    speaker: str
    original: str
    cleaned: str
    normalized: str
    tokens: list[str]
    language: str


class AnalyzeResponse(BaseModel):
    messages: list[Message]
    processed_messages: list[ProcessedMessage]
    conversation_text: str
    language: str
    source: str
    detected: bool
    prediction: str
    confidence: float
    rule_score: int
    rule_max_score: int
    legal_indicators: list[IndicatorEvidence]
    category_scores: list[CategoryScore]
    legal_explanation: str
    mbert: dict
    model: str
    disclaimer: str


class OcrRequest(BaseModel):
    data_url: str


class OcrBubble(BaseModel):
    speaker: str
    text: str
    side: Literal["left", "right"]


class OcrResponse(BaseModel):
    bubbles: list[OcrBubble]
    error: str | None = None
