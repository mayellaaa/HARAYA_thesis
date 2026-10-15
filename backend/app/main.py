from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import mbert
from .pipeline import analyze
from .schemas import AnalyzeRequest, AnalyzeResponse, OcrRequest, OcrResponse
from .vision import transcribe_screenshot

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    loaded = mbert.load_mbert()
    mbert.load_optional_classifier()
    if os.getenv("HARAYA_REQUIRE_MBERT", "false").lower() == "true" and not loaded:
        raise RuntimeError(f"mBERT failed to load: {mbert.mbert_status()['encoder_error']}")
    if loaded:
        logger.info("HARAYA 2.0 mBERT encoder is ready.")
    else:
        logger.warning("HARAYA 2.0 started without mBERT; legal scoring still runs.")
    yield


app = FastAPI(
    title="HARAYA 2.0",
    version="2.0.0",
    description="Rule-guided contextual classification using multilingual BERT and a legal scoring engine.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("HARAYA_ALLOWED_ORIGINS", "http://localhost:5173").split(","),
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    status = mbert.mbert_status()
    return {"ok": True, "version": "2.0.0", "mbert": status}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_conversation(request: AnalyzeRequest) -> AnalyzeResponse:
    if not request.messages:
        raise HTTPException(status_code=400, detail="At least one message is required.")
    return analyze(request)


@app.post("/ocr", response_model=OcrResponse)
def ocr_conversation(request: OcrRequest) -> OcrResponse:
    return transcribe_screenshot(request.data_url)
