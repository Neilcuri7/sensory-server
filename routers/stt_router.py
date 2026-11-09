"""
STT Router: High-performance Speech-to-Text router delegating to ISpeechToTextEngine.
Maintains /v1/audio/transcriptions and /stt endpoints.
"""
from __future__ import annotations

import base64
import io
import logging
import wave
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from adapters.stt.sherpa_whisper import SherpaWhisperAdapter
from domain.ports.speech_to_text import ISpeechToTextEngine

logger = logging.getLogger("Sensory.STT")
router = APIRouter()

_default_engine: Optional[ISpeechToTextEngine] = None


def get_stt_engine(app=None) -> ISpeechToTextEngine:
    """Retrieve active STT engine from app state or fallback singleton."""
    global _default_engine
    if app and hasattr(app, "state") and getattr(app.state, "stt_engine", None):
        return app.state.stt_engine
    if _default_engine is None:
        _default_engine = SherpaWhisperAdapter()
    return _default_engine


def set_stt_engine(engine: ISpeechToTextEngine) -> None:
    """Explicitly set or override active STT engine (e.g. for testing)."""
    global _default_engine
    _default_engine = engine


def _extract_pcm_and_sample_rate(audio_bytes: bytes) -> tuple[bytes, int]:
    """Extract raw 16-bit linear PCM and sample rate from audio bytes (WAV or raw PCM)."""
    if len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                sr = wf.getframerate()
                nchannels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())
                if sampwidth == 2 and nchannels == 1:
                    return frames, sr
                import soundfile as sf
                data, read_sr = sf.read(io.BytesIO(audio_bytes), dtype="int16")
                if len(data.shape) > 1:
                    data = data.mean(axis=1).astype(np.int16)
                return data.tobytes(), read_sr
        except Exception as exc:
            logger.warning(f"Failed to parse WAV header: {exc}, treating as raw PCM")
    return audio_bytes, 16000


class TranscriptionResponse(BaseModel):
    text: str


@router.post(
    "/v1/audio/transcriptions",
    response_model=TranscriptionResponse,
    tags=["Audio V1"],
    summary="Transcribe Audio File (OpenAI standard)",
    description="Standard multipart/form-data audio file transcription compatible with OpenAI Audio API.",
)
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = Form(default="whisper-base"),
    language: Optional[str] = Form(default="es"),
    temperature: Optional[float] = Form(default=0.0),
):
    """OpenAI-compatible transcription endpoint."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Archivo de audio vacío.")

    engine = get_stt_engine(request.app)
    if not engine.is_ready():
        raise HTTPException(status_code=503, detail="STT model not loaded or unavailable")

    pcm_bytes, sample_rate = _extract_pcm_and_sample_rate(audio_bytes)

    try:
        result = await engine.transcribe(pcm_bytes, sample_rate=sample_rate)
    except RuntimeError as r_err:
        logger.error(f"[STT Unavailable] {r_err}")
        raise HTTPException(status_code=503, detail=str(r_err))
    except Exception as exc:
        logger.error(f"Error en transcripción: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    logger.info(
        f"[STT Ingest] {len(pcm_bytes)}B PCM, {sample_rate}Hz | "
        f"duration={result.duration_ms:.1f}ms | text='{result.text[:60]}'"
    )
    return TranscriptionResponse(text=result.text)


class STTJsonRequest(BaseModel):
    model_config = {"extra": "ignore"}

    audio_base64: str
    sample_rate: Optional[int] = 16000
    language: Optional[str] = "es"


@router.post(
    "/v1/audio/transcribe-json",
    tags=["Audio V1"],
    summary="Transcribe Base64 JSON Audio (V1)",
    description="Canonical V1 endpoint for transcribing base64-encoded audio within a JSON payload.",
)
@router.post(
    "/stt",
    deprecated=True,
    tags=["Legacy (Deprecated)"],
    summary="Transcribe Base64 JSON Audio (Deprecated)",
    description="Deprecated alias for /v1/audio/transcribe-json. Maintained for backward compatibility.",
)
async def transcribe_json(req: STTJsonRequest, request: Request):
    """JSON Base64-based transcription endpoint."""
    raw_b64 = req.audio_base64
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        audio_bytes = base64.b64decode(raw_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio base64 vacío.")

    engine = get_stt_engine(request.app)
    if not engine.is_ready():
        raise HTTPException(status_code=503, detail="STT model not loaded or unavailable")

    pcm_bytes, sr = _extract_pcm_and_sample_rate(audio_bytes)
    sample_rate = sr if sr != 16000 or audio_bytes[:4] == b"RIFF" else (req.sample_rate or 16000)

    try:
        result = await engine.transcribe(pcm_bytes, sample_rate=sample_rate)
    except RuntimeError as r_err:
        logger.error(f"[STT Unavailable] {r_err}")
        raise HTTPException(status_code=503, detail=str(r_err))
    except Exception as exc:
        logger.error(f"Error en transcripción JSON: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "status": "ok",
        "text": result.text,
        "confidence": result.confidence,
        "is_speech": len(result.text) > 0,
        "sample_rate": sample_rate,
        "duration_sec": result.duration_ms / 1000.0,
    }
