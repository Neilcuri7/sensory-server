"""
TTS Router: High-performance Text-to-Speech router delegating to ITextToSpeechEngine.
Outputs pure 16-bit linear PCM at 22,050 Hz without legacy locks, OLA pitch shift, or RVC.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from adapters.tts.piper_engine import PiperEngine
from domain.ports.text_to_speech import ITextToSpeechEngine

logger = logging.getLogger("Sensory.TTS")
router = APIRouter(tags=["TTS"])

_default_engine: Optional[ITextToSpeechEngine] = None


def get_tts_engine(app=None) -> ITextToSpeechEngine:
    """Retrieve active TTS engine from app state or fallback singleton."""
    global _default_engine
    if app and hasattr(app, "state") and getattr(app.state, "tts_engine", None):
        return app.state.tts_engine
    if _default_engine is None:
        _default_engine = PiperEngine()
    return _default_engine


def set_tts_engine(engine: ITextToSpeechEngine) -> None:
    """Explicitly set or override active TTS engine (e.g. for testing or lifespan injection)."""
    global _default_engine
    _default_engine = engine


class SpeechRequest(BaseModel):
    model_config = {"extra": "ignore"}

    input: Optional[str] = None
    text: Optional[str] = None
    model: Optional[str] = "piper"
    voice: Optional[str] = "piper_f"
    speed: Optional[float] = 1.05
    generation_id: Optional[str] = None


class AbortRequest(BaseModel):
    model_config = {"extra": "ignore"}

    generation_id: Optional[str] = None


@router.post("/v1/audio/speech")
@router.post("/tts")
async def generate_speech(req: SpeechRequest, request: Request):
    """Synthesize speech returning pure 16-bit linear PCM at 22,050 Hz."""
    text = (req.input or req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Texto de entrada vacío.")

    engine = get_tts_engine(request.app)
    if not engine.is_ready():
        raise HTTPException(status_code=503, detail="TTS engine not ready")

    speed = float(req.speed) if req.speed is not None else 1.05

    try:
        result = await engine.synthesize(
            text=text,
            speed=speed,
            generation_id=req.generation_id,
        )
    except asyncio.CancelledError:
        raise HTTPException(status_code=499, detail="Client closed request or task aborted")
    except Exception as exc:
        logger.error(f"TTS synthesis failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return Response(
        content=result.pcm_bytes,
        media_type="audio/pcm; rate=22050",
        headers={
            "Content-Type": "audio/pcm; rate=22050",
            "X-TTS-Engine": "piper",
            "X-TTS-Voice": req.voice or "piper_f",
            "X-TTS-Sample-Rate": str(result.sample_rate),
            "X-TTS-Duration-Ms": str(int(result.duration_ms)),
        },
    )


@router.post("/v1/audio/speech/abort")
@router.post("/tts/abort")
async def abort_speech(
    request: Request,
    req: Optional[AbortRequest] = None,
    generation_id: Optional[str] = None,
):
    """Abort in-flight or queued synthesis tasks matching generation_id."""
    gen_id = (req.generation_id if req and req.generation_id else generation_id) or ""
    if not gen_id:
        return {"status": "ok", "message": "No generation_id provided"}

    engine = get_tts_engine(request.app)
    await engine.abort(gen_id)
    return {"status": "ok", "aborted": gen_id}


@router.get("/v1/audio/voices")
@router.get("/tts/voices")
async def list_voices():
    """List available voices and default synthesis engine."""
    return {
        "voices": [
            {
                "id": "piper_f",
                "name": "Piper Voice (Spanish Female - CPU)",
                "engine": "piper",
                "sample_rate": 22050,
            }
        ],
        "default_engine": "piper",
        "default_voice": "piper_f",
    }
