"""
TTS Router: Síntesis de Voz OpenAI-compatible (Piper ONNX / Kokoro / Edge-TTS).
"""
import os
import io
import re
import wave
import logging
from typing import Optional, List, Dict, Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("Aoi.Sensory.TTS")
router = APIRouter(tags=["TTS"])

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
PIPER_DIR = os.path.join(MODELS_DIR, "piper")
PIPER_ES_MODEL = os.path.join(PIPER_DIR, "es_ES-sharvard-medium.onnx")
PIPER_ES_CONFIG = os.path.join(PIPER_DIR, "es_ES-sharvard-medium.onnx.json")

piper_voice = None
kokoro_voice = None


def get_piper_voice():
    global piper_voice
    if piper_voice is not None:
        return piper_voice

    try:
        from piper import PiperVoice
        if os.path.exists(PIPER_ES_MODEL) and os.path.exists(PIPER_ES_CONFIG):
            piper_voice = PiperVoice.load(PIPER_ES_MODEL, config_path=PIPER_ES_CONFIG)
            logger.info("Motor Piper TTS inicializado con éxito.")
            return piper_voice
    except Exception as e:
        logger.warning(f"Piper TTS no disponible: {e}")
    return None


class SpeechRequest(BaseModel):
    model: Optional[str] = Field(default="piper", description="Motor o modelo de voz ('piper', 'kokoro', 'edge-tts').")
    input: str = Field(..., description="Texto a sintetizar.")
    voice: Optional[str] = Field(default="es_ES-sharvard-medium", description="Voz seleccionada.")
    response_format: Optional[str] = Field(default="wav", description="Formato de salida ('wav', 'mp3').")
    speed: Optional[float] = Field(default=1.0, description="Velocidad de reproducción (0.25 a 4.0).")


class HealthResponse(BaseModel):
    status: str = "ok"
    default_engine: str = "piper"
    active_engine: str = "piper"
    piper_ready: bool = False
    kokoro_ready: bool = False


def split_sentences_for_streaming(text: str) -> List[str]:
    """Divide un texto en oraciones/cláusulas para síntesis incremental de baja latencia."""
    if not text:
        return []
    # Divide preservando puntuación delimitadora (;, ., ?, !, \n)
    raw_chunks = re.split(r'([;.\?!]+|\n+)', text)
    sentences: List[str] = []
    current = ""
    for part in raw_chunks:
        if not part:
            continue
        current += part
        if re.search(r'[;.\?!]+|\n+', part):
            s = current.strip()
            if s:
                sentences.append(s)
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences


def _write_wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convierte un array de samples numpy (float32 o int16) en un WAV PCM de 16 bits."""
    if samples.dtype in (np.float32, np.float64):
        # Clip y escalar a int16
        clipped = np.clip(samples, -1.0, 1.0)
        int_data = (clipped * 32767.0).astype(np.int16)
    elif samples.dtype == np.int16:
        int_data = samples
    else:
        int_data = np.asarray(samples, dtype=np.int16)

    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_data.tobytes())
    return out.getvalue()


def _read_wav_samples(wav_bytes: bytes) -> tuple:
    """Lee bytes de audio WAV y retorna (samples_array_float32, sample_rate)."""
    with io.BytesIO(wav_bytes) as bio:
        with wave.open(bio, "rb") as wf:
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            # Convertir PCM 16-bit a float32 [-1.0, 1.0]
            int_data = np.frombuffer(frames, dtype=np.int16)
            float_data = int_data.astype(np.float32) / 32767.0
            return float_data, sample_rate


def _generate_wav_bytes(audio_data: bytes, sample_rate: int = 22050) -> bytes:
    """Genera archivo WAV estándar PCM 16-bit a partir de datos raw o array."""
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)
    return out.getvalue()



@router.post("/v1/audio/speech")
async def generate_speech(req: SpeechRequest):
    text = req.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Texto de entrada vacío.")

    # 1. Intentar con Piper TTS
    voice = get_piper_voice()
    if voice is not None:
        try:
            out_buf = io.BytesIO()
            with wave.open(out_buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(getattr(getattr(voice, "config", None), "sample_rate", 22050))
                voice.synthesize(text, wf)
            return Response(content=out_buf.getvalue(), media_type="audio/wav")
        except Exception as e:
            logger.error(f"Error sintetizando con Piper: {e}")

    # 2. Fallback con Edge-TTS si está instalado
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice="es-ES-ElviraNeural")
        audio_stream = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_stream.write(chunk["data"])
        return Response(content=audio_stream.getvalue(), media_type="audio/mpeg")
    except Exception as e:
        logger.warning(f"Edge-TTS fallback no disponible: {e}")

    # 3. Fallback dummy silence si no hay motor instalado
    silent_wav = _generate_wav_bytes(b"\x00" * 4410, sample_rate=22050)
    return Response(content=silent_wav, media_type="audio/wav")


@router.get("/v1/audio/voices")
async def list_voices():
    return {
        "voices": [
            {"id": "es_ES-sharvard-medium", "name": "Sharvard (Piper ES)", "language": "es"},
            {"id": "es-ES-ElviraNeural", "name": "Elvira (Edge Neural)", "language": "es"},
            {"id": "ef_dora", "name": "Dora (Kokoro ES)", "language": "es"}
        ]
    }
