"""
STT Router: Transcripción de Audio OpenAI-compatible (Sherpa-ONNX / Whisper).
"""
import os
import io
import re
import sys
import logging
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("Aoi.Sensory.STT")
router = APIRouter(tags=["STT"])

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
WHISPER_BASE_DIR = os.path.join(MODELS_DIR, "sherpa-onnx-whisper-base")
WHISPER_BASE_ENCODER = os.path.join(WHISPER_BASE_DIR, "base-encoder.int8.onnx")
WHISPER_BASE_DECODER = os.path.join(WHISPER_BASE_DIR, "base-decoder.int8.onnx")
WHISPER_BASE_TOKENS = os.path.join(WHISPER_BASE_DIR, "base-tokens.txt")

recognizer = None


def get_recognizer():
    global recognizer
    if recognizer is not None:
        return recognizer

    try:
        import sherpa_onnx
        if os.path.exists(WHISPER_BASE_ENCODER) and os.path.exists(WHISPER_BASE_DECODER) and os.path.exists(WHISPER_BASE_TOKENS):
            num_threads = min(4, os.cpu_count() or 4)
            recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=WHISPER_BASE_ENCODER,
                decoder=WHISPER_BASE_DECODER,
                tokens=WHISPER_BASE_TOKENS,
                num_threads=num_threads,
                language="es",
                task="transcribe",
                debug=False
            )
            logger.info("Reconocedor Whisper Base ONNX inicializado con éxito.")
            return recognizer
        else:
            logger.warning(f"Modelos Whisper Base no encontrados en {WHISPER_BASE_DIR}.")
            return None
    except Exception as e:
        logger.error(f"Error inicializando reconocedor STT: {e}")
        return None


class TranscriptionResponse(BaseModel):
    text: str


@router.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default="whisper-base"),
    language: Optional[str] = Form(default="es"),
    temperature: Optional[float] = Form(default=0.0)
):
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Archivo de audio vacío.")

        rec = get_recognizer()
        if rec is None:
            # Fallback en caso de que el modelo aún no esté listo
            return TranscriptionResponse(text="[STT Service Ready]")

        audio_file = io.BytesIO(audio_bytes)
        data, sample_rate = sf.read(audio_file, dtype="float32")

        # Convertir a mono si es estéreo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Resamplear a 16kHz si es necesario
        if sample_rate != 16000:
            import scipy.signal
            target_length = int(len(data) * 16000 / sample_rate)
            data = scipy.signal.resample(data, target_length).astype(np.float32)
            sample_rate = 16000

        stream = rec.create_stream()
        stream.accept_waveform(sample_rate, data)
        rec.decode_stream(stream)
        text = stream.result.text.strip()

        return TranscriptionResponse(text=text)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en transcripción: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
