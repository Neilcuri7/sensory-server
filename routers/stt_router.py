"""
STT Router: Transcripción de Audio OpenAI-compatible (Sherpa-ONNX / Whisper).
Ejecución asíncrona no bloqueante (threading) para alta concurrencia.
"""
import os
import io
import re
import math
import time
import logging
import threading
import asyncio
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("Sensory.STT")
router = APIRouter(tags=["STT"])

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
WHISPER_BASE_DIR = os.path.join(MODELS_DIR, "sherpa-onnx-whisper-base")
WHISPER_BASE_ENCODER = os.path.join(WHISPER_BASE_DIR, "base-encoder.int8.onnx")
WHISPER_BASE_DECODER = os.path.join(WHISPER_BASE_DIR, "base-decoder.int8.onnx")
WHISPER_BASE_TOKENS = os.path.join(WHISPER_BASE_DIR, "base-tokens.txt")

WHISPER_SMALL_DIR = os.path.join(MODELS_DIR, "sherpa-onnx-whisper-small")
WHISPER_SMALL_ENCODER = os.path.join(WHISPER_SMALL_DIR, "small-encoder.int8.onnx")
WHISPER_SMALL_DECODER = os.path.join(WHISPER_SMALL_DIR, "small-decoder.int8.onnx")
WHISPER_SMALL_TOKENS = os.path.join(WHISPER_SMALL_DIR, "small-tokens.txt")

recognizer = None
_stt_lock = threading.Lock()

_RYLAI_MISSPELLINGS_REGEX = re.compile(
    r"\b(rilai|rilay|railai|raily|rylay|relai)\b",
    re.IGNORECASE,
)


def _apply_domain_lexicon_corrections(text: str) -> str:
    """Corrige transcripciones fonéticas hispanas frecuentes del nombre 'Rylai'."""
    if not text:
        return ""
    return _RYLAI_MISSPELLINGS_REGEX.sub("Rylai", text)


def _trim_stt_silence(
    data: np.ndarray,
    sample_rate: int = 16000,
    threshold: float = 0.015,
    pad_ms: int = 80,
) -> np.ndarray:
    """
    Recorta silencios muertos al inicio y final antes de Whisper para acelerar inferencia.
    Conserva pad_ms de margen (default 80ms = 1280 muestras a 16kHz).
    Si todo el array es silencio o la señal activa recortada es menor a 0.1s (1600 muestras), conserva data intacto.
    """
    if len(data) == 0:
        return data
    active_indices = np.where(np.abs(data) > threshold)[0]
    if len(active_indices) == 0:
        return data
    pad_samples = max(0, int((pad_ms / 1000.0) * sample_rate))
    start_idx = max(0, int(active_indices[0]) - pad_samples)
    end_idx = min(len(data), int(active_indices[-1]) + 1 + pad_samples)
    trimmed = data[start_idx:end_idx]
    if len(trimmed) < 1600:
        return data
    return trimmed


def _preprocess_stt_audio(data: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """
    Acondicionamiento DSP de señal previa a Whisper:
    1. Conversión estéreo -> mono.
    2. Remuestreo lineal ultrarrápido con NumPy (np.interp) a 16000 Hz (sin scipy).
    3. Eliminación de componente continua (DC offset).
    4. Normalización automática de ganancia (Peak Auto-Gain) para micrófonos con volumen bajo.
    5. Recorte de silencios muertos al inicio y final (_trim_stt_silence).
    """
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    data = np.asarray(data, dtype=np.float32)

    if sample_rate != 16000 and sample_rate > 0 and len(data) > 0:
        target_len = max(1, int(round(len(data) * 16000.0 / float(sample_rate))))
        orig_idx = np.linspace(0.0, len(data) - 1, target_len, dtype=np.float32)
        data = np.interp(orig_idx, np.arange(len(data), dtype=np.float32), data).astype(np.float32)
        sample_rate = 16000

    if len(data) > 0:
        # Eliminar DC offset
        data = data - float(np.mean(data))
        # Auto-gain peak normalization (solo si hay señal activa y está por debajo del nivel óptimo)
        peak = float(np.max(np.abs(data)))
        if 0.015 <= peak < 0.85:
            data = (data * (0.89 / peak)).astype(np.float32)
        # Recorte de silencio muerto antes de Whisper para acelerar inferencia
        data = _trim_stt_silence(data, sample_rate=sample_rate)

    return data, sample_rate


def _resolve_whisper_model_paths() -> tuple[str, str, str, str]:
    """
    Selecciona los pesos de Whisper (small si se solicita explícitamente, o base por defecto para máxima velocidad).
    Retorna (encoder_path, decoder_path, tokens_path, model_label).
    """
    pref = os.environ.get("SENSORY_STT_MODEL", "base").lower().strip()
    small_exists = (
        os.path.exists(WHISPER_SMALL_ENCODER)
        and os.path.exists(WHISPER_SMALL_DECODER)
        and os.path.exists(WHISPER_SMALL_TOKENS)
    )
    if pref in ("small", "whisper-small") and small_exists:
        return WHISPER_SMALL_ENCODER, WHISPER_SMALL_DECODER, WHISPER_SMALL_TOKENS, "Whisper Small"
    return WHISPER_BASE_ENCODER, WHISPER_BASE_DECODER, WHISPER_BASE_TOKENS, "Whisper Base"


def get_recognizer():
    global recognizer
    if recognizer is not None:
        return recognizer

    try:
        import sherpa_onnx
        encoder_path, decoder_path, tokens_path, model_label = _resolve_whisper_model_paths()
        if os.path.exists(encoder_path) and os.path.exists(decoder_path) and os.path.exists(tokens_path):
            default_threads = min(4, os.cpu_count() or 4)
            try:
                num_threads = int(os.environ.get("SENSORY_CPU_THREADS", str(default_threads)))
            except ValueError:
                num_threads = default_threads

            # Detección elástica de acelerador GPU (CUDA)
            try:
                import onnxruntime as ort
                available = ort.get_available_providers()
                if "CUDAExecutionProvider" in available:
                    try:
                        recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                            encoder=encoder_path,
                            decoder=decoder_path,
                            tokens=tokens_path,
                            num_threads=num_threads,
                            language="es",
                            task="transcribe",
                            debug=False,
                            provider="cuda"
                        )
                        logger.info(f"Reconocedor {model_label} ONNX inicializado con éxito [provider=cuda, threads={num_threads}].")
                        return recognizer
                    except Exception as e_cuda:
                        logger.info(f"GPU no disponible para STT ({e_cuda}), degradando limpiamente a CPU.")
            except Exception as e_ort:
                logger.debug(f"Aviso en detección de provider STT: {e_ort}")

            recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=encoder_path,
                decoder=decoder_path,
                tokens=tokens_path,
                num_threads=num_threads,
                language="es",
                task="transcribe",
                debug=False,
                provider="cpu"
            )
            logger.info(f"Reconocedor {model_label} ONNX inicializado con éxito [provider=cpu, threads={num_threads}].")
            return recognizer
        else:
            logger.warning(f"Modelos Whisper no encontrados en {WHISPER_BASE_DIR}.")
            return None
    except Exception as e:
        logger.error(f"Error inicializando reconocedor STT: {e}")
        return None


class TranscriptionResponse(BaseModel):
    text: str


def _transcribe_bytes_sync(audio_bytes: bytes) -> tuple[str, dict]:
    """Función de transcripción síncrona thread-safe protegida por _stt_lock."""
    t_wait_start = time.perf_counter()
    with _stt_lock:
        lock_wait_ms = (time.perf_counter() - t_wait_start) * 1000.0
        rec = get_recognizer()
        if rec is None:
            raise RuntimeError("STT model not loaded or unavailable")

        t_infer_start = time.perf_counter()
        audio_file = io.BytesIO(audio_bytes)
        data, sample_rate = sf.read(audio_file, dtype="float32")

        raw_duration_sec = (
            (data.shape[0] / float(sample_rate)) if (sample_rate > 0 and len(data) > 0) else 0.0
        )

        data, sample_rate = _preprocess_stt_audio(data, sample_rate)

        stream = rec.create_stream()
        stream.accept_waveform(sample_rate, data)
        rec.decode_stream(stream)
        text = _apply_domain_lexicon_corrections(stream.result.text.strip())

        infer_duration_sec = time.perf_counter() - t_infer_start
        rtf = (infer_duration_sec / raw_duration_sec) if raw_duration_sec > 0 else 0.0

        metrics = {
            "bytes": len(audio_bytes),
            "sample_rate": sample_rate,
            "duration_sec": raw_duration_sec,
            "lock_wait_ms": lock_wait_ms,
            "infer_ms": infer_duration_sec * 1000.0,
            "rtf": rtf,
            "text_len": len(text)
        }
        return text, metrics


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

        try:
            text, metrics = await asyncio.to_thread(_transcribe_bytes_sync, audio_bytes)
        except RuntimeError as r_err:
            logger.error(f"[STT Unavailable] {r_err}")
            raise HTTPException(status_code=503, detail=str(r_err))

        logger.info(
            f"[STT Ingest] {metrics['bytes']}B, {metrics['sample_rate']}Hz, "
            f"{metrics['duration_sec']:.2f}s audio | lock_wait={metrics['lock_wait_ms']:.1f}ms, "
            f"infer={metrics['infer_ms']:.1f}ms (RTF={metrics['rtf']:.2f}) | "
            f"text_len={metrics['text_len']}: '{text[:60]}'"
        )
        return TranscriptionResponse(text=text)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en transcripción: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class STTJsonRequest(BaseModel):
    audio_base64: str
    sample_rate: Optional[int] = 16000
    language: Optional[str] = "es"


@router.post("/stt")
async def transcribe_json(req: STTJsonRequest):
    try:
        import base64
        raw_b64 = req.audio_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        audio_bytes = base64.b64decode(raw_b64)
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Audio base64 vacío.")

        text, metrics = await asyncio.to_thread(_transcribe_bytes_sync, audio_bytes)
        return {
            "status": "ok",
            "text": text,
            "confidence": 1.0,
            "is_speech": len(text) > 0,
            "sample_rate": metrics.get("sample_rate", req.sample_rate),
            "duration_sec": metrics.get("duration_sec", 0.0)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en endpoint /stt: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
