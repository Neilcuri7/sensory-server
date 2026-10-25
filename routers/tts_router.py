"""
TTS Router: Síntesis de Voz 100% Local (Piper ONNX / Kokoro). Exclusivamente voces femeninas.
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

KOKORO_DIR = os.path.join(MODELS_DIR, "kokoro")

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


def get_kokoro_voice():
    global kokoro_voice
    if kokoro_voice is not None:
        return kokoro_voice
    try:
        from kokoro_onnx import Kokoro
        model_path = os.path.join(KOKORO_DIR, "kokoro-v1.0.int8.onnx")
        if not os.path.exists(model_path):
            model_path = os.path.join(KOKORO_DIR, "kokoro-v1.0.onnx")
        voices_path = os.path.join(KOKORO_DIR, "voices-v1.0.bin")
        if os.path.exists(model_path) and os.path.exists(voices_path):
            kokoro_voice = Kokoro(model_path, voices_path)
            logger.info("Motor Kokoro ONNX inicializado con éxito.")
            return kokoro_voice
    except Exception as e:
        logger.warning(f"Kokoro ONNX no disponible: {e}")
    return None


# Mapeo de voces Kokoro exclusivamente femeninas y su idioma respectivo
KOKORO_FEMALE_VOICES: Dict[str, str] = {
    "ef_dora": "es",
    "af_heart": "en-us",
    "af_bella": "en-us",
    "af_sarah": "en-us",
    "af_nicole": "en-us",
    "af_sky": "en-us",
    "bf_emma": "en-gb",
    "bf_isabella": "en-gb",
    "jf_alpha": "ja",
    "jf_gongitsune": "ja",
    "jf_nezumi": "ja",
    "jf_tebukuro": "ja"
}


class SpeechRequest(BaseModel):
    model: Optional[str] = Field(default="kokoro", description="Motor o modelo de voz ('kokoro', 'piper').")
    input: str = Field(..., description="Texto a sintetizar.")
    voice: Optional[str] = Field(default="ef_dora", description="Voz seleccionada (100% local femenina).")
    response_format: Optional[str] = Field(default="wav", description="Formato de salida ('wav').")
    speed: Optional[float] = Field(default=1.0, description="Velocidad de reproducción (0.25 a 4.0).")


class HealthResponse(BaseModel):
    status: str = "ok"
    default_engine: str = "kokoro"
    active_engine: str = "kokoro"
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

    preferred_engine = (req.model or "kokoro").lower()
    voice_name = (req.voice or "ef_dora").strip()
    user_speed = max(0.5, min(2.0, float(req.speed or 1.0)))

    # Determinar si la solicitud apunta explícitamente a Piper
    is_piper_request = (
        preferred_engine == "piper"
        or voice_name in ("piper_f", "piper_sharvard_f", "es_ES-sharvard-medium", "1", "sharvard")
    )

    # 1. Síntesis con Kokoro ONNX (si no se pide explícitamente Piper)
    if not is_piper_request:
        kokoro = get_kokoro_voice()
        if kokoro is not None:
            try:
                # Normalización a voces femeninas permitidas de Kokoro
                target_kokoro_voice = voice_name if voice_name in KOKORO_FEMALE_VOICES else "ef_dora"
                lang = KOKORO_FEMALE_VOICES.get(target_kokoro_voice)
                if not lang:
                    if target_kokoro_voice.startswith("ef_"):
                        lang = "es"
                    elif target_kokoro_voice.startswith("af_"):
                        lang = "en-us"
                    elif target_kokoro_voice.startswith("bf_"):
                        lang = "en-gb"
                    elif target_kokoro_voice.startswith("jf_"):
                        lang = "ja"
                    else:
                        lang = "es"

                samples, sample_rate = kokoro.create(
                    text,
                    voice=target_kokoro_voice,
                    speed=user_speed,
                    lang=lang
                )
                wav_bytes = _write_wav_bytes(samples, sample_rate=sample_rate)
                return Response(content=wav_bytes, media_type="audio/wav")
            except Exception as e:
                logger.warning(f"Kokoro ONNX falló durante síntesis, probando Piper como fallback: {e}")

    # 2. Síntesis 100% Local con Piper TTS (Voz femenina: speaker_id=1 'F' con ajuste real de speed)
    voice = get_piper_voice()
    if voice is not None:
        try:
            from piper.config import SynthesisConfig

            # En Piper ONNX, length_scale es el inverso de la velocidad (mayor length_scale = audio más lento)
            length_scale = 1.0 / user_speed
            try:
                syn_cfg = SynthesisConfig(speaker_id=1, length_scale=length_scale)
            except (TypeError, ValueError):
                syn_cfg = SynthesisConfig(speaker_id=1)

            sample_rate = getattr(getattr(voice, "config", None), "sample_rate", 22050)
            audio_frames = bytearray()

            for chunk in voice.synthesize(text, syn_config=syn_cfg):
                if hasattr(chunk, "audio_int16_bytes"):
                    audio_frames.extend(chunk.audio_int16_bytes)
                elif isinstance(chunk, bytes):
                    audio_frames.extend(chunk)

            if len(audio_frames) > 0:
                wav_bytes = _generate_wav_bytes(bytes(audio_frames), sample_rate=sample_rate)
                return Response(content=wav_bytes, media_type="audio/wav")
            else:
                logger.warning("Piper sintetizó 0 bytes de audio, generando respaldo silencioso.")
        except Exception as e:
            logger.error(f"Error sintetizando con Piper: {e}")

    # 3. Fallback dummy silence 100% local si ningún motor está listo
    silent_wav = _generate_wav_bytes(b"\x00" * 4410, sample_rate=22050)
    return Response(content=silent_wav, media_type="audio/wav")


@router.get("/v1/audio/voices")
async def list_voices():
    return {
        "voices": [
            {"id": "ef_dora", "name": "🌸 Dora (Kokoro Local - Español Femenino)", "language": "es", "gender": "female"},
            {"id": "piper_f", "name": "⚡ Sharvard (Piper Local - Español Femenino)", "language": "es", "gender": "female"},
            {"id": "af_heart", "name": "🇺🇸 Heart (Kokoro Local - Inglés Femenino)", "language": "en", "gender": "female"},
            {"id": "af_bella", "name": "🇺🇸 Bella (Kokoro Local - Inglés Femenino)", "language": "en", "gender": "female"},
            {"id": "jf_alpha", "name": "🇯🇵 Alpha (Kokoro Local - Japonés Femenino)", "language": "ja", "gender": "female"}
        ]
    }
