"""
TTS Router: Síntesis de Voz 100% Local (Kokoro ONNX / Piper ONNX).
Exclusivamente voces femeninas con ejecución asíncrona no bloqueante (threading).
"""
import os
import io
import re
import time
import wave
import logging
import threading
import asyncio
from typing import Optional, Dict

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

# Silenciar desalineación de palabras en phonemizer (causada por signos ¿¡ en espeak-ng)
logging.getLogger("phonemizer").setLevel(logging.ERROR)

logger = logging.getLogger("Aoi.Sensory.TTS")
router = APIRouter(tags=["TTS"])

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
PIPER_DIR = os.path.join(MODELS_DIR, "piper")
PIPER_ES_MODEL = os.path.join(PIPER_DIR, "es_ES-sharvard-medium.onnx")
PIPER_ES_CONFIG = os.path.join(PIPER_DIR, "es_ES-sharvard-medium.onnx.json")

KOKORO_DIR = os.path.join(MODELS_DIR, "kokoro")

piper_voice = None
kokoro_voice = None
_tts_lock = threading.Lock()

# Exclusivamente voces femeninas locales soportadas
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

_SPANISH_WORD_REGEX = re.compile(
    r'\b(el|la|los|las|de|que|en|un|una|con|para|por|como|pero|más|hola|estoy|aquí|buenas|tardes|días|noches|claro|puedo|todo|bien|gracias|qué|cómo|está|estás|soy|tienes|quieres)\b',
    re.IGNORECASE
)
_JAPANESE_REGEX = re.compile(r'[\u3040-\u30ff\u4e00-\u9faf]')
_SPANISH_CHAR_REGEX = re.compile(r'[áéíóúüñÁÉÍÓÚÜÑ]')


def _sanitize_text_for_tts(text: str) -> str:
    """Limpia puntuación que desincroniza el conteo de palabras en espeak-ng/phonemizer."""
    if not text:
        return ""
    t = text.replace("¿", "").replace("¡", "")
    t = t.replace("—", ", ").replace("–", ", ").replace("…", ". ").replace("...", ". ")
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def _detect_phoneme_lang(text: str, voice_id: str) -> str:
    """Detecta el idioma fonético adecuado para Kokoro ONNX evitando fallos de phonemizer."""
    if _JAPANESE_REGEX.search(text):
        return "ja"
    if voice_id == "ef_dora" or _SPANISH_CHAR_REGEX.search(text) or _SPANISH_WORD_REGEX.search(text):
        return "es"
    if voice_id in ("af_heart", "af_bella", "af_sarah", "af_nicole", "af_sky"):
        return "en-us"
    if voice_id in ("bf_emma", "bf_isabella"):
        return "en-gb"
    if voice_id.startswith("jf_"):
        return "ja"
    return "es"


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


def split_sentences_for_streaming(text: str) -> list[str]:
    """Divide un texto en oraciones/cláusulas para síntesis incremental."""
    if not text:
        return []
    raw_chunks = re.split(r'([;.\?!]+|\n+)', text)
    sentences = []
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


def _read_wav_samples(wav_bytes: bytes) -> tuple:
    """Lee bytes de audio WAV y retorna (samples_array_float32, sample_rate)."""
    with io.BytesIO(wav_bytes) as bio:
        with wave.open(bio, "rb") as wf:
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            int_data = np.frombuffer(frames, dtype=np.int16)
            float_data = int_data.astype(np.float32) / 32767.0
            return float_data, sample_rate


def _write_wav_bytes(samples: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convierte un array de samples numpy (float32 o int16) en un WAV PCM de 16 bits."""
    if samples.dtype in (np.float32, np.float64):
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


def _generate_wav_bytes(audio_data: bytes, sample_rate: int = 22050) -> bytes:
    """Genera archivo WAV estándar PCM 16-bit a partir de datos raw int16."""
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)
    return out.getvalue()


def _synthesize_wav_sync(text: str, voice_id: str, speed: float) -> tuple[bytes, dict]:
    """Función de síntesis síncrona thread-safe protegida por _tts_lock."""
    t_wait_start = time.perf_counter()
    with _tts_lock:
        lock_wait_ms = (time.perf_counter() - t_wait_start) * 1000.0
        clean_text = _sanitize_text_for_tts(text)
        chars = len(clean_text)
        if not clean_text:
            dummy = _generate_wav_bytes(b"\x00" * 4410, sample_rate=22050)
            metrics = {
                "voice": voice_id,
                "engine": "empty",
                "lang": "none",
                "chars": 0,
                "lock_wait_ms": lock_wait_ms,
                "infer_ms": 0.0,
                "audio_dur_ms": 100.0,
                "bytes": len(dummy)
            }
            return dummy, metrics

        user_speed = max(0.5, min(2.0, float(speed)))
        is_piper = voice_id in ("piper_f", "piper_sharvard_f", "es_ES-sharvard-medium", "1", "sharvard")

        target_voice = voice_id if voice_id in KOKORO_FEMALE_VOICES else "ef_dora"
        detected_lang = _detect_phoneme_lang(clean_text, target_voice) if not is_piper else "es"
        engine_used = "fallback"

        t_infer_start = time.perf_counter()
        wav_bytes = None

        # 1. Kokoro ONNX (si no es explícitamente Piper)
        if not is_piper:
            kokoro = get_kokoro_voice()
            if kokoro is not None:
                try:
                    samples, sample_rate = kokoro.create(
                        clean_text,
                        voice=target_voice,
                        speed=user_speed,
                        lang=detected_lang
                    )
                    wav_bytes = _write_wav_bytes(samples, sample_rate=sample_rate)
                    engine_used = "kokoro"
                except Exception as e:
                    logger.warning(f"Kokoro ONNX falló durante síntesis, probando Piper como fallback: {e}")

        # 2. Piper TTS (100% Local - Femenino speaker_id=1 con control real de speed)
        if wav_bytes is None:
            voice = get_piper_voice()
            if voice is not None:
                try:
                    from piper.config import SynthesisConfig
                    length_scale = 1.0 / user_speed
                    try:
                        syn_cfg = SynthesisConfig(speaker_id=1, length_scale=length_scale)
                    except (TypeError, ValueError):
                        syn_cfg = SynthesisConfig(speaker_id=1)

                    sample_rate = getattr(getattr(voice, "config", None), "sample_rate", 22050)
                    audio_frames = bytearray()

                    for chunk in voice.synthesize(clean_text, syn_config=syn_cfg):
                        if hasattr(chunk, "audio_int16_bytes"):
                            audio_frames.extend(chunk.audio_int16_bytes)
                        elif isinstance(chunk, bytes):
                            audio_frames.extend(chunk)

                    if len(audio_frames) > 0:
                        wav_bytes = _generate_wav_bytes(bytes(audio_frames), sample_rate=sample_rate)
                        engine_used = "piper"
                    else:
                        logger.warning("Piper sintetizó 0 bytes de audio, generando silencio.")
                except Exception as e:
                    logger.error(f"Error sintetizando con Piper: {e}")

        # 3. Fallback dummy silence si ningún motor está listo
        if wav_bytes is None:
            wav_bytes = _generate_wav_bytes(b"\x00" * 4410, sample_rate=22050)
            engine_used = "fallback"

        infer_ms = (time.perf_counter() - t_infer_start) * 1000.0

        audio_dur_ms = 0.0
        if len(wav_bytes) > 44:
            sr = 24000 if engine_used == "kokoro" else 22050
            audio_dur_ms = ((len(wav_bytes) - 44) / (sr * 2)) * 1000.0

        metrics = {
            "voice": voice_id,
            "engine": engine_used,
            "lang": detected_lang,
            "chars": chars,
            "lock_wait_ms": lock_wait_ms,
            "infer_ms": infer_ms,
            "audio_dur_ms": audio_dur_ms,
            "bytes": len(wav_bytes)
        }
        return wav_bytes, metrics


@router.post("/v1/audio/speech")
async def generate_speech(req: SpeechRequest):
    text = req.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Texto de entrada vacío.")

    voice_id = (req.voice or "ef_dora").strip()
    if (req.model or "").lower() == "piper" and voice_id not in ("piper_f", "piper_sharvard_f"):
        voice_id = "piper_f"

    wav_bytes, metrics = await asyncio.to_thread(
        _synthesize_wav_sync,
        text,
        voice_id,
        float(req.speed or 1.0)
    )
    logger.info(
        f"🔊 [TTS Synthesis] voice='{metrics['voice']}' ({metrics['engine']}), lang='{metrics['lang']}', "
        f"chars={metrics['chars']} | lock_wait={metrics['lock_wait_ms']:.1f}ms, "
        f"infer={metrics['infer_ms']:.1f}ms | audio_dur={metrics['audio_dur_ms']:.0f}ms ({metrics['bytes']}B)"
    )
    return Response(content=wav_bytes, media_type="audio/wav")


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
