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
import onnxruntime as ort
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

# Silenciar desalineación de palabras en phonemizer (causada por signos ¿¡ en espeak-ng)
logging.getLogger("phonemizer").setLevel(logging.ERROR)

logger = logging.getLogger("Sensory.TTS")
router = APIRouter(tags=["TTS"])

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
PIPER_DIR = os.path.join(MODELS_DIR, "piper")
PIPER_ES_MODEL = os.path.join(PIPER_DIR, "es_ES-sharvard-medium.onnx")
PIPER_ES_CONFIG = os.path.join(PIPER_DIR, "es_ES-sharvard-medium.onnx.json")

KOKORO_DIR = os.path.join(MODELS_DIR, "kokoro")

piper_voice = None
kokoro_voice = None
_kokoro_model = None
_model_init_lock = threading.Lock()
_espeak_piper_lock = threading.Lock()

SENSORY_TIER = os.environ.get("SENSORY_TIER", "full").lower()
preferred_engine = "piper" if SENSORY_TIER == "light" else "kokoro"


def get_sensory_tier() -> str:
    """Retorna el tier sensorial activo ('light', 'full')."""
    return os.environ.get("SENSORY_TIER", SENSORY_TIER).lower()


def get_preferred_engine() -> str:
    """Retorna el motor preferido según el tier ('piper' en light, 'kokoro' en full)."""
    return "piper" if get_sensory_tier() == "light" else "kokoro"

# Exclusivamente voces femeninas locales soportadas
KOKORO_FEMALE_VOICES: Dict[str, str] = {
    "ef_blend": "es",
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
_JAPANESE_REGEX = re.compile(r'[\u3040-\u30ff\u4e00-\u9fff]')
_SPANISH_CHAR_REGEX = re.compile(r'[áéíóúüñÁÉÍÓÚÜÑ¿¡]')


def _strip_emojis_and_symbols(text: str) -> str:
    """Elimina emojis, dingbats y símbolos decorativos preservando caracteres y puntuación en español."""
    import unicodedata
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        cp = ord(ch)
        # Excluir emojis y pictografías suplementarias, dingbats, símbolos decorativos y selectores de variación
        if (
            cat in ("So", "Sk")
            or (0x1F300 <= cp <= 0x1FAFF)
            or (0x2600 <= cp <= 0x27BF)
            or cp in (0x200D, 0xFE0E, 0xFE0F)
        ):
            cleaned.append(" ")
        else:
            cleaned.append(ch)
    return "".join(cleaned)


def _sanitize_text_for_tts(text: str) -> str:
    """Limpia markdown, URLs, acciones roleplay y emojis preservando ¿ y ¡ para entonación nativa en español."""
    if not text:
        return ""
    # 1. Bloques de código formateados (fenced code blocks)
    t = re.sub(r"```[\s\S]*?```", " ", text)
    # 2. URLs
    t = re.sub(r"https?://\S+", " ", t)
    # 3. Acciones de roleplay entre asteriscos simples (*sonríe*)
    t = re.sub(r"\*[^*\n]+\*", " ", t)
    # 4. Símbolos markdown residuales (*, _, #, `, ~)
    t = re.sub(r"[*_#`~]", " ", t)
    # 5. Emojis y pictogramas
    t = _strip_emojis_and_symbols(t)
    # 6. Puntuación: NO eliminar ¿ ni ¡ (espeak-ng los usa para entonación exclamativa e interrogativa)
    t = t.replace("—", ", ").replace("–", ", ").replace("…", ". ").replace("...", ". ")
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def _detect_phoneme_lang(text: str, voice_id: str, requested_lang: Optional[str] = None) -> str:
    """Detecta el idioma fonético adecuado para Kokoro ONNX evitando fallos de phonemizer."""
    if requested_lang and requested_lang.lower() not in ("auto", ""):
        req = requested_lang.lower()
        if req == "ja":
            return "ja"
        if req in ("es", "en-us", "en-gb", "en"):
            return "en-us" if req == "en" else req

    # Solo retornar "ja" si el texto efectivamente contiene caracteres japoneses
    if _JAPANESE_REGEX.search(text):
        return "ja"

    if voice_id in ("af_heart", "af_bella", "af_sarah", "af_nicole", "af_sky"):
        return "en-us"
    if voice_id in ("bf_emma", "bf_isabella"):
        return "en-gb"

    # En caso de texto en alfabeto latino/español o voces jf_* usadas en español, siempre "es"
    return "es"


def _resolve_kokoro_voice(kokoro_instance, voice_id: str, lang: str):
    """
    Resuelve la voz para Kokoro ONNX:
    - Si lang == 'es' y voice_id en ('ef_blend', 'jf_alpha') o jf_*:
      mezcla linealmente el estilo neuronal español de 'ef_dora' con el timbre anime de 'jf_alpha' (o voice_id).
    - Mantiene compatibilidad con mocks de test unitarios.
    """
    if not hasattr(kokoro_instance, "get_voice_style") or not hasattr(kokoro_instance, "get_voices"):
        return voice_id

    try:
        available = kokoro_instance.get_voices()
    except Exception:
        return voice_id

    if not isinstance(available, (list, tuple, set)):
        return voice_id

    if lang == "es":
        target_anime = "jf_alpha" if voice_id in ("ef_blend", "jf_alpha") else voice_id
        if "ef_dora" in available and target_anime in available:
            try:
                style_dora = kokoro_instance.get_voice_style("ef_dora")
                style_anime = kokoro_instance.get_voice_style(target_anime)
                weight_dora = 0.70 if voice_id in ("ef_blend", "jf_alpha") else 0.65
                weight_anime = 1.0 - weight_dora
                blended = (weight_dora * style_dora + weight_anime * style_anime).astype(np.float32)
                return blended
            except Exception as e:
                logger.warning(f"Error mezclando estilos Kokoro para {voice_id}: {e}")

    if voice_id in available:
        return voice_id
    if "ef_dora" in available and lang == "es":
        return "ef_dora"
    if "jf_alpha" in available:
        return "jf_alpha"
    return voice_id


def _apply_anime_pitch_shift(
    samples: np.ndarray,
    sample_rate: int,
    pitch_semitones: float,
) -> np.ndarray:
    """
    Aplica cambio de tono (pitch shift) en semitonos conservando exactamente la duración y tempo (OLA con ventana Hann).
    Garantiza len(output) == len(samples) y ejecución ultrarrápida (<10ms).
    """
    if abs(pitch_semitones) < 0.05 or len(samples) == 0:
        return samples

    factor = 2.0 ** (pitch_semitones / 12.0)
    orig_n = len(samples)
    win_size = min(512, orig_n)
    if win_size < 32:
        return samples

    hop = win_size // 4
    rate = 1.0 / factor
    s_hop = hop
    a_hop = max(1, int(s_hop * rate))

    window = np.hanning(win_size).astype(np.float32)
    num_frames = (orig_n - win_size) // a_hop
    if num_frames <= 0:
        return samples

    out_len = win_size + num_frames * s_hop
    out = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)

    for i in range(num_frames):
        a_pos = i * a_hop
        s_pos = i * s_hop
        out[s_pos:s_pos + win_size] += samples[a_pos:a_pos + win_size] * window
        norm[s_pos:s_pos + win_size] += window

    mask = norm > 1e-4
    out[mask] /= norm[mask]

    target_stretched = max(1, int(orig_n * factor))
    stretched = out[:target_stretched] if len(out) >= target_stretched else np.pad(out, (0, target_stretched - len(out)))

    # Remuestrear stretched exactamente a orig_n para restaurar tempo y longitud original
    orig_indices = np.linspace(0, len(stretched) - 1, orig_n, dtype=np.float32)
    shifted = np.interp(orig_indices, np.arange(len(stretched), dtype=np.float32), stretched)
    return shifted.astype(samples.dtype)


def _get_elastic_session_options() -> ort.SessionOptions:
    """Configura SessionOptions elásticas con intra_op_num_threads óptimo para CPU i7 P-Cores."""
    sess_options = ort.SessionOptions()
    cpu_threads = min(4, os.cpu_count() or 4)
    sess_options.intra_op_num_threads = cpu_threads
    sess_options.inter_op_num_threads = 1
    # Silenciar advertencias internas de ONNX Runtime para degradación limpia a CPU
    sess_options.log_severity_level = 3
    return sess_options


def _create_elastic_inference_session(model_path: str) -> tuple[ort.InferenceSession, str]:
    """
    Crea una sesión ONNX Runtime elástica:
    1. Detecta si CUDAExecutionProvider o DmlExecutionProvider (DirectML en Windows)
       están disponibles en onnxruntime.get_available_providers().
    2. Si están disponibles, configura límites de VRAM (6GB) y arena_extend_strategy.
    3. Si no están disponibles o falla la inicialización en GPU, degrada limpiamente a CPUExecutionProvider.
    4. En CPU configura SessionOptions con intra_op_num_threads = min(4, os.cpu_count() or 4).
    """
    available = ort.get_available_providers()
    sess_options = _get_elastic_session_options()
    cpu_threads = sess_options.intra_op_num_threads
    model_name = os.path.basename(model_path)

    session = None
    active_provider = "CPUExecutionProvider"

    if "CUDAExecutionProvider" in available:
        cuda_options = {
            "arena_extend_strategy": "kSameAsRequested",
            "gpu_mem_limit": 6 * 1024 * 1024 * 1024,
        }
        candidate_providers = [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
        try:
            session = ort.InferenceSession(model_path, sess_options=sess_options, providers=candidate_providers)
            actual = session.get_providers()
            if actual and actual[0] == "CUDAExecutionProvider":
                active_provider = actual[0]
                logger.info(f"ONNX Runtime '{model_name}' acelerado por GPU: {active_provider}")
            else:
                active_provider = "CPUExecutionProvider"
                logger.info(f"ONNX Runtime '{model_name}' degrado a CPUExecutionProvider (threads={cpu_threads}).")
        except Exception as e:
            logger.info(f"GPU no disponible o fallo en inicializacion para '{model_name}' ({e}). Degradando limpiamente a CPUExecutionProvider.")
            session = None
    elif "DmlExecutionProvider" in available:
        candidate_providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        try:
            session = ort.InferenceSession(model_path, sess_options=sess_options, providers=candidate_providers)
            actual = session.get_providers()
            if actual and actual[0] == "DmlExecutionProvider":
                active_provider = actual[0]
                logger.info(f"ONNX Runtime '{model_name}' acelerado por DirectML: {active_provider}")
            else:
                active_provider = "CPUExecutionProvider"
        except Exception as e:
            logger.info(f"DirectML no disponible para '{model_name}' ({e}). Degradando a CPU.")
            session = None

    if session is None:
        session = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
        logger.info(f"ONNX Runtime '{model_name}' inicializado con CPUExecutionProvider (threads={cpu_threads}).")
        active_provider = "CPUExecutionProvider"

    return session, active_provider


def get_piper_voice():
    global piper_voice
    if piper_voice is not None:
        return piper_voice

    with _model_init_lock:
        if piper_voice is not None:
            return piper_voice
        try:
            from piper import PiperVoice
            if os.path.exists(PIPER_ES_MODEL) and os.path.exists(PIPER_ES_CONFIG):
                session, provider = _create_elastic_inference_session(PIPER_ES_MODEL)
                piper_voice = PiperVoice.load(PIPER_ES_MODEL, config_path=PIPER_ES_CONFIG)
                piper_voice.session = session
                logger.info(f"Motor Piper TTS inicializado con éxito [provider={provider}].")
                return piper_voice
            else:
                logger.warning(f"Archivos de modelo Piper no encontrados en {PIPER_DIR}.")
        except Exception as e:
            logger.warning(f"Piper TTS no disponible: {e}")
    return None


def _load_kokoro():
    """Carga o retorna la instancia de Kokoro ONNX con soporte para degradación elegante en light tier."""
    global kokoro_voice, _kokoro_model
    if kokoro_voice is not None:
        _kokoro_model = kokoro_voice
        return kokoro_voice

    tier = get_sensory_tier()
    with _model_init_lock:
        if kokoro_voice is not None:
            _kokoro_model = kokoro_voice
            return kokoro_voice
        try:
            from kokoro_onnx import Kokoro
            model_path = os.path.join(KOKORO_DIR, "kokoro-v1.0.int8.onnx")
            if not os.path.exists(model_path):
                model_path = os.path.join(KOKORO_DIR, "kokoro-v1.0.onnx")
            voices_path = os.path.join(KOKORO_DIR, "voices-v1.0.bin")
            if os.path.exists(model_path) and os.path.exists(voices_path):
                session, provider = _create_elastic_inference_session(model_path)
                kokoro_voice = Kokoro.from_session(session, voices_path)
                _kokoro_model = kokoro_voice
                logger.info(f"Motor Kokoro ONNX inicializado con éxito [provider={provider}].")
                return kokoro_voice
            else:
                if tier == "light":
                    logger.info("SENSORY_TIER=light activo: modelo Kokoro omitido (Piper listo para TTS).")
                else:
                    logger.warning(f"Archivos de modelo Kokoro no encontrados en {KOKORO_DIR}.")
                kokoro_voice = None
                _kokoro_model = None
                return None
        except Exception as e:
            if tier == "light":
                logger.info(f"SENSORY_TIER=light activo: Kokoro no disponible ({e}). Usando Piper.")
            else:
                logger.warning(f"Kokoro ONNX no disponible: {e}")
            kokoro_voice = None
            _kokoro_model = None
    return None


get_kokoro_voice = _load_kokoro


class SpeechRequest(BaseModel):
    model: Optional[str] = Field(default="kokoro", description="Motor o modelo de voz ('kokoro', 'piper').")
    input: Optional[str] = Field(default=None, description="Texto a sintetizar (OpenAI format).")
    text: Optional[str] = Field(default=None, description="Texto a sintetizar.")
    voice: Optional[str] = Field(default="jf_alpha", description="Voz seleccionada (100% local femenina).")
    language: Optional[str] = Field(default="es", description="Idioma fonético ('es', 'ja', 'en-us').")
    response_format: Optional[str] = Field(default="wav", description="Formato de salida ('wav').")
    speed: Optional[float] = Field(default=1.05, description="Velocidad de reproducción (0.25 a 4.0).")
    pitch_semitones: Optional[float] = Field(default=2.0, description="Modificación de tono en semitonos (ej. +2.0 anime).")
    sequence_id: Optional[int] = Field(default=0, description="ID secuencial de streaming de audio.")


class HealthResponse(BaseModel):
    status: str = "ok"
    tier: str = "full"
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
    """Convierte un array de samples numpy (float32 o int16) en un WAV PCM de 16 bits con soft-limiting de picos."""
    if samples.dtype in (np.float32, np.float64):
        if len(samples) > 0:
            peak = float(np.max(np.abs(samples)))
            if peak > 0.98:
                samples = samples * (0.98 / peak)
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


def _synthesize_wav_sync(
    text: str,
    voice_id: str,
    speed: float,
    language: Optional[str] = "es",
    pitch_semitones: float = 2.0,
    model: Optional[str] = None,
) -> tuple[bytes, dict]:
    """Función de síntesis síncrona thread-safe con concurrencia fina para Kokoro y lock para Piper."""
    tier = get_sensory_tier()
    lock_wait_ms = 0.0
    clean_text = _sanitize_text_for_tts(text)
    chars = len(clean_text)
    if not clean_text:
        dummy = _generate_wav_bytes(b"\x00" * 4410, sample_rate=22050)
        metrics = {
            "voice": voice_id,
            "engine": "empty",
            "lang": "none",
            "chars": 0,
            "lock_wait_ms": 0.0,
            "infer_ms": 0.0,
            "onnx_ms": 0.0,
            "dsp_ms": 0.0,
            "audio_dur_ms": 100.0,
            "bytes": len(dummy)
        }
        return dummy, metrics

    user_speed = max(0.5, min(2.0, float(speed)))
    req_model = (model or "").lower().strip()
    is_piper = (
        tier == "light" or
        preferred_engine == "piper" or
        get_preferred_engine() == "piper" or
        req_model in ("piper", "piper-tts", "piper_f", "sharvard") or
        voice_id in ("piper_f", "piper_sharvard_f", "es_ES-sharvard-medium", "1", "sharvard", "sharvard-medium")
    )

    target_voice = "piper_f" if is_piper else (voice_id if voice_id in KOKORO_FEMALE_VOICES else "jf_alpha")
    detected_lang = "es" if is_piper else _detect_phoneme_lang(clean_text, target_voice, requested_lang=language)
    engine_used = "fallback"

    t_infer_start = time.perf_counter()
    wav_bytes = None
    onnx_ms = 0.0
    dsp_ms = 0.0

    # Si se solicita explícitamente Piper o tier == 'light': sintetizar prioritariamente con Piper (protegido por lock de espeakbridge)
    if is_piper:
        t_lock_start = time.perf_counter()
        with _espeak_piper_lock:
            lock_wait_ms = (time.perf_counter() - t_lock_start) * 1000.0
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

                    t_onnx_start = time.perf_counter()
                    for chunk in voice.synthesize(clean_text, syn_config=syn_cfg):
                        if hasattr(chunk, "audio_int16_bytes"):
                            audio_frames.extend(chunk.audio_int16_bytes)
                        elif isinstance(chunk, bytes):
                            audio_frames.extend(chunk)
                    onnx_ms = (time.perf_counter() - t_onnx_start) * 1000.0

                    if len(audio_frames) > 0:
                        raw_samples = (
                            np.frombuffer(audio_frames, dtype=np.int16).astype(np.float32) / 32767.0
                        )
                        t_dsp_start = time.perf_counter()
                        if abs(pitch_semitones) >= 0.05:
                            raw_samples = _apply_anime_pitch_shift(
                                raw_samples, sample_rate, pitch_semitones
                            )
                        dsp_ms = (time.perf_counter() - t_dsp_start) * 1000.0
                        wav_bytes = _write_wav_bytes(raw_samples, sample_rate=sample_rate)
                        engine_used = "piper"
                    else:
                        logger.warning("Piper sintetizó 0 bytes de audio.")
                except Exception as e:
                    logger.error(f"Error sintetizando con Piper: {e}")

        # Fallback a Kokoro si Piper no estuviese listo (solo en full tier)
        if wav_bytes is None and tier != "light":
            kokoro = get_kokoro_voice()
            if kokoro is not None:
                try:
                    t_onnx_start = time.perf_counter()
                    resolved_voice = _resolve_kokoro_voice(kokoro, "jf_alpha", "es")
                    samples, sample_rate = kokoro.create(
                        clean_text,
                        voice=resolved_voice,
                        speed=user_speed,
                        lang="es"
                    )
                    onnx_ms = (time.perf_counter() - t_onnx_start) * 1000.0
                    t_dsp_start = time.perf_counter()
                    if abs(pitch_semitones) >= 0.05:
                        samples = _apply_anime_pitch_shift(samples, sample_rate, pitch_semitones)
                    dsp_ms = (time.perf_counter() - t_dsp_start) * 1000.0
                    wav_bytes = _write_wav_bytes(samples, sample_rate=sample_rate)
                    engine_used = "kokoro"
                except Exception as e:
                    logger.warning(f"Fallback Kokoro también falló: {e}")
    else:
        # 1. Kokoro ONNX (Estándar de Alta Fidelidad - Ejecución concurrente no bloqueante)
        kokoro = get_kokoro_voice()
        if kokoro is not None:
            try:
                t_onnx_start = time.perf_counter()
                resolved_voice = _resolve_kokoro_voice(kokoro, target_voice, detected_lang)
                samples, sample_rate = kokoro.create(
                    clean_text,
                    voice=resolved_voice,
                    speed=user_speed,
                    lang=detected_lang
                )
                onnx_ms = (time.perf_counter() - t_onnx_start) * 1000.0
                t_dsp_start = time.perf_counter()
                if abs(pitch_semitones) >= 0.05:
                    samples = _apply_anime_pitch_shift(samples, sample_rate, pitch_semitones)
                dsp_ms = (time.perf_counter() - t_dsp_start) * 1000.0
                wav_bytes = _write_wav_bytes(samples, sample_rate=sample_rate)
                engine_used = "kokoro"
            except Exception as e:
                logger.warning(f"Kokoro ONNX falló durante síntesis, probando Piper como fallback: {e}")

        # 2. Piper TTS (Fallback si Kokoro no está listo o falló)
        if wav_bytes is None:
            with _espeak_piper_lock:
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

                        t_onnx_start = time.perf_counter()
                        for chunk in voice.synthesize(clean_text, syn_config=syn_cfg):
                            if hasattr(chunk, "audio_int16_bytes"):
                                audio_frames.extend(chunk.audio_int16_bytes)
                            elif isinstance(chunk, bytes):
                                audio_frames.extend(chunk)
                        onnx_ms = (time.perf_counter() - t_onnx_start) * 1000.0

                        if len(audio_frames) > 0:
                            raw_samples = (
                                np.frombuffer(audio_frames, dtype=np.int16).astype(np.float32) / 32767.0
                            )
                            t_dsp_start = time.perf_counter()
                            if abs(pitch_semitones) >= 0.05:
                                raw_samples = _apply_anime_pitch_shift(
                                    raw_samples, sample_rate, pitch_semitones
                                )
                            dsp_ms = (time.perf_counter() - t_dsp_start) * 1000.0
                            wav_bytes = _write_wav_bytes(raw_samples, sample_rate=sample_rate)
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
        "voice": target_voice,
        "engine": engine_used,
        "lang": detected_lang,
        "chars": chars,
        "lock_wait_ms": lock_wait_ms,
        "infer_ms": infer_ms,
        "onnx_ms": onnx_ms,
        "dsp_ms": dsp_ms,
        "audio_dur_ms": audio_dur_ms,
        "bytes": len(wav_bytes)
    }
    return wav_bytes, metrics


synthesize_tts = _synthesize_wav_sync


@router.post("/v1/audio/speech")
async def generate_speech(req: SpeechRequest):
    text = (req.input or req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Texto de entrada vacío.")

    tier = get_sensory_tier()
    model_req = (req.model or "").lower().strip()
    voice_id = (req.voice or "").strip()

    if tier == "light":
        model_req = "piper"
        if not voice_id or voice_id not in ("piper_f", "piper_sharvard_f", "es_ES-sharvard-medium"):
            voice_id = "piper_f"
    elif model_req in ("piper", "piper-tts", "piper_f", "sharvard"):
        if not voice_id or voice_id in ("jf_alpha", "default"):
            voice_id = "piper_f"
    elif not voice_id:
        voice_id = "jf_alpha"

    wav_bytes, metrics = await asyncio.to_thread(
        _synthesize_wav_sync,
        text,
        voice_id,
        float(req.speed or 1.05),
        req.language or "es",
        float(req.pitch_semitones if req.pitch_semitones is not None else 2.0),
        model=model_req
    )
    logger.info(
        f"[TTS Synthesis] voice='{metrics['voice']}' ({metrics['engine']}), lang='{metrics['lang']}', "
        f"chars={metrics['chars']} | lock_wait={metrics['lock_wait_ms']:.1f}ms, "
        f"infer={metrics['infer_ms']:.1f}ms (onnx={metrics.get('onnx_ms', 0.0):.1f}ms, dsp={metrics.get('dsp_ms', 0.0):.1f}ms) | "
        f"audio_dur={metrics['audio_dur_ms']:.0f}ms ({metrics['bytes']}B)"
    )
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-TTS-Engine": metrics["engine"],
            "X-TTS-Voice": metrics["voice"],
            "X-TTS-Duration-Ms": str(int(metrics["audio_dur_ms"]))
        }
    )


@router.post("/tts")
async def tts_endpoint(req: SpeechRequest):
    text = (req.input or req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Texto de entrada vacío.")

    tier = get_sensory_tier()
    model_req = (req.model or "").lower().strip()
    voice_id = (req.voice or "").strip()

    if tier == "light":
        model_req = "piper"
        if not voice_id or voice_id not in ("piper_f", "piper_sharvard_f", "es_ES-sharvard-medium"):
            voice_id = "piper_f"
    elif model_req in ("piper", "piper-tts", "piper_f", "sharvard"):
        if not voice_id or voice_id in ("jf_alpha", "default"):
            voice_id = "piper_f"
    elif not voice_id:
        voice_id = "jf_alpha"

    speed = float(req.speed or 1.05)
    pitch = float(req.pitch_semitones if req.pitch_semitones is not None else 2.0)
    lang = req.language or "es"

    wav_bytes, metrics = await asyncio.to_thread(
        _synthesize_wav_sync,
        text,
        voice_id,
        speed,
        lang,
        pitch,
        model=model_req
    )
    logger.info(
        f"[TTS Synthesis] voice='{metrics['voice']}' ({metrics['engine']}), lang='{metrics['lang']}', "
        f"chars={metrics['chars']} | lock_wait={metrics['lock_wait_ms']:.1f}ms, "
        f"infer={metrics['infer_ms']:.1f}ms (onnx={metrics.get('onnx_ms', 0.0):.1f}ms, dsp={metrics.get('dsp_ms', 0.0):.1f}ms) | "
        f"audio_dur={metrics['audio_dur_ms']:.0f}ms ({metrics['bytes']}B)"
    )
    import base64
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    sample_rate = 24000 if metrics["engine"] == "kokoro" else 22050

    return {
        "status": "ok",
        "audio_base64": audio_b64,
        "sample_rate": sample_rate,
        "duration_ms": metrics["audio_dur_ms"],
        "voice": metrics["voice"],
        "engine": metrics["engine"],
        "sequence_id": req.sequence_id or 0
    }


@router.get("/v1/audio/voices")
async def list_voices():
    tier = get_sensory_tier()
    default_engine = "piper" if tier == "light" else "kokoro"
    default_voice = "piper_f" if tier == "light" else "jf_alpha"
    return {
        "voices": [
            {
                "id": "ef_blend",
                "name": "Blend (Kokoro Blend Español Nativo + Anime Dulce)",
                "engine": "kokoro",
                "tier": "high_fidelity",
                "description": "Mezcla neuronal nativa en español con entonación anime dulce (ef_dora + jf_alpha)",
                "language": "es",
                "gender": "female"
            },
            {
                "id": "jf_alpha",
                "name": "Alpha (Kokoro Anime Japonés / Español Dulce)",
                "engine": "kokoro",
                "tier": "high_fidelity",
                "description": "Estándar de alta fidelidad neuronal recomendada",
                "language": "es",
                "gender": "female"
            },
            {
                "id": "piper_f",
                "name": "Sharvard (Piper Local - Bajo consumo / Ultra-rápido en CPU)",
                "engine": "piper",
                "tier": "low_consumption_cpu",
                "description": "Bajo consumo y ultra-rápido en CPU para nodos modestos sin GPU",
                "language": "es",
                "gender": "female"
            },
            {
                "id": "jf_nezumi",
                "name": "Nezumi (Kokoro Anime Tierno)",
                "engine": "kokoro",
                "tier": "high_fidelity",
                "description": "Voz tierna de alta fidelidad",
                "language": "es",
                "gender": "female"
            },
            {
                "id": "jf_gongitsune",
                "name": "Gongitsune (Kokoro Anime Elegante)",
                "engine": "kokoro",
                "tier": "high_fidelity",
                "description": "Voz elegante de alta fidelidad",
                "language": "es",
                "gender": "female"
            },
            {
                "id": "jf_tebukuro",
                "name": "Tebukuro (Kokoro Anime Suave)",
                "engine": "kokoro",
                "tier": "high_fidelity",
                "description": "Voz suave de alta fidelidad",
                "language": "es",
                "gender": "female"
            },
            {
                "id": "ef_dora",
                "name": "Dora (Kokoro Local - Español Femenino)",
                "engine": "kokoro",
                "tier": "high_fidelity",
                "description": "Español nativo de alta fidelidad",
                "language": "es",
                "gender": "female"
            },
            {
                "id": "af_bella",
                "name": "Bella (Kokoro Local - Inglés Femenino)",
                "engine": "kokoro",
                "tier": "high_fidelity",
                "description": "Inglés nativo de alta fidelidad",
                "language": "en",
                "gender": "female"
            },
            {
                "id": "af_sky",
                "name": "Sky (Kokoro Local - Inglés Femenino)",
                "engine": "kokoro",
                "tier": "high_fidelity",
                "description": "Inglés nativo de alta fidelidad",
                "language": "en",
                "gender": "female"
            }
        ],
        "default_voice": default_voice,
        "default_engine": default_engine,
        "tier": tier,
        "engines": {
            "kokoro": {
                "name": "Kokoro ONNX",
                "tier": "high_fidelity",
                "description": "Estándar de alta fidelidad neuronal (GPU o CPU multi-hilo)"
            },
            "piper": {
                "name": "Piper ONNX",
                "tier": "low_consumption_cpu",
                "description": "Bajo consumo y ultra-rápido en CPU para nodos sin GPU"
            }
        }
    }
