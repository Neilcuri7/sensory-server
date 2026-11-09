"""Piper ONNX Text-to-Speech Engine Adapter with Phase-Separated Phonemizer."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Optional

import numpy as np
import onnxruntime as ort

from domain.ports.text_to_speech import ITextToSpeechEngine, SynthesisResult

try:
    from piper import PiperVoice
    from piper.config import PiperConfig, SynthesisConfig
except ImportError:
    PiperVoice = None  # type: ignore[assignment, misc]
    PiperConfig = None  # type: ignore[assignment, misc]
    SynthesisConfig = None  # type: ignore[assignment, misc]

logger = logging.getLogger("Sensory.PiperEngine")

# Module-level lock guarding the C-level libespeak-ng mutable phonemizer state
_phonemizer_lock = threading.Lock()

DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "piper",
)
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "es_ES-sharvard-medium.onnx")
DEFAULT_CONFIG_PATH = os.path.join(DEFAULT_MODEL_DIR, "es_ES-sharvard-medium.onnx.json")


class PiperEngine(ITextToSpeechEngine):
    """
    ITextToSpeechEngine adapter wrapping PiperVoice with:
    - Persistent ONNX session pinned to intra_op=2, inter_op=1 threads.
    - Phase Separation: libespeak-ng phonemization guarded by _phonemizer_lock (<2ms),
      ONNX inference running completely lock-free across worker threads (40-60ms).
    - SynthesisConfig tuned for avatar conversational pacing (speaker_id=1, length_scale=0.95).
    - In-flight task registry for instantaneous barge-in aborts.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        voice: Optional[Any] = None,
        intra_op_threads: int = 2,
        inter_op_threads: int = 1,
    ) -> None:
        self._intra_op_threads = intra_op_threads
        self._inter_op_threads = inter_op_threads
        self._active_tasks: dict[str, list[asyncio.Task[Any]]] = {}
        self._aborted_generations: set[str] = set()

        if voice is not None:
            self._voice = voice
            return

        resolved_model = str(model_path) if model_path else DEFAULT_MODEL_PATH
        resolved_config = (
            str(config_path)
            if config_path
            else (f"{resolved_model}.json" if os.path.exists(f"{resolved_model}.json") else DEFAULT_CONFIG_PATH)
        )

        self._voice = None
        if os.path.exists(resolved_model) and os.path.exists(resolved_config):
            self._load_voice(resolved_model, resolved_config)
        else:
            logger.warning(
                "Piper voice model not found at %s. Engine marked not ready.",
                resolved_model,
            )

    def _load_voice(self, model_path: str, config_path: str) -> None:
        """Initialize persistent ONNX InferenceSession and PiperVoice."""
        if PiperVoice is None or PiperConfig is None:
            logger.error("piper-tts package is not installed.")
            return

        try:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self._intra_op_threads
            sess_options.inter_op_num_threads = self._inter_op_threads
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )

            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

            voice_config = PiperConfig.from_dict(config_dict)

            # Honor ESPEAK_DATA_PATH from environment if configured
            espeak_data_dir = os.environ.get("ESPEAK_DATA_PATH")
            kwargs: dict[str, Any] = {}
            if espeak_data_dir and os.path.exists(espeak_data_dir):
                kwargs["espeak_data_dir"] = Path(espeak_data_dir)

            self._voice = PiperVoice(
                config=voice_config,
                session=session,
                **kwargs,
            )
            logger.info(
                "PiperEngine loaded successfully (intra=%d, inter=%d, sample_rate=%d).",
                self._intra_op_threads,
                self._inter_op_threads,
                self._voice.config.sample_rate,
            )
        except Exception as exc:
            logger.error("Failed to load Piper voice model: %s", exc, exc_info=True)
            self._voice = None

    def is_ready(self) -> bool:
        """Check if Piper engine is initialized with an active session."""
        return self._voice is not None and getattr(self._voice, "session", None) is not None

    def _phonemize(self, text: str) -> list[list[str]]:
        """Phase 1: Serialized text-to-phonemes step protected by process-level lock."""
        with _phonemizer_lock:
            return self._voice.phonemize(text)

    def _synthesize_waveform(
        self,
        sentence_phonemes: list[list[str]],
        syn_config: Any,
    ) -> bytes:
        """Phase 2: Lock-free phonemes-to-waveform ONNX synthesis."""
        pcm_chunks: list[bytes] = []
        for phonemes in sentence_phonemes:
            if not phonemes:
                continue
            phoneme_ids = self._voice.phonemes_to_ids(phonemes)
            # Execute ONNX Runtime waveform generation lock-free
            audio = self._voice.phoneme_ids_to_audio(phoneme_ids, syn_config=syn_config)
            # Convert float32 array in [-1.0, 1.0] to 16-bit linear PCM
            int16_audio = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
            pcm_chunks.append(int16_audio.tobytes())

        return b"".join(pcm_chunks)

    async def synthesize(
        self,
        text: str,
        speed: float = 1.05,
        generation_id: str | None = None,
    ) -> SynthesisResult:
        """
        Synthesize text into pure 16-bit linear PCM at 22,050 Hz.
        Tracks active task for instant cancellation on abort(generation_id).
        """
        if not self.is_ready():
            raise RuntimeError("Piper TTS voice is not loaded or ready")

        sample_rate = getattr(getattr(self._voice, "config", None), "sample_rate", 22050)

        if not text or not text.strip():
            return SynthesisResult(
                pcm_bytes=b"",
                sample_rate=sample_rate,
                duration_ms=0.0,
                latency_metrics={
                    "phonemize_ms": 0.0,
                    "inference_ms": 0.0,
                    "total_ms": 0.0,
                    "rtf": 0.0,
                },
            )

        current_task = asyncio.current_task()
        if generation_id and current_task:
            self._active_tasks.setdefault(generation_id, []).append(current_task)

        try:
            if generation_id and generation_id in self._aborted_generations:
                raise asyncio.CancelledError(f"Generation '{generation_id}' was aborted.")

            # Calculate length_scale calibrated to speed: base 0.95 / speed
            effective_length_scale = 0.95 / speed if speed > 0 else 0.95

            # Speaker 1 corresponds to female voice in multi-speaker Piper voices (e.g., sharvard)
            num_speakers = getattr(getattr(self._voice, "config", None), "num_speakers", 1)
            speaker_id = 1 if num_speakers > 1 else None

            try:
                syn_config = SynthesisConfig(
                    speaker_id=speaker_id,
                    length_scale=effective_length_scale,
                    noise_scale=0.667,
                    noise_w_scale=0.80,
                )
            except TypeError:
                syn_config = SynthesisConfig(
                    speaker_id=speaker_id,
                    length_scale=effective_length_scale,
                    noise_scale=0.667,
                    noise_w=0.80,
                )

            t_start = time.perf_counter()

            # Phase 1: Phonemize with libespeak lock
            t0 = time.perf_counter()
            sentence_phonemes = await asyncio.to_thread(self._phonemize, text)
            phonemize_ms = (time.perf_counter() - t0) * 1000.0

            if generation_id and generation_id in self._aborted_generations:
                raise asyncio.CancelledError(f"Generation '{generation_id}' was aborted.")

            # Phase 2: Lock-free ONNX waveform generation
            t1 = time.perf_counter()
            pcm_bytes = await asyncio.to_thread(
                self._synthesize_waveform, sentence_phonemes, syn_config
            )
            inference_ms = (time.perf_counter() - t1) * 1000.0

            total_ms = (time.perf_counter() - t_start) * 1000.0
            duration_ms = (
                (len(pcm_bytes) / (sample_rate * 2)) * 1000.0 if sample_rate > 0 else 0.0
            )
            rtf = (total_ms / duration_ms) if duration_ms > 0 else 0.0

            return SynthesisResult(
                pcm_bytes=pcm_bytes,
                sample_rate=sample_rate,
                duration_ms=duration_ms,
                latency_metrics={
                    "phonemize_ms": phonemize_ms,
                    "inference_ms": inference_ms,
                    "total_ms": total_ms,
                    "rtf": rtf,
                },
            )
        finally:
            if generation_id and current_task:
                tasks = self._active_tasks.get(generation_id)
                if tasks and current_task in tasks:
                    tasks.remove(current_task)
                    if not tasks:
                        self._active_tasks.pop(generation_id, None)
                if generation_id not in self._active_tasks:
                    self._aborted_generations.discard(generation_id)

    async def abort(self, generation_id: str) -> None:
        """Cancel in-flight synthesis tasks matching generation_id immediately."""
        self._aborted_generations.add(generation_id)
        tasks = self._active_tasks.pop(generation_id, [])
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.sleep(0)
