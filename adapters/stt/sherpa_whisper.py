"""Sherpa-ONNX Whisper STT Engine Adapter consuming raw 16-bit linear PCM frames."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Optional

import numpy as np

from domain.ports.speech_to_text import ISpeechToTextEngine, TranscriptionResult

try:
    import sherpa_onnx
except ImportError:
    sherpa_onnx = None  # type: ignore[assignment]

logger = logging.getLogger("Sensory.SherpaWhisper")

_RYLAI_MISSPELLINGS_REGEX = re.compile(
    r"\b(rilai|rilay|railai|raily|rylay|relai)\b",
    re.IGNORECASE,
)

DEFAULT_WHISPER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "sherpa-onnx-whisper-base",
)
DEFAULT_ENCODER = os.path.join(DEFAULT_WHISPER_DIR, "base-encoder.int8.onnx")
DEFAULT_DECODER = os.path.join(DEFAULT_WHISPER_DIR, "base-decoder.int8.onnx")
DEFAULT_TOKENS = os.path.join(DEFAULT_WHISPER_DIR, "base-tokens.txt")


def _apply_domain_lexicon_corrections(text: str) -> str:
    """Correct frequent Spanish phonetic misspellings of the companion name 'Rylai'."""
    if not text:
        return ""
    return _RYLAI_MISSPELLINGS_REGEX.sub("Rylai", text)


class SherpaWhisperAdapter(ISpeechToTextEngine):
    """
    ISpeechToTextEngine adapter wrapping sherpa_onnx.OfflineRecognizer with:
    - Direct ingestion of raw 16-bit linear PCM frames (zero WAV container overhead).
    - Hardware thread budget pinned to 2 intra-op threads.
    - Fast numpy linear resampling to 16 kHz.
    - Thread-safe decoding stream management.
    """

    def __init__(
        self,
        encoder_path: str | Path | None = None,
        decoder_path: str | Path | None = None,
        tokens_path: str | Path | None = None,
        num_threads: int = 2,
        language: str = "es",
        task: str = "transcribe",
        provider: str = "cpu",
        recognizer: Optional[Any] = None,
    ) -> None:
        self._num_threads = num_threads
        self._language = language
        self._task = task
        self._provider = provider
        self._lock = threading.Lock()

        if recognizer is not None:
            self._recognizer = recognizer
            return

        resolved_encoder = str(encoder_path) if encoder_path else DEFAULT_ENCODER
        resolved_decoder = str(decoder_path) if decoder_path else DEFAULT_DECODER
        resolved_tokens = str(tokens_path) if tokens_path else DEFAULT_TOKENS

        self._recognizer = None
        if (
            os.path.exists(resolved_encoder)
            and os.path.exists(resolved_decoder)
            and os.path.exists(resolved_tokens)
        ):
            self._load_recognizer(resolved_encoder, resolved_decoder, resolved_tokens)
        else:
            logger.warning(
                "Sherpa Whisper models not found in %s. Engine marked not ready.",
                DEFAULT_WHISPER_DIR,
            )

    def _load_recognizer(self, encoder: str, decoder: str, tokens: str) -> None:
        """Initialize sherpa_onnx.OfflineRecognizer pinned to num_threads."""
        if sherpa_onnx is None:
            logger.error("sherpa-onnx package is not installed.")
            return

        try:
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=encoder,
                decoder=decoder,
                tokens=tokens,
                num_threads=self._num_threads,
                language=self._language,
                task=self._task,
                debug=False,
                provider=self._provider,
            )
            logger.info(
                "SherpaWhisperAdapter loaded successfully (threads=%d, provider=%s, lang=%s).",
                self._num_threads,
                self._provider,
                self._language,
            )
        except Exception as exc:
            logger.error("Failed to load Sherpa Whisper recognizer: %s", exc, exc_info=True)
            self._recognizer = None

    def is_ready(self) -> bool:
        """Return True if recognizer is loaded and ready for transcription."""
        return self._recognizer is not None

    def _sync_decode(self, samples: np.ndarray, sample_rate: int) -> str:
        """Execute synchronous stream decoding under lock."""
        with self._lock:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            self._recognizer.decode_stream(stream)
            raw_text = stream.result.text.strip()
            return _apply_domain_lexicon_corrections(raw_text)

    async def transcribe(
        self, audio_bytes: bytes, sample_rate: int = 16000
    ) -> TranscriptionResult:
        """
        Transcribe raw 16-bit linear PCM audio bytes directly without intermediate WAV headers.
        """
        if not self.is_ready():
            raise RuntimeError("Sherpa Whisper STT engine is not loaded or ready")

        if not audio_bytes:
            return TranscriptionResult(text="", confidence=0.0, duration_ms=0.0)

        t_start = time.perf_counter()

        # Ingest pure 16-bit signed PCM frames directly into float32 array in [-1.0, 1.0]
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        raw_duration_ms = (
            (len(samples) / float(sample_rate)) * 1000.0 if sample_rate > 0 else 0.0
        )

        # Fast linear resampling if sample_rate is not 16,000 Hz
        target_sample_rate = sample_rate
        if sample_rate != 16000 and sample_rate > 0 and len(samples) > 0:
            target_len = max(1, int(round(len(samples) * 16000.0 / float(sample_rate))))
            orig_idx = np.linspace(0.0, len(samples) - 1, target_len, dtype=np.float32)
            samples = np.interp(
                orig_idx, np.arange(len(samples), dtype=np.float32), samples
            ).astype(np.float32)
            target_sample_rate = 16000

        # DC offset removal and peak gain adjustment
        if len(samples) > 0:
            samples = samples - float(np.mean(samples))
            peak = float(np.max(np.abs(samples)))
            if 0.015 <= peak < 0.85:
                samples = (samples * (0.89 / peak)).astype(np.float32)

        # Offload sync ONNX decoding to worker thread
        text = await asyncio.to_thread(self._sync_decode, samples, target_sample_rate)

        return TranscriptionResult(
            text=text,
            confidence=1.0 if text else 0.0,
            duration_ms=raw_duration_ms,
        )


SherpaWhisperEngine = SherpaWhisperAdapter
