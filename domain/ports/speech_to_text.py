"""Speech-to-text domain port and result models."""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class TranscriptionResult:
    """Dataclass holding the result of a speech-to-text transcription."""
    text: str
    confidence: float = 1.0
    duration_ms: float = 0.0


@runtime_checkable
class ISpeechToTextEngine(Protocol):
    """Abstract protocol for speech-to-text transcription engines."""

    async def transcribe(
        self, audio_bytes: bytes, sample_rate: int = 16000
    ) -> TranscriptionResult:
        """Transcribe raw PCM audio bytes to text."""
        ...

    def is_ready(self) -> bool:
        """Return True if the STT engine is loaded and ready for inference."""
        ...
