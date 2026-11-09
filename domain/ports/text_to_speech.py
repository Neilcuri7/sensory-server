"""Text-to-speech domain port and result models."""
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class SynthesisResult:
    """Dataclass holding synthesized raw PCM audio and latency telemetry."""
    pcm_bytes: bytes
    sample_rate: int = 22050
    duration_ms: float = 0.0
    latency_metrics: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class ITextToSpeechEngine(Protocol):
    """Abstract protocol for text-to-speech synthesis engines."""

    async def synthesize(
        self,
        text: str,
        speed: float = 1.05,
        generation_id: str | None = None,
    ) -> SynthesisResult:
        """Synthesize input text into pure linear PCM audio frames."""
        ...

    async def abort(self, generation_id: str) -> None:
        """Cancel any ongoing or queued synthesis tasks matching generation_id."""
        ...

    def is_ready(self) -> bool:
        """Return True if the TTS engine is loaded and ready for inference."""
        ...
