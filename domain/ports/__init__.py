"""Domain ports for Sensory Server."""
from .speech_to_text import ISpeechToTextEngine, TranscriptionResult
from .text_to_speech import ITextToSpeechEngine, SynthesisResult

__all__ = [
    "ISpeechToTextEngine",
    "TranscriptionResult",
    "ITextToSpeechEngine",
    "SynthesisResult",
]
