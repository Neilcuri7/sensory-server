"""Unit tests for SherpaWhisperAdapter."""
import asyncio
import numpy as np
import pytest

from adapters.stt.sherpa_whisper import SherpaWhisperAdapter
from domain.ports.speech_to_text import ISpeechToTextEngine, TranscriptionResult


@pytest.fixture
def sherpa_adapter() -> SherpaWhisperAdapter:
    """Fixture initializing SherpaWhisperAdapter with local models."""
    adapter = SherpaWhisperAdapter()
    if not adapter.is_ready():
        pytest.skip("Sherpa whisper models not available on disk")
    return adapter


def test_sherpa_whisper_satisfies_protocol(sherpa_adapter: SherpaWhisperAdapter):
    """Verify SherpaWhisperAdapter satisfies ISpeechToTextEngine protocol."""
    assert isinstance(sherpa_adapter, ISpeechToTextEngine)
    assert sherpa_adapter.is_ready() is True


def test_sherpa_whisper_empty_audio(sherpa_adapter: SherpaWhisperAdapter):
    """Verify empty audio bytes return empty TranscriptionResult immediately."""
    async def _test():
        result = await sherpa_adapter.transcribe(b"", sample_rate=16000)
        assert isinstance(result, TranscriptionResult)
        assert result.text == ""
        assert result.confidence == 0.0
        assert result.duration_ms == 0.0

    asyncio.run(_test())


def test_sherpa_whisper_ingests_raw_linear_pcm(sherpa_adapter: SherpaWhisperAdapter):
    """Verify raw 16-bit linear PCM frames are directly transcribed without WAV containers."""
    async def _test():
        # 0.5s of silence as pure 16-bit linear PCM
        silence_pcm = np.zeros(8000, dtype=np.int16).tobytes()
        result = await sherpa_adapter.transcribe(silence_pcm, sample_rate=16000)

        assert isinstance(result, TranscriptionResult)
        assert abs(result.duration_ms - 500.0) < 1e-2

    asyncio.run(_test())
