"""Unit tests for PiperEngine adapter."""
import asyncio
from unittest.mock import patch

import pytest

from adapters.tts.piper_engine import PiperEngine, _phonemizer_lock
from domain.ports.text_to_speech import ITextToSpeechEngine, SynthesisResult


@pytest.fixture
def piper_engine() -> PiperEngine:
    """Fixture initializing PiperEngine with local models."""
    engine = PiperEngine()
    if not engine.is_ready():
        pytest.skip("Piper models not available on disk in sensory-server/models/piper")
    return engine


def test_piper_engine_satisfies_protocol(piper_engine: PiperEngine):
    """Verify PiperEngine satisfies ITextToSpeechEngine protocol."""
    assert isinstance(piper_engine, ITextToSpeechEngine)
    assert piper_engine.is_ready() is True


def test_piper_engine_synthesize_output_rate_and_pure_pcm(piper_engine: PiperEngine):
    """Verify synthesize output rate is strictly 22,050 Hz and pure 16-bit linear PCM."""
    async def _test():
        text = "Hola Rylai."
        result = await piper_engine.synthesize(text, speed=1.05)

        assert isinstance(result, SynthesisResult)
        assert result.sample_rate == 22050
        assert len(result.pcm_bytes) > 0
        # Must be 16-bit linear PCM (2 bytes per sample)
        assert len(result.pcm_bytes) % 2 == 0
        # Must NOT have a WAV container header ('RIFF')
        assert not result.pcm_bytes.startswith(b"RIFF")
        # Duration must be consistent with byte count: (len / (22050 * 2)) * 1000
        expected_duration = (len(result.pcm_bytes) / (22050 * 2)) * 1000.0
        assert abs(result.duration_ms - expected_duration) < 1e-3

    asyncio.run(_test())


def test_piper_engine_latency_telemetry(piper_engine: PiperEngine):
    """Verify latency telemetry dictionary is populated with timing breakdown and RTF."""
    async def _test():
        result = await piper_engine.synthesize("Sí, estoy lista.")

        metrics = result.latency_metrics
        assert "phonemize_ms" in metrics
        assert "inference_ms" in metrics
        assert "total_ms" in metrics
        assert "rtf" in metrics

        assert metrics["phonemize_ms"] > 0.0
        assert metrics["inference_ms"] > 0.0
        assert metrics["total_ms"] > 0.0
        assert metrics["rtf"] > 0.0
        # Inference + phonemize should be less than or equal to total elapsed time
        assert metrics["total_ms"] >= (metrics["phonemize_ms"] + metrics["inference_ms"]) * 0.90

    asyncio.run(_test())


def test_piper_engine_empty_input(piper_engine: PiperEngine):
    """Verify empty or whitespace-only text returns empty PCM result without error."""
    async def _test():
        result = await piper_engine.synthesize("   ")
        assert result.pcm_bytes == b""
        assert result.sample_rate == 22050
        assert result.duration_ms == 0.0
        assert result.latency_metrics["total_ms"] == 0.0

    asyncio.run(_test())


def test_piper_engine_speed_scaling(piper_engine: PiperEngine):
    """Verify speed scaling alters audio duration proportionally."""
    async def _test():
        text = "Probando la velocidad de reproducción de voz."
        result_normal = await piper_engine.synthesize(text, speed=1.0)
        result_fast = await piper_engine.synthesize(text, speed=1.5)

        assert result_fast.duration_ms < result_normal.duration_ms
        assert len(result_fast.pcm_bytes) < len(result_normal.pcm_bytes)

    asyncio.run(_test())


def test_piper_engine_phonemizer_phase_separation_and_thread_safety(piper_engine: PiperEngine):
    """
    Verify Phase Separation:
    - Phonemizer is guarded by _phonemizer_lock.
    - ONNX inference runs lock-free (lock is released).
    - Multiple concurrent synthesis tasks run safely.
    """
    async def _test():
        lock_held_during_phonemize = []
        lock_held_during_synthesis = []
        original_voice_phonemize = piper_engine._voice.phonemize
        original_waveform = piper_engine._synthesize_waveform

        def instrumented_phonemize(text: str):
            lock_held_during_phonemize.append(_phonemizer_lock.locked())
            return original_voice_phonemize(text)

        def instrumented_waveform(sentence_phonemes, syn_config):
            lock_held_during_synthesis.append(_phonemizer_lock.locked())
            return original_waveform(sentence_phonemes, syn_config)

        with patch.object(piper_engine._voice, "phonemize", side_effect=instrumented_phonemize), \
             patch.object(piper_engine, "_synthesize_waveform", side_effect=instrumented_waveform):
            texts = [
                "Primera frase simultánea.",
                "Segunda frase simultánea.",
                "Tercera frase simultánea.",
            ]
            tasks = [piper_engine.synthesize(t, generation_id=f"concurrent_{i}") for i, t in enumerate(texts)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 3
            # In all phonemize calls, the lock was held
            assert len(lock_held_during_phonemize) == 3
            assert all(lock_held_during_phonemize)
            # In all synthesis calls, the lock was NOT held (phase separation!)
            assert len(lock_held_during_synthesis) == 3
            assert not any(lock_held_during_synthesis)
            for res in results:
                assert len(res.pcm_bytes) > 0
                assert res.sample_rate == 22050

    asyncio.run(_test())


def test_piper_engine_abort_cancellation(piper_engine: PiperEngine):
    """Verify abort(generation_id) immediately cancels in-flight synthesis tasks."""
    async def _test():
        gen_id = "test_turn_abort_123"

        async def slow_synthesize():
            return await piper_engine.synthesize(
                "Esta es una frase deliberadamente larga para ser cancelada a mitad de camino.",
                generation_id=gen_id,
            )

        task = asyncio.create_task(slow_synthesize())
        # Yield control briefly so task registers in _active_tasks
        await asyncio.sleep(0.001)

        assert gen_id in piper_engine._active_tasks
        assert task in piper_engine._active_tasks[gen_id]

        await piper_engine.abort(gen_id)

        with pytest.raises(asyncio.CancelledError):
            await task

        # Task must be cleaned up from registry
        assert gen_id not in piper_engine._active_tasks

    asyncio.run(_test())
