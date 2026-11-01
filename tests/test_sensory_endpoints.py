"""
Tests unitarios para los endpoints de Aoi Sensory Hub (FastAPI).
Verifica:
- Middleware de logging y propagación del header X-Request-ID.
- Endpoint /health reportando estado de modelos y degradación.
- Endpoint /v1/audio/transcriptions devolviendo HTTP 503 si el reconocedor no está disponible.
- Capa 3: Detección elástica de hardware (GPU vs CPU) y session options.
- Capa 3: Selector de motor y exposición de voces (/v1/audio/voices y /v1/audio/speech).
"""
import os
import io
import time
import wave
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import numpy as np

from app import app
import routers.stt_router as stt_router
import routers.tts_router as tts_router


class TestSensoryEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check_endpoint(self):
        """Verifica la respuesta del health check con estructura de modelos y estado."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn(data["status"], ("healthy", "degraded", "down"))
        self.assertIn("models", data)
        self.assertIn("stt", data["models"])
        self.assertIn("kokoro", data["models"])
        self.assertIn("piper", data["models"])
        self.assertTrue("x-request-id" in response.headers or "X-Request-ID" in response.headers)

    def test_request_logging_middleware_preserves_custom_request_id(self):
        """Verifica que si el cliente envía X-Request-ID, el middleware lo devuelva inalterado."""
        custom_id = "test-req-12345"
        response = self.client.get("/health", headers={"X-Request-ID": custom_id})
        returned_id = response.headers.get("X-Request-ID") or response.headers.get("x-request-id")
        self.assertEqual(returned_id, custom_id)

    def test_stt_returns_503_when_recognizer_is_none(self):
        """Verifica que si el reconocedor Whisper no carga, devuelva HTTP 503."""
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 8000)
        wav_bytes = wav_io.getvalue()

        with patch.object(stt_router, "get_recognizer", return_value=None):
            files = {"file": ("test.wav", wav_bytes, "audio/wav")}
            response = self.client.post("/v1/audio/transcriptions", files=files, data={"model": "whisper-base"})
            self.assertEqual(response.status_code, 503)
            data = response.json()
            self.assertIn("STT model not loaded", data.get("detail", ""))

    def test_stt_recognizer_threads_and_hardware_elasticity(self):
        """Verifica que get_recognizer configure num_threads = min(4, os.cpu_count() or 4) para P-cores."""
        expected_threads = min(4, os.cpu_count() or 4)
        with patch("sherpa_onnx.OfflineRecognizer.from_whisper") as mock_whisper:
            mock_whisper.return_value = MagicMock()
            saved_rec = stt_router.recognizer
            try:
                stt_router.recognizer = None
                rec = stt_router.get_recognizer()
                self.assertIsNotNone(rec)
                self.assertTrue(mock_whisper.called)
                _, kwargs = mock_whisper.call_args
                self.assertEqual(kwargs.get("num_threads"), expected_threads)
            finally:
                stt_router.recognizer = saved_rec

    def test_tts_session_options_threads(self):
        """Verifica que SessionOptions en CPU configure intra_op_num_threads = min(4, os.cpu_count() or 4) e inter_op=1."""
        expected_threads = min(4, os.cpu_count() or 4)
        sess_options = tts_router._get_elastic_session_options()
        self.assertEqual(sess_options.intra_op_num_threads, expected_threads)
        self.assertEqual(sess_options.inter_op_num_threads, 1)
        self.assertEqual(sess_options.log_severity_level, 3)

    def test_tts_hardware_detection_gpu_to_cpu_fallback(self):
        """Verifica que si GPU falla o no está disponible, degrade limpiamente a CPUExecutionProvider."""
        with patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]):
            with patch("onnxruntime.InferenceSession") as mock_sess:
                # Simular que el intento de GPU lanza excepción y luego el de CPU funciona
                cpu_session_mock = MagicMock()
                cpu_session_mock.get_providers.return_value = ["CPUExecutionProvider"]
                mock_sess.side_effect = [RuntimeError("CUDA driver not found"), cpu_session_mock]

                session, provider = tts_router._create_elastic_inference_session("dummy_model.onnx")
                self.assertEqual(provider, "CPUExecutionProvider")
                self.assertEqual(mock_sess.call_count, 2)

    def test_voices_endpoint_exposes_piper_and_kokoro(self):
        """Verifica que /v1/audio/voices exponga piper_f como bajo consumo en CPU y jf_alpha como alta fidelidad."""
        response = self.client.get("/v1/audio/voices")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("voices", data)
        voices = data["voices"]
        voice_ids = [v["id"] for v in voices]

        # Verificar presencia de jf_alpha (Kokoro) y piper_f (Piper)
        self.assertIn("jf_alpha", voice_ids)
        self.assertIn("piper_f", voice_ids)

        # Verificar metadata de piper_f
        piper_meta = next(v for v in voices if v["id"] == "piper_f")
        self.assertEqual(piper_meta.get("engine"), "piper")
        self.assertIn("CPU", piper_meta.get("name", ""))
        self.assertEqual(piper_meta.get("tier"), "low_consumption_cpu")

        # Verificar metadata de jf_alpha
        kokoro_meta = next(v for v in voices if v["id"] == "jf_alpha")
        self.assertEqual(kokoro_meta.get("engine"), "kokoro")
        self.assertEqual(kokoro_meta.get("tier"), "high_fidelity")

    def test_speech_endpoint_engine_routing_piper(self):
        """Verifica que model='piper' active el motor Piper para síntesis ultra-rápida."""
        dummy_wav = tts_router._generate_wav_bytes(b"\x00" * 4410, sample_rate=22050)
        fake_metrics = {
            "voice": "piper_f",
            "engine": "piper",
            "lang": "es",
            "chars": 4,
            "lock_wait_ms": 1.0,
            "infer_ms": 5.0,
            "audio_dur_ms": 100.0,
            "bytes": len(dummy_wav)
        }
        with patch.object(tts_router, "_synthesize_wav_sync", return_value=(dummy_wav, fake_metrics)) as mock_synth:
            response = self.client.post("/v1/audio/speech", json={"model": "piper", "input": "Hola"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers.get("X-TTS-Engine"), "piper")
            self.assertEqual(response.headers.get("X-TTS-Voice"), "piper_f")
            _, kwargs = mock_synth.call_args
            self.assertEqual(kwargs.get("model"), "piper")

    def test_anime_pitch_shift_benchmark_under_10ms(self):
        """Verifica que _apply_anime_pitch_shift sobre 24000 muestras (1s a 24kHz) ejecute en < 10ms."""
        sample_rate = 24000
        # 1 segundo de audio sintético a 24kHz
        samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate, dtype=np.float32))

        # Warm-up previo
        tts_router._apply_anime_pitch_shift(samples, sample_rate, 2.0)

        # Benchmark de ejecución
        t_start = time.perf_counter()
        shifted = tts_router._apply_anime_pitch_shift(samples, sample_rate, 2.0)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Verificación de rendimiento estricto (< 10 ms)
        self.assertLess(
            elapsed_ms,
            10.0,
            f"El remuestreo tardó {elapsed_ms:.2f}ms, excediendo el límite de 10ms"
        )

        # Verificaciones de sanidad y preservación estricta de tempo y duración
        self.assertEqual(len(shifted), len(samples))
        self.assertEqual(shifted.dtype, samples.dtype)

        # Casos borde: tono despreciable o array vacío
        noop = tts_router._apply_anime_pitch_shift(samples, sample_rate, 0.0)
        np.testing.assert_array_equal(noop, samples)

        empty = np.array([], dtype=np.float32)
        empty_shifted = tts_router._apply_anime_pitch_shift(empty, sample_rate, 2.0)
        self.assertEqual(len(empty_shifted), 0)

    def test_spanish_inverted_punctuation_preserved_and_markdown_stripped(self):
        """Verifica que ¿ y ¡ se preserven para espeak-ng, mientras se eliminan roleplay, emojis y markdown."""
        raw_text = "*sonríe con emoción* ¡Hola Rylai! ¿Cómo estás hoy? 🌸✨ Mira esto: https://example.com ```código en bloque``` **genial** # Título ~tachado~"
        cleaned = tts_router._sanitize_text_for_tts(raw_text)
        # ¿ e ¡ deben preservarse intactos
        self.assertIn("¿Cómo estás hoy?", cleaned)
        self.assertIn("¡Hola Rylai!", cleaned)
        # *sonríe con emoción* debe eliminarse por completo (no leerse)
        self.assertNotIn("sonríe", cleaned)
        # Emojis eliminados
        self.assertNotIn("🌸", cleaned)
        self.assertNotIn("✨", cleaned)
        # URLs y bloques de código eliminados
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("código en bloque", cleaned)
        self.assertNotIn("*", cleaned)
        self.assertNotIn("#", cleaned)

    def test_detect_phoneme_lang_prefers_spanish(self):
        """Verifica que voces jf_* no fuercen fonemas japoneses 'ja' en texto latino/español."""
        # En español no debe devolver 'ja'
        self.assertEqual(tts_router._detect_phoneme_lang("Hola Rylai, ¿cómo estás?", "jf_alpha"), "es")
        self.assertEqual(tts_router._detect_phoneme_lang("Buenos días", "ef_dora"), "es")
        # Solo en presencia de caracteres japoneses devuelve 'ja'
        self.assertEqual(tts_router._detect_phoneme_lang("こんにちは", "jf_alpha"), "ja")
        # Idioma explícito
        self.assertEqual(tts_router._detect_phoneme_lang("Hello friend", "af_bella", requested_lang="en-us"), "en-us")

    def test_soft_limiting_peak_normalization(self):
        """Verifica que _write_wav_bytes aplique soft-limiting cuando el pico supera 0.98."""
        loud_samples = np.array([0.0, 1.5, -1.8, 0.5], dtype=np.float32)
        wav_bytes = tts_router._write_wav_bytes(loud_samples, sample_rate=24000)
        read_samples, sr = tts_router._read_wav_samples(wav_bytes)
        self.assertEqual(sr, 24000)
        # El pico normalizado debe ser <= 1.0 (sin clipping digital de techo plano)
        self.assertLessEqual(float(np.max(np.abs(read_samples))), 1.0)

    def test_kokoro_voice_blending_resolution(self):
        """Verifica que _resolve_kokoro_voice mezcle ef_dora + jf_alpha para español anime."""
        mock_kokoro = MagicMock()
        mock_kokoro.get_voices.return_value = ["ef_dora", "jf_alpha", "jf_nezumi"]
        mock_kokoro.get_voice_style.side_effect = lambda name: np.ones((510, 1, 256), dtype=np.float32) * (1.0 if name == "ef_dora" else 2.0)

        # Caso ef_rylai o jf_alpha en español: debe devolver el vector de estilo mezclado
        blended = tts_router._resolve_kokoro_voice(mock_kokoro, "jf_alpha", "es")
        self.assertIsInstance(blended, np.ndarray)
        self.assertEqual(blended.shape, (510, 1, 256))
        # 0.70 * 1.0 + 0.30 * 2.0 = 1.30
        expected_val = 0.70 * 1.0 + 0.30 * 2.0
        np.testing.assert_allclose(blended[0, 0, 0], expected_val, atol=1e-5)

        # Fallback si kokoro no tiene get_voice_style (mock simple de test)
        simple_mock = MagicMock(spec=["create"])
        fallback_voice = tts_router._resolve_kokoro_voice(simple_mock, "jf_alpha", "es")
        self.assertEqual(fallback_voice, "jf_alpha")

    def test_voices_endpoint_includes_ef_rylai(self):
        """Verifica que /v1/audio/voices incluya la voz ef_rylai."""
        response = self.client.get("/v1/audio/voices")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        voice_ids = [v["id"] for v in data["voices"]]
        self.assertIn("ef_rylai", voice_ids)


if __name__ == "__main__":
    unittest.main()
