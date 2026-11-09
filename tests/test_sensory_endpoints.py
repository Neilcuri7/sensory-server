"""
Sensory Server Endpoint Unit and Integration Tests.
Verifies:
- GET /health reporting active_engine='piper', stt_ready, and tts_ready.
- RequestLoggingMiddleware propagating X-Request-ID headers.
- POST /v1/audio/speech and POST /tts returning pure 16-bit linear PCM at 22,050 Hz.
- POST /v1/audio/speech/abort and POST /tts/abort returning 200.
- POST /v1/audio/transcriptions and POST /stt transcribing cleanly.
- Error handling (empty text/audio -> 400, unready engine -> 503).
"""
from __future__ import annotations

import base64
import io
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import wave

from fastapi.testclient import TestClient
import numpy as np

from app import app
from domain.ports.speech_to_text import ISpeechToTextEngine, TranscriptionResult
from domain.ports.text_to_speech import ITextToSpeechEngine, SynthesisResult
import routers.stt_router as stt_router
import routers.tts_router as tts_router


class TestSensoryEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check_endpoint(self):
        """Verify GET /health reports status, active_engine='piper', and engine readiness."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("status", data)
        self.assertIn(data["status"], ("ok", "degraded"))
        self.assertEqual(data["active_engine"], "piper")
        self.assertIn("stt_ready", data)
        self.assertIn("tts_ready", data)
        self.assertIn("models", data)
        self.assertIn("piper", data["models"])
        self.assertIn("stt", data["models"])
        self.assertTrue("X-Request-ID" in response.headers or "x-request-id" in response.headers)

    def test_request_logging_middleware_preserves_custom_request_id(self):
        """Verify X-Request-ID supplied by client is echoed in response headers."""
        custom_id = "test-req-custom-999"
        response = self.client.get("/health", headers={"X-Request-ID": custom_id})
        returned_id = response.headers.get("X-Request-ID") or response.headers.get("x-request-id")
        self.assertEqual(returned_id, custom_id)

    def test_speech_endpoint_returns_pure_pcm(self):
        """Verify POST /v1/audio/speech returns pure 16-bit linear PCM at 22,050 Hz."""
        dummy_pcm = (np.sin(np.linspace(0, 100, 2205)) * 10000).astype(np.int16).tobytes()
        mock_result = SynthesisResult(
            pcm_bytes=dummy_pcm,
            sample_rate=22050,
            duration_ms=100.0,
            latency_metrics={"inference_ms": 15.0},
        )

        mock_engine = MagicMock(spec=ITextToSpeechEngine)
        mock_engine.is_ready.return_value = True
        mock_engine.synthesize = AsyncMock(return_value=mock_result)

        with patch.object(tts_router, "get_tts_engine", return_value=mock_engine):
            response = self.client.post(
                "/v1/audio/speech",
                json={
                    "model": "piper",
                    "input": "Hola Rylai",
                    "voice": "piper_f",
                    "speed": 1.05,
                    "generation_id": "gen_test_123",
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("audio/pcm", response.headers.get("Content-Type", ""))
            self.assertEqual(response.headers.get("X-TTS-Engine"), "piper")
            self.assertEqual(response.headers.get("X-TTS-Sample-Rate"), "22050")
            self.assertEqual(response.content, dummy_pcm)
            # Pure PCM has no RIFF/WAVE header
            self.assertFalse(response.content.startswith(b"RIFF"))
            self.assertEqual(len(response.content) % 2, 0)

            mock_engine.synthesize.assert_called_once_with(
                text="Hola Rylai",
                speed=1.05,
                generation_id="gen_test_123",
            )

    def test_tts_legacy_alias_endpoint_returns_pure_pcm(self):
        """Verify POST /tts returns pure PCM matching POST /v1/audio/speech."""
        dummy_pcm = (np.zeros(1000, dtype=np.int16)).tobytes()
        mock_result = SynthesisResult(
            pcm_bytes=dummy_pcm,
            sample_rate=22050,
            duration_ms=45.0,
        )

        mock_engine = MagicMock(spec=ITextToSpeechEngine)
        mock_engine.is_ready.return_value = True
        mock_engine.synthesize = AsyncMock(return_value=mock_result)

        with patch.object(tts_router, "get_tts_engine", return_value=mock_engine):
            response = self.client.post("/tts", json={"text": "Prueba de audio", "speed": 1.0})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, dummy_pcm)

    def test_speech_endpoint_empty_text_returns_400(self):
        """Verify empty input or text returns HTTP 400."""
        response = self.client.post("/v1/audio/speech", json={"input": "   "})
        self.assertEqual(response.status_code, 400)

        response_tts = self.client.post("/tts", json={"text": ""})
        self.assertEqual(response_tts.status_code, 400)

    def test_speech_endpoint_unready_engine_returns_503(self):
        """Verify HTTP 503 is returned when TTS engine is not ready."""
        mock_engine = MagicMock(spec=ITextToSpeechEngine)
        mock_engine.is_ready.return_value = False

        with patch.object(tts_router, "get_tts_engine", return_value=mock_engine):
            response = self.client.post("/v1/audio/speech", json={"input": "Texto cualquiera"})
            self.assertEqual(response.status_code, 503)

    def test_speech_abort_endpoints(self):
        """Verify POST /v1/audio/speech/abort and POST /tts/abort invoke engine.abort."""
        mock_engine = MagicMock(spec=ITextToSpeechEngine)
        mock_engine.abort = AsyncMock()

        with patch.object(tts_router, "get_tts_engine", return_value=mock_engine):
            # 1. /v1/audio/speech/abort with JSON body
            resp1 = self.client.post(
                "/v1/audio/speech/abort",
                json={"generation_id": "gen_abort_001"},
            )
            self.assertEqual(resp1.status_code, 200)
            self.assertEqual(resp1.json(), {"status": "ok", "aborted": "gen_abort_001"})
            mock_engine.abort.assert_called_with("gen_abort_001")

            # 2. /tts/abort with query param
            resp2 = self.client.post("/tts/abort?generation_id=gen_abort_002")
            self.assertEqual(resp2.status_code, 200)
            self.assertEqual(resp2.json(), {"status": "ok", "aborted": "gen_abort_002"})
            mock_engine.abort.assert_called_with("gen_abort_002")

    def test_transcriptions_endpoint_with_wav_file(self):
        """Verify POST /v1/audio/transcriptions transcribes cleanly."""
        # Create a synthetic WAV file
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 8000)
        wav_bytes = wav_io.getvalue()

        mock_engine = MagicMock(spec=ISpeechToTextEngine)
        mock_engine.is_ready.return_value = True
        mock_engine.transcribe = AsyncMock(
            return_value=TranscriptionResult(text="Hola Rylai", confidence=0.98, duration_ms=500.0)
        )

        with patch.object(stt_router, "get_stt_engine", return_value=mock_engine):
            files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
            response = self.client.post("/v1/audio/transcriptions", files=files)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["text"], "Hola Rylai")

    def test_transcriptions_endpoint_unready_engine_returns_503(self):
        """Verify HTTP 503 is returned when STT engine is not loaded."""
        mock_engine = MagicMock(spec=ISpeechToTextEngine)
        mock_engine.is_ready.return_value = False

        with patch.object(stt_router, "get_stt_engine", return_value=mock_engine):
            files = {"file": ("test.wav", b"dummy_content", "audio/wav")}
            response = self.client.post("/v1/audio/transcriptions", files=files)
            self.assertEqual(response.status_code, 503)

    def test_transcriptions_empty_file_returns_400(self):
        """Verify empty audio upload returns HTTP 400."""
        files = {"file": ("empty.wav", b"", "audio/wav")}
        response = self.client.post("/v1/audio/transcriptions", files=files)
        self.assertEqual(response.status_code, 400)

    def test_stt_json_endpoint(self):
        """Verify POST /v1/audio/transcribe-json and legacy POST /stt accept base64 audio and return transcription."""
        pcm_bytes = b"\x00\x00" * 4000
        b64_audio = base64.b64encode(pcm_bytes).decode("ascii")

        mock_engine = MagicMock(spec=ISpeechToTextEngine)
        mock_engine.is_ready.return_value = True
        mock_engine.transcribe = AsyncMock(
            return_value=TranscriptionResult(text="Prueba JSON STT", confidence=1.0, duration_ms=250.0)
        )

        with patch.object(stt_router, "get_stt_engine", return_value=mock_engine):
            # 1. Canonical /v1/audio/transcribe-json
            response_v1 = self.client.post(
                "/v1/audio/transcribe-json",
                json={"audio_base64": b64_audio, "sample_rate": 16000},
            )
            self.assertEqual(response_v1.status_code, 200)
            data_v1 = response_v1.json()
            self.assertEqual(data_v1["status"], "ok")
            self.assertEqual(data_v1["text"], "Prueba JSON STT")
            self.assertTrue(data_v1["is_speech"])

            # 2. Backward-compatible legacy alias /stt
            response_legacy = self.client.post(
                "/stt",
                json={"audio_base64": b64_audio, "sample_rate": 16000},
            )
            self.assertEqual(response_legacy.status_code, 200)
            data_legacy = response_legacy.json()
            self.assertEqual(data_legacy["status"], "ok")
            self.assertEqual(data_legacy["text"], "Prueba JSON STT")
            self.assertTrue(data_legacy["is_speech"])

    def test_voices_endpoint_returns_piper(self):
        """Verify GET /v1/audio/voices and legacy GET /tts/voices expose Piper default voice."""
        # 1. Canonical /v1/audio/voices
        response_v1 = self.client.get("/v1/audio/voices")
        self.assertEqual(response_v1.status_code, 200)
        data_v1 = response_v1.json()
        self.assertEqual(data_v1["default_engine"], "piper")
        self.assertEqual(data_v1["default_voice"], "piper_f")
        voice_ids_v1 = [v["id"] for v in data_v1["voices"]]
        self.assertIn("piper_f", voice_ids_v1)

        # 2. Backward-compatible legacy alias /tts/voices
        response_legacy = self.client.get("/tts/voices")
        self.assertEqual(response_legacy.status_code, 200)
        data_legacy = response_legacy.json()
        self.assertEqual(data_legacy["default_engine"], "piper")
        self.assertEqual(data_legacy["default_voice"], "piper_f")
        self.assertEqual(data_legacy, data_v1)

    def test_openapi_versioning_and_deprecation_metadata(self):
        """Verify OpenAPI spec tags canonical /v1/audio routes and marks legacy routes deprecated."""
        schema = app.openapi()
        paths = schema.get("paths", {})

        # Canonical endpoints check
        canonical_expectations = {
            "/v1/audio/speech": "post",
            "/v1/audio/speech/abort": "post",
            "/v1/audio/voices": "get",
            "/v1/audio/transcriptions": "post",
            "/v1/audio/transcribe-json": "post",
        }
        for path, method in canonical_expectations.items():
            self.assertIn(path, paths, f"Canonical path {path} missing from OpenAPI schema")
            op = paths[path].get(method, {})
            self.assertFalse(op.get("deprecated", False), f"{path} should NOT be marked deprecated")
            self.assertIn("Audio V1", op.get("tags", []), f"{path} should be tagged 'Audio V1'")

        # Legacy endpoints check
        legacy_expectations = {
            "/tts": "post",
            "/tts/abort": "post",
            "/tts/voices": "get",
            "/stt": "post",
        }
        for path, method in legacy_expectations.items():
            self.assertIn(path, paths, f"Legacy path {path} missing from OpenAPI schema")
            op = paths[path].get(method, {})
            self.assertTrue(op.get("deprecated", False), f"Legacy path {path} MUST be marked deprecated")
            self.assertIn(
                "Legacy (Deprecated)",
                op.get("tags", []),
                f"Legacy path {path} MUST be tagged 'Legacy (Deprecated)'",
            )


if __name__ == "__main__":
    unittest.main()
