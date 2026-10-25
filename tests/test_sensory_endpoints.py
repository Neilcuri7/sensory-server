"""
Tests unitarios para los endpoints de Aoi Sensory Hub (FastAPI).
Verifica:
- Middleware de logging y propagación del header X-Request-ID.
- Endpoint /health reportando estado de modelos y degradación.
- Endpoint /v1/audio/transcriptions devolviendo HTTP 503 si el reconocedor no está disponible.
"""
import io
import wave
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app import app
import routers.stt_router as stt_router


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check_endpoint(client):
    """Verifica la respuesta del health check con estructura de modelos y estado."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ("healthy", "degraded", "down")
    assert "models" in data
    assert "stt" in data["models"]
    assert "kokoro" in data["models"]
    assert "piper" in data["models"]
    # Verifica que el middleware haya inyectado X-Request-ID
    assert "x-request-id" in response.headers or "X-Request-ID" in response.headers


def test_request_logging_middleware_preserves_custom_request_id(client):
    """Verifica que si el cliente envía X-Request-ID, el middleware lo devuelva inalterado."""
    custom_id = "test-req-12345"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.headers.get("X-Request-ID") == custom_id or response.headers.get("x-request-id") == custom_id


def test_stt_returns_503_when_recognizer_is_none(client):
    """Verifica que si el reconocedor Whisper no carga, devuelva HTTP 503 en lugar de '[STT Service Ready]'."""
    # Generar WAV sintético de 0.5s
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 8000)
    wav_bytes = wav_io.getvalue()

    with patch.object(stt_router, "get_recognizer", return_value=None):
        files = {"file": ("test.wav", wav_bytes, "audio/wav")}
        response = client.post("/v1/audio/transcriptions", files=files, data={"model": "whisper-base"})
        assert response.status_code == 503
        data = response.json()
        assert "STT model not loaded" in data.get("detail", "")
