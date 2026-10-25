"""
🌸 Aoi Sensory Hub Server (Unified Perception & Audio Microservice).
Consolida STT (Sherpa/Whisper), TTS (Kokoro/Piper 100% Local), Visión (Qwen2-VL) y OCR (RapidOCR)
en un único proceso FastAPI de alta eficiencia y bajo consumo de memoria.
Puerto por defecto: 8888
"""
import os
import time
import uuid
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from routers.stt_router import router as stt_router
from routers.tts_router import router as tts_router
from routers.vision_router import router as vision_router
from routers.ocr_router import router as ocr_router

log_level_name = os.getenv("AOI_SENSORY_LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Aoi.SensoryHub")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware que registra latencia, método, ruta, IP y X-Request-ID en cada petición."""
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        start_time = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Request-ID"] = req_id

        # No saturar logs en requests periódicos de healthcheck salvo si hay error
        if request.url.path == "/health" and response.status_code == 200:
            logger.debug(f"[{req_id}] GET /health - 200 ({duration_ms:.1f}ms) [client={client_ip}]")
        else:
            logger.info(
                f"[{req_id}] {request.method} {request.url.path} - {response.status_code} "
                f"({duration_ms:.1f}ms) [client={client_ip}]"
            )
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🌸 Aoi Sensory Hub iniciando en puerto 8888...")

    # Pre-calentar modelos ONNX en memoria para eliminar latencia en el primer turno
    def _warmup_models():
        from routers.stt_router import get_recognizer
        from routers.tts_router import get_kokoro_voice, get_piper_voice
        try:
            get_recognizer()
            get_kokoro_voice()
            get_piper_voice()
            logger.info("⚡ Modelos STT y TTS pre-cargados exitosamente en RAM.")
        except Exception as e:
            logger.warning(f"Aviso durante pre-carga de modelos: {e}")

    asyncio.create_task(asyncio.to_thread(_warmup_models))
    yield
    logger.info("🛑 Deteniendo Aoi Sensory Hub...")


app = FastAPI(
    title="🌸 Aoi Sensory Hub",
    version="3.0.0",
    description="Servidor unificado de percepción sensorial: Voz, Oídos, Visión y OCR para Aoi Desktop Companion.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

# Montar todos los routers sensoriales
app.include_router(stt_router)
app.include_router(tts_router)
app.include_router(vision_router)
app.include_router(ocr_router)


@app.get("/health")
async def health_check():
    import routers.stt_router as stt_mod
    import routers.tts_router as tts_mod

    stt_loaded = getattr(stt_mod, "recognizer", None) is not None
    kokoro_loaded = getattr(tts_mod, "kokoro_voice", None) is not None
    piper_loaded = getattr(tts_mod, "piper_voice", None) is not None

    if stt_loaded and kokoro_loaded and piper_loaded:
        health_status = "healthy"
    elif stt_loaded or kokoro_loaded or piper_loaded:
        health_status = "degraded"
    else:
        health_status = "down"

    return {
        "status": health_status,
        "service": "aoi-sensory-hub",
        "port": 8888,
        "models": {
            "stt": stt_loaded,
            "kokoro": kokoro_loaded,
            "piper": piper_loaded
        },
        "modules": {
            "stt": "Sherpa-ONNX / Whisper-Base",
            "tts": "Kokoro ONNX / Piper ONNX (100% Local Female Voices)",
            "vision": "RapidOCR ONNX UI Grounding",
            "ocr": "RapidOCR ONNX"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("AOI_SENSORY_PORT", "8888"))
    host = os.getenv("AOI_SENSORY_HOST", "0.0.0.0")
    logger.info(f"Iniciando Aoi Sensory Hub en {host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=False)
