"""
Sensory Server (Unified Perception & Audio Microservice).
Consolida STT (Sherpa/Whisper), TTS (Piper ONNX 100% Local), Visión (Qwen2-VL) y OCR (RapidOCR)
en un único proceso FastAPI de alta eficiencia y bajo consumo de memoria.
Puerto por defecto: 8888
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from adapters.stt.sherpa_whisper import SherpaWhisperAdapter
from adapters.tts.piper_engine import PiperEngine
from routers.ocr_router import router as ocr_router
from routers.stt_router import get_stt_engine, router as stt_router, set_stt_engine
from routers.tts_router import get_tts_engine, router as tts_router, set_tts_engine
from routers.vision_router import router as vision_router

log_level_name = os.getenv("SENSORY_LOG_LEVEL", os.getenv("AOI_SENSORY_LOG_LEVEL", "INFO")).upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SensoryHub")


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
    logger.info("Sensory Server iniciando en puerto 8888...")

    tts_engine = PiperEngine()
    stt_engine = SherpaWhisperAdapter()

    app.state.tts_engine = tts_engine
    app.state.stt_engine = stt_engine
    set_tts_engine(tts_engine)
    set_stt_engine(stt_engine)

    logger.info(
        f"Motores cargados: TTS (Piper) ready={tts_engine.is_ready()}, "
        f"STT (Sherpa/Whisper) ready={stt_engine.is_ready()}"
    )
    yield
    logger.info("Deteniendo Sensory Server...")


app = FastAPI(
    title="Sensory Server",
    version="3.0.0",
    description="Servidor unificado de percepción sensorial: Voz, Oídos, Visión y OCR.",
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
async def health_check(request: Request):
    stt_engine = get_stt_engine(request.app)
    tts_engine = get_tts_engine(request.app)

    stt_ready = bool(stt_engine and stt_engine.is_ready())
    tts_ready = bool(tts_engine and tts_engine.is_ready())
    status = "ok" if (stt_ready and tts_ready) else "degraded"

    return {
        "status": status,
        "service": "sensory-hub",
        "active_engine": "piper",
        "stt_ready": stt_ready,
        "tts_ready": tts_ready,
        "port": 8888,
        "models": {
            "stt": stt_ready,
            "piper": tts_ready,
            "tts": tts_ready,
        },
        "modules": {
            "stt": "Sherpa-ONNX / Whisper-Base",
            "tts": "Piper ONNX (100% Local Female Voice)",
            "vision": "RapidOCR ONNX UI Grounding",
            "ocr": "RapidOCR ONNX"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("SENSORY_PORT", os.getenv("AOI_SENSORY_PORT", "8888")))
    host = os.getenv("SENSORY_HOST", os.getenv("AOI_SENSORY_HOST", "0.0.0.0"))
    logger.info(f"Iniciando Sensory Server en {host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=False)
