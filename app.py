"""
🌸 Aoi Sensory Hub Server (Unified Perception & Audio Microservice).
Consolida STT (Sherpa/Whisper), TTS (Piper/Kokoro/Edge), Visión (Qwen2-VL) y OCR (RapidOCR)
en un único proceso FastAPI de alta eficiencia y bajo consumo de memoria.
Puerto por defecto: 8888
"""
import os
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routers.stt_router import router as stt_router
from routers.tts_router import router as tts_router
from routers.vision_router import router as vision_router
from routers.ocr_router import router as ocr_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Aoi.SensoryHub")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🌸 Aoi Sensory Hub iniciado exitosamente en puerto 8888.")
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

# Montar todos los routers sensoriales
app.include_router(stt_router)
app.include_router(tts_router)
app.include_router(vision_router)
app.include_router(ocr_router)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "aoi-sensory-hub",
        "port": 8888,
        "modules": {
            "stt": "Sherpa-ONNX / Whisper-Base",
            "tts": "Piper ONNX / Kokoro / Edge-TTS",
            "vision": "Qwen2-VL",
            "ocr": "RapidOCR ONNX"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("AOI_SENSORY_PORT", "8888"))
    host = os.getenv("AOI_SENSORY_HOST", "0.0.0.0")
    logger.info(f"Iniciando Aoi Sensory Hub en {host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=False)
