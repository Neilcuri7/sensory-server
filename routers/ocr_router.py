"""
OCR Router: Extracción de Texto y Detección de UI Grounding (RapidOCR / ONNX).
"""
import io
import time
import base64
import logging
from typing import Optional, List, Tuple

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from PIL import Image

logger = logging.getLogger("Aoi.Sensory.OCR")
router = APIRouter(tags=["OCR"])

ocr_engine = None


def get_ocr_engine():
    global ocr_engine
    if ocr_engine is not None:
        return ocr_engine

    try:
        from rapidocr_onnxruntime import RapidOCR
        ocr_engine = RapidOCR()
        logger.info("Motor RapidOCR inicializado.")
        return ocr_engine
    except Exception as e:
        logger.warning(f"RapidOCR no disponible: {e}")
        return None


class OCRBase64Request(BaseModel):
    image_b64: str = Field(..., description="Imagen en formato Base64 (PNG/JPEG)")
    min_confidence: float = Field(default=0.5, description="Umbral mínimo de confianza (0.0 a 1.0)")


class OCRBox(BaseModel):
    text: str
    confidence: float
    box: List[List[int]] = Field(description="Polígono 4 puntos [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]")
    box_rect: List[int] = Field(description="Rectángulo [xmin, ymin, xmax, ymax]")
    center: List[int] = Field(description="Centro del elemento [cx, cy]")
    center_permille: List[int] = Field(description="Centro en permil [0..1000] relativo al ancho/alto")


class OCRResponse(BaseModel):
    status: str = "ok"
    text: str
    elements: List[OCRBox]
    count: int
    image_size: Tuple[int, int]
    latency_ms: float


def _process_image_bytes(image_bytes: bytes, min_confidence: float = 0.5) -> OCRResponse:
    t0 = time.perf_counter()
    image = Image.open(io.BytesIO(image_bytes))
    width, height = image.size

    elements: List[OCRBox] = []
    text_lines: List[str] = []
    engine = get_ocr_engine()

    if engine is not None:
        results, elapse = engine(image_bytes)
        if results:
            for item in results:
                box_pts, text, score = item[0], item[1], float(item[2])
                if score < min_confidence or not text.strip():
                    continue

                xs = [int(p[0]) for p in box_pts]
                ys = [int(p[1]) for p in box_pts]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                cx = (xmin + xmax) // 2
                cy = (ymin + ymax) // 2

                cx_permil = int((cx / max(1, width)) * 1000)
                cy_permil = int((cy / max(1, height)) * 1000)

                elements.append(OCRBox(
                    text=text.strip(),
                    confidence=round(score, 3),
                    box=[[int(p[0]), int(p[1])] for p in box_pts],
                    box_rect=[xmin, ymin, xmax, ymax],
                    center=[cx, cy],
                    center_permille=[cx_permil, cy_permil]
                ))
                text_lines.append(text.strip())

    latency = (time.perf_counter() - t0) * 1000.0

    return OCRResponse(
        status="ok",
        text="\n".join(text_lines),
        elements=elements,
        count=len(elements),
        image_size=(width, height),
        latency_ms=round(latency, 2)
    )


@router.post("/v1/ocr/base64", response_model=OCRResponse)
async def ocr_base64(req: OCRBase64Request):
    try:
        raw_b64 = req.image_b64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(raw_b64)
        return _process_image_bytes(img_bytes, min_confidence=req.min_confidence)
    except Exception as e:
        logger.error(f"Error procesando OCR base64: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/v1/ocr/image", response_model=OCRResponse)
async def ocr_image(file: UploadFile = File(...), min_confidence: float = 0.5):
    try:
        img_bytes = await file.read()
        return _process_image_bytes(img_bytes, min_confidence=min_confidence)
    except Exception as e:
        logger.error(f"Error procesando OCR archivo: {e}")
        raise HTTPException(status_code=400, detail=str(e))
