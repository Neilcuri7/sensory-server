"""
Vision Router: Inspección Visual y Localización Multimodal (Qwen2-VL / Fallback).
"""
import io
import time
import base64
import logging
from typing import Optional, List, Dict, Any, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

logger = logging.getLogger("Aoi.Sensory.Vision")
router = APIRouter(tags=["Vision"])


class InspectRequest(BaseModel):
    image_b64: Optional[str] = Field(default=None, description="Imagen en formato Base64.")
    question: Optional[str] = Field(
        default="Describe the active window and main UI elements visible on the screen.",
        description="Pregunta o consulta sobre la imagen."
    )
    max_tokens: Optional[int] = Field(default=512)
    temperature: Optional[float] = Field(default=0.2)


class LocateUIRequest(BaseModel):
    image_b64: Optional[str] = Field(default=None, description="Imagen en Base64.")
    element_name: str = Field(..., description="Elemento visual o botón a localizar.")
    context: Optional[str] = Field(default=None)


class InspectResponse(BaseModel):
    status: str = "ok"
    description: str
    elements: List[Dict[str, Any]] = Field(default_factory=list)
    model: str
    image_size: Tuple[int, int]


class LocateUIResponse(BaseModel):
    status: str = "ok"
    found: bool
    element: str
    x_permille: int
    y_permille: int
    box_permille: List[int] = Field(default_factory=list)
    confidence: float = 1.0
    model: str


@router.post("/v1/vision/inspect", response_model=InspectResponse)
async def inspect_image(req: InspectRequest):
    img_size = (1920, 1080)
    detected_texts: List[str] = []
    elements: List[Dict[str, Any]] = []

    if req.image_b64:
        try:
            raw = req.image_b64.split(",", 1)[1] if "," in req.image_b64 else req.image_b64
            img_bytes = base64.b64decode(raw)
            img = Image.open(io.BytesIO(img_bytes))
            img_size = img.size
            width, height = img_size

            # Ejecutar RapidOCR para extraer todos los textos y posiciones de la pantalla
            try:
                from routers.ocr_router import get_ocr_engine
                engine = get_ocr_engine()
                if engine is not None:
                    results, _ = engine(img_bytes)
                    if results:
                        for item in results:
                            box_pts, text, score = item[0], item[1], float(item[2])
                            if score < 0.4 or not text.strip():
                                continue
                            
                            xs = [int(p[0]) for p in box_pts]
                            ys = [int(p[1]) for p in box_pts]
                            xmin, xmax = min(xs), max(xs)
                            ymin, ymax = min(ys), max(ys)
                            cx = (xmin + xmax) // 2
                            cy = (ymin + ymax) // 2
                            
                            cx_permil = int((cx / max(1, width)) * 1000)
                            cy_permil = int((cy / max(1, height)) * 1000)

                            detected_texts.append(text.strip())
                            elements.append({
                                "text": text.strip(),
                                "confidence": round(score, 3),
                                "center": [cx, cy],
                                "center_permille": [cx_permil, cy_permil],
                                "box_rect": [xmin, ymin, xmax, ymax]
                            })
            except Exception as ocr_err:
                logger.warning(f"Error procesando OCR en inspect_image: {ocr_err}")

        except Exception as e:
            logger.error(f"Error decodificando imagen en inspect_image: {e}")

    if detected_texts:
        formatted_summary = (
            f"Análisis visual y OCR de la pantalla ({img_size[0]}x{img_size[1]}):\n"
            f"Se detectaron {len(detected_texts)} elementos y fragmentos de texto visibles:\n"
            + "\n".join(f"- {t}" for t in detected_texts[:40])
        )
    else:
        formatted_summary = f"No se detectó texto claro en la pantalla o la imagen está vacía ({img_size[0]}x{img_size[1]})."

    return InspectResponse(
        status="ok",
        description=formatted_summary,
        elements=elements,
        model="RapidOCR-ONNX-Perception-Engine",
        image_size=img_size
    )


@router.post("/v1/vision/locate", response_model=LocateUIResponse)
@router.post("/v1/vision/locate-ui", response_model=LocateUIResponse)
async def locate_ui_element(req: LocateUIRequest):
    target = req.element_name.strip().lower()
    
    if req.image_b64:
        try:
            raw = req.image_b64.split(",", 1)[1] if "," in req.image_b64 else req.image_b64
            img_bytes = base64.b64decode(raw)
            img = Image.open(io.BytesIO(img_bytes))
            width, height = img.size

            from routers.ocr_router import get_ocr_engine
            engine = get_ocr_engine()
            if engine is not None:
                results, _ = engine(img_bytes)
                if results:
                    best_match = None
                    for item in results:
                        box_pts, text, score = item[0], item[1], float(item[2])
                        text_clean = text.strip().lower()
                        
                        if target in text_clean or text_clean in target:
                            xs = [int(p[0]) for p in box_pts]
                            ys = [int(p[1]) for p in box_pts]
                            xmin, xmax = min(xs), max(xs)
                            ymin, ymax = min(ys), max(ys)
                            cx = (xmin + xmax) // 2
                            cy = (ymin + ymax) // 2

                            best_match = {
                                "found": True,
                                "element": text.strip(),
                                "x_permille": int((cx / max(1, width)) * 1000),
                                "y_permille": int((cy / max(1, height)) * 1000),
                                "box_permille": [
                                    int((xmin / max(1, width)) * 1000),
                                    int((ymin / max(1, height)) * 1000),
                                    int((xmax / max(1, width)) * 1000),
                                    int((ymax / max(1, height)) * 1000)
                                ],
                                "confidence": round(score, 3),
                                "model": "RapidOCR-ONNX-Perception-Engine"
                            }
                            break
                    if best_match:
                        return LocateUIResponse(status="ok", **best_match)
        except Exception as e:
            logger.error(f"Error en locate_ui_element: {e}")

    return LocateUIResponse(
        status="ok",
        found=False,
        element=req.element_name,
        x_permille=500,
        y_permille=500,
        box_permille=[0, 0, 0, 0],
        confidence=0.0,
        model="RapidOCR-ONNX-Perception-Engine"
    )
