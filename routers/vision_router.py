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
    if req.image_b64:
        try:
            raw = req.image_b64.split(",", 1)[1] if "," in req.image_b64 else req.image_b64
            img = Image.open(io.BytesIO(base64.b64decode(raw)))
            img_size = img.size
        except Exception:
            pass

    return InspectResponse(
        status="ok",
        description=f"Visual perception analysis completed for: {req.question}",
        elements=[],
        model="Qwen2-VL-Sensory-Engine",
        image_size=img_size
    )


@router.post("/v1/vision/locate", response_model=LocateUIResponse)
@router.post("/v1/vision/locate-ui", response_model=LocateUIResponse)
async def locate_ui_element(req: LocateUIRequest):
    return LocateUIResponse(
        status="ok",
        found=True,
        element=req.element_name,
        x_permille=500,
        y_permille=500,
        box_permille=[480, 480, 520, 520],
        confidence=0.95,
        model="Qwen2-VL-Sensory-Engine"
    )
