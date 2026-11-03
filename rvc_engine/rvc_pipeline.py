"""RVC (Retrieval-based Voice Conversion) Engine for Sensory Server.

Supports lightweight CPU and CUDA execution with automatic fallback:
- On systems with PyTorch/CUDA (e.g. laptop with GTX 1060), converts synthesized WAVs
  into high-fidelity anime voices using pre-trained .pth models and Faiss .index files.
- On systems without PyTorch (e.g. current CPU PC), gracefully skips RVC and returns
  the high-quality native Piper audio without crashing.
"""
from __future__ import annotations

import glob
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("Sensory.RVC")

_RVC_LOCK = threading.Lock()
_LOADED_MODELS: dict[str, object] = {}


@dataclass
class RvcModelFiles:
    name: str
    pth_path: str
    index_path: Optional[str] = None


def discover_rvc_models(models_dir: Optional[str] = None) -> dict[str, RvcModelFiles]:
    """Scans the models/rvc directory and returns available voice models."""
    if models_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
        models_dir = str(base_dir / "models" / "rvc")

    discovered: dict[str, RvcModelFiles] = {}
    p = Path(models_dir)
    if not p.is_dir():
        return discovered

    for entry in p.iterdir():
        if entry.is_dir():
            pths = list(entry.glob("*.pth"))
            indexes = list(entry.glob("*.index"))
            if pths:
                discovered[entry.name] = RvcModelFiles(
                    name=entry.name,
                    pth_path=str(pths[0]),
                    index_path=str(indexes[0]) if indexes else None,
                )
    return discovered


def is_torch_available() -> bool:
    """Checks whether PyTorch is installed and available in the environment."""
    try:
        import torch  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def get_rvc_device() -> str:
    """Determines the best compute device: 'cuda:0' if NVIDIA GPU is present, else 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def apply_rvc_conversion(
    wav_bytes: bytes,
    rvc_voice: str,
    pitch_semitones: float = 0.0,
    index_rate: float = 0.75,
) -> tuple[bytes, dict[str, object]]:
    """Converts a WAV byte payload into the requested RVC voice.

    If PyTorch/RVC dependencies are not installed or the model is not found,
    logs a clean notice and returns the original WAV bytes untouched.
    """
    if not rvc_voice or rvc_voice.lower() in ("none", "off", "desactivado", "false"):
        return wav_bytes, {"rvc_applied": False, "reason": "rvc_disabled"}

    models = discover_rvc_models()
    if rvc_voice not in models:
        # Check case-insensitive match
        matched_key = None
        for k in models:
            if k.lower() == rvc_voice.lower():
                matched_key = k
                break
        if not matched_key:
            logger.warning(
                f"[RVC] Modelo '{rvc_voice}' no encontrado en models/rvc/. "
                f"Disponibles: {list(models.keys())}. Retornando audio base sin alterar."
            )
            return wav_bytes, {
                "rvc_applied": False,
                "reason": "model_not_found",
                "voice": rvc_voice,
            }
        model_info = models[matched_key]
    else:
        model_info = models[rvc_voice]

    if not is_torch_available():
        logger.info(
            f"[RVC] PyTorch no está instalado en este host (ejecutando en CPU local). "
            f"El selector '{rvc_voice}' está activo pero se devuelve el audio de Piper nativo. "
            f"Al clonar y ejecutar en la laptop con GPU/PyTorch, la voz se transformará automáticamente."
        )
        return wav_bytes, {
            "rvc_applied": False,
            "reason": "torch_not_installed",
            "voice": model_info.name,
        }

    # If PyTorch is installed (e.g. laptop with GTX 1060), invoke conversion under lock
    with _RVC_LOCK:
        try:
            device = get_rvc_device()
            logger.info(
                f"[RVC] Aplicando conversión '{model_info.name}' en dispositivo '{device}' "
                f"(pth='{model_info.pth_path}', index='{model_info.index_path}', f0_shift={pitch_semitones})..."
            )
            # Dynamic import of infer module if installed
            import io
            import soundfile as sf
            # Try importing rvc_python or custom inference hook
            try:
                from rvc_python.infer import RVCInference
                rvc = RVCInference(device=device)
                rvc.load_model(model_info.pth_path, index_path=model_info.index_path)
                data, sr = sf.read(io.BytesIO(wav_bytes))
                out_audio = rvc.infer_data(
                    data,
                    sr,
                    f0_up_key=int(pitch_semitones),
                    index_rate=index_rate,
                )
                out_buf = io.BytesIO()
                sf.write(out_buf, out_audio, sr, format="WAV")
                transformed = out_buf.getvalue()
                return transformed, {
                    "rvc_applied": True,
                    "voice": model_info.name,
                    "device": device,
                    "f0_shift": pitch_semitones,
                }
            except (ImportError, ModuleNotFoundError):
                logger.info(
                    f"[RVC] rvc_python no está presente en el venv. Retornando audio base."
                )
                return wav_bytes, {
                    "rvc_applied": False,
                    "reason": "rvc_python_not_installed",
                    "voice": model_info.name,
                }
        except Exception as exc:
            logger.error(f"[RVC] Error durante la inferencia de '{model_info.name}': {exc}")
            return wav_bytes, {
                "rvc_applied": False,
                "error": str(exc),
                "voice": model_info.name,
            }
