"""RVC (Retrieval-based Voice Conversion) Engine for Sensory Server.

Neural RVC v2 Inference (PyTorch + CUDA/CPU + ONNX Runtime / TorchAudio + FAISS)
Optimized for NVIDIA GeForce GTX 1060 (6 GB VRAM) on WSL2 / Linux as well as CPU inference.
When neural inference is not active, cleanly passes through base TTS without distortion.
"""
from __future__ import annotations

import io
import math
import logging
import os
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import numpy as np
import scipy.signal

logger = logging.getLogger("Sensory.RVC")

_RVC_LOCK = threading.Lock()
_LOADED_MODELS: dict[str, Any] = {}


@dataclass
class RvcModelFiles:
    name: str
    pth_path: str
    index_path: Optional[str] = None


def get_wav_info(wav_bytes: bytes) -> tuple[int, int]:
    """Returns (sample_rate, num_frames) directly from WAV bytes header."""
    if not wav_bytes or len(wav_bytes) < 44:
        return (22050, 0)
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            return (wf.getframerate(), wf.getnframes())
    except Exception:
        pass
    try:
        if wav_bytes[:4] == b"RIFF" and wav_bytes[8:12] == b"WAVE":
            sr = int.from_bytes(wav_bytes[24:28], "little")
            data_idx = wav_bytes.find(b"data")
            if data_idx != -1:
                data_size = int.from_bytes(wav_bytes[data_idx + 4 : data_idx + 8], "little")
                channels = int.from_bytes(wav_bytes[22:24], "little") or 1
                bits = int.from_bytes(wav_bytes[34:36], "little") or 16
                bpf = channels * (bits // 8)
                frames = data_size // bpf if bpf > 0 else 0
                return (sr, frames)
    except Exception:
        pass
    return (22050, 0)


def _read_wav_samples(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Reads WAV bytes into a 1D float32 numpy array normalized to [-1.0, 1.0] and sample rate."""
    with io.BytesIO(wav_bytes) as bio:
        with wave.open(bio, "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            sampwidth = wf.getsampwidth()
            n_channels = wf.getnchannels()
            raw = wf.readframes(n_frames)

            if sampwidth == 2:
                int_data = np.frombuffer(raw, dtype=np.int16)
                float_data = int_data.astype(np.float32) / 32767.0
            elif sampwidth == 4:
                int_data = np.frombuffer(raw, dtype=np.int32)
                float_data = int_data.astype(np.float32) / 2147483647.0
            elif sampwidth == 1:
                int_data = np.frombuffer(raw, dtype=np.uint8)
                float_data = (int_data.astype(np.float32) - 128.0) / 128.0
            else:
                int_data = np.frombuffer(raw, dtype=np.int16)
                float_data = int_data.astype(np.float32) / 32767.0

            if n_channels > 1:
                float_data = float_data.reshape(-1, n_channels).mean(axis=1)

            return float_data, sr


def _write_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Converts 1D float32 samples to standard 16-bit PCM WAV bytes with peak soft limiting."""
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.98, neginf=-0.98)
    if len(samples) > 0:
        peak = float(np.max(np.abs(samples)))
        if peak > 0.98:
            samples = samples * (0.98 / peak)
    clipped = np.clip(samples, -1.0, 1.0)
    int_data = (clipped * 32767.0).astype(np.int16)

    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_data.tobytes())
    return out.getvalue()


def discover_rvc_models(models_dir: Optional[str] = None) -> dict[str, RvcModelFiles]:
    """Scans the models/rvc directory and returns available voice models."""
    if models_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
        models_dir = str(base_dir / "models" / "rvc")

    discovered: dict[str, RvcModelFiles] = {}
    p = Path(models_dir)
    if not p.is_dir():
        return discovered

    for entry in sorted(p.iterdir()):
        if entry.is_dir():
            pths = sorted(entry.glob("*.pth"))
            indexes = sorted(entry.glob("*.index"))
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


def _get_models_dir() -> Path:
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "models"


def _find_contentvec_onnx() -> Optional[str]:
    """Locates contentvec-f.onnx or vec-768-layer-12.onnx on disk."""
    models_dir = _get_models_dir()
    candidates = [
        models_dir / "rvc" / "contentvec-f.onnx",
        models_dir / "rvc" / "vec-768-layer-12.onnx",
        models_dir / "contentvec" / "contentvec-f.onnx",
        models_dir / "contentvec-f.onnx",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def _find_rmvpe_onnx() -> Optional[str]:
    """Locates rmvpe.onnx or rmvpe_20231006.onnx on disk."""
    models_dir = _get_models_dir()
    candidates = [
        models_dir / "rvc" / "rmvpe.onnx",
        models_dir / "rvc" / "rmvpe_20231006.onnx",
        models_dir / "rmvpe" / "rmvpe.onnx",
        models_dir / "rmvpe.onnx",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def get_rvc_engine_status() -> dict[str, object]:
    """Reports current operational status of the RVC engine."""
    torch_avail = is_torch_available()
    cuda_avail = False
    device = "cpu"
    if torch_avail:
        try:
            import torch
            cuda_avail = torch.cuda.is_available()
            device = "cuda:0" if cuda_avail else "cpu"
        except Exception:
            pass

    has_contentvec = _find_contentvec_onnx() is not None
    has_rmvpe = _find_rmvpe_onnx() is not None
    has_torchaudio = False
    if torch_avail:
        try:
            import torchaudio
            has_torchaudio = hasattr(torchaudio.pipelines, "HUBERT_BASE")
        except Exception:
            pass

    neural_ready = torch_avail and (has_contentvec or has_torchaudio)
    models = discover_rvc_models()
    available_voices = list(models.keys())

    if neural_ready:
        mode = "neural"
    else:
        mode = "disabled"

    return {
        "neural_ready": neural_ready,
        "torch_available": torch_avail,
        "cuda_available": cuda_avail,
        "device": device,
        "mode": mode,
        "has_contentvec_onnx": has_contentvec,
        "has_rmvpe_onnx": has_rmvpe,
        "has_torchaudio_hubert": has_torchaudio,
        "available_voices": available_voices,
    }


# ==============================================================================
# Neural RVC v2 Inference (PyTorch + CUDA / CPU)
# ==============================================================================

def _extract_f0_autocorr(
    audio_16k: np.ndarray,
    sr: int = 16000,
    hop_length: int = 160,
    fmin: float = 50.0,
    fmax: float = 1100.0,
) -> np.ndarray:
    """Fast, robust autocorrelation pitch tracker running at 100 fps (10 ms / frame)."""
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    n_frames = max(1, len(audio_16k) // hop_length)
    f0 = np.zeros(n_frames, dtype=np.float32)
    win_len = 640
    pad_len = win_len // 2
    padded = np.pad(audio_16k, (pad_len, pad_len), mode="reflect")

    for i in range(n_frames):
        start = i * hop_length
        frame = padded[start : start + win_len]
        energy = np.sum(frame**2)
        if energy < 1e-4:
            continue
        corr = np.correlate(frame, frame, mode="full")[win_len - 1 :]
        corr_search = corr[min_lag:max_lag]
        if len(corr_search) == 0:
            continue
        best_lag = min_lag + int(np.argmax(corr_search))
        peak_val = corr[best_lag] / (energy + 1e-6)
        if peak_val > 0.35:
            f0[i] = float(sr / best_lag)
    return f0


def _get_ort_providers() -> list[str]:
    """Returns viable ONNX Runtime execution providers without missing-library spam."""
    import ctypes
    import onnxruntime as ort

    if "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            ctypes.CDLL("libcublasLt.so.11")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except OSError:
            pass
    return ["CPUExecutionProvider"]


def _extract_features_onnx(
    onnx_path: str,
    audio_16k: np.ndarray,
    is_v2: bool = True,
) -> np.ndarray:
    """Extracts 768-dim (v2) or 256-dim (v1) features from ContentVec ONNX."""
    import onnxruntime as ort

    sess_key = f"ort_session_{onnx_path}"
    if sess_key not in _LOADED_MODELS:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        providers = _get_ort_providers()
        _LOADED_MODELS[sess_key] = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    session = _LOADED_MODELS[sess_key]

    inp = session.get_inputs()[0]
    inp_name = inp.name
    inp_shape = inp.shape

    if len(inp_shape) == 3:
        inp_data = audio_16k.reshape(1, 1, -1).astype(np.float32)
    else:
        inp_data = audio_16k.reshape(1, -1).astype(np.float32)

    target_dim = 768 if is_v2 else 256
    output_names = [o.name for o in session.get_outputs()]
    if is_v2 and "unit12" in output_names:
        outputs = session.run(["unit12"], {inp_name: inp_data})
        feat = outputs[0]
    elif not is_v2 and "units9" in output_names:
        outputs = session.run(["units9"], {inp_name: inp_data})
        feat = outputs[0]
    else:
        outputs = session.run(None, {inp_name: inp_data})
        feat = outputs[0]
        for cand in outputs:
            if cand.ndim == 3 and (cand.shape[-1] == target_dim or cand.shape[1] == target_dim):
                feat = cand
                break

    if feat.ndim == 3 and feat.shape[-1] != target_dim and feat.shape[1] == target_dim:
        feat = np.transpose(feat, (0, 2, 1))
    return feat


def _run_neural_rvc_inference(
    wav_bytes: bytes,
    model_files: RvcModelFiles,
    pitch_semitones: float = 0.0,
    index_rate: float = 0.75,
) -> tuple[bytes, dict[str, object]]:
    """Executes full neural RVC v2 inference via PyTorch on CUDA GTX 1060 or CPU."""
    import torch
    import torch.nn.functional as F

    samples, sr = _read_wav_samples(wav_bytes)
    # Resample input audio to 16 kHz for feature & pitch extraction
    if sr != 16000:
        gcd = math.gcd(sr, 16000)
        audio_16k = scipy.signal.resample_poly(samples, 16000 // gcd, sr // gcd).astype(np.float32)
    else:
        audio_16k = samples.astype(np.float32)

    device_str = get_rvc_device()
    device = torch.device(device_str)
    is_half = (device.type == "cuda")

    # 1. Load and cache RVC v2 Synthesizer model
    net_key = f"rvc_net_{model_files.pth_path}"
    if net_key not in _LOADED_MODELS:
        logger.info(f"[RVC] Loading checkpoint '{model_files.pth_path}' onto {device_str}...")
        cpt = torch.load(model_files.pth_path, map_location="cpu", weights_only=False)
        f0_val = cpt.get("f0", 1)
        is_f0 = bool(f0_val == 1 or f0_val is True or f0_val == "1")
        config = cpt.get("config", [])
        tgt_sr = cpt.get("sr", config[-1] if config else 40000)
        is_v2 = (cpt.get("version") == "v2" or (config and len(config) > 4 and config[4] == 768))

        from .infer_pack.models import (
            SynthesizerTrnMs768NSFsid,
            SynthesizerTrnMs768NSFsid_nono,
            SynthesizerTrnMs256NSFsid,
            SynthesizerTrnMs256NSFsid_nono,
        )

        if is_v2:
            model_cls = SynthesizerTrnMs768NSFsid if is_f0 else SynthesizerTrnMs768NSFsid_nono
        else:
            model_cls = SynthesizerTrnMs256NSFsid if is_f0 else SynthesizerTrnMs256NSFsid_nono

        net = model_cls(*config, is_half=is_half)
        net.load_state_dict(cpt["weight"], strict=False)
        net.eval()
        net = net.to(device)
        if is_half:
            net = net.half()
        _LOADED_MODELS[net_key] = (net, tgt_sr, is_f0, is_v2)

    net, tgt_sr, is_f0, is_v2 = _LOADED_MODELS[net_key]

    # 2. Extract 768-dim features (ContentVec ONNX or torchaudio HuBERT Base)
    contentvec_path = _find_contentvec_onnx()
    if contentvec_path:
        feat_np = _extract_features_onnx(contentvec_path, audio_16k, is_v2=is_v2)
        feats = torch.from_numpy(feat_np).to(device)
    else:
        import torchaudio
        bundle = torchaudio.pipelines.HUBERT_BASE
        hubert_key = f"hubert_model_{device_str}"
        if hubert_key not in _LOADED_MODELS:
            hubert = bundle.get_model().to(device)
            hubert.eval()
            if is_half:
                hubert = hubert.half()
            _LOADED_MODELS[hubert_key] = hubert
        hubert = _LOADED_MODELS[hubert_key]

        wav_t = torch.from_numpy(audio_16k).unsqueeze(0).to(device)
        if is_half:
            wav_t = wav_t.half()
        with torch.no_grad():
            feats_list, _ = hubert.extract_features(wav_t)
            feats = feats_list[-1]  # Layer 12, shape (1, T, 768)

    # 3. FAISS feature retrieval (.index)
    if model_files.index_path and index_rate > 0.0:
        try:
            import faiss
            idx_key = f"faiss_idx_{model_files.index_path}"
            if idx_key not in _LOADED_MODELS:
                index = faiss.read_index(model_files.index_path)
                if hasattr(index, "make_direct_map"):
                    try:
                        index.make_direct_map()
                    except Exception:
                        pass
                big_npy = index.reconstruct_n(0, index.ntotal)
                _LOADED_MODELS[idx_key] = (index, big_npy)
            index, big_npy = _LOADED_MODELS[idx_key]

            feats_np = feats[0].cpu().numpy().astype(np.float32)
            score, ix = index.search(feats_np, k=8)
            weight = np.square(1.0 / (score + 1e-6))
            weight /= weight.sum(axis=1, keepdims=True)
            retrieved = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1).astype(np.float32)
            feats = (
                torch.from_numpy(retrieved).unsqueeze(0).to(device=device, dtype=feats.dtype) * index_rate
                + (1.0 - index_rate) * feats
            )
        except Exception as e:
            logger.debug(f"[RVC] FAISS retrieval bypassed: {e}")

    # 4. Pitch extraction (RMVPE ONNX or Autocorrelation)
    rmvpe_path = _find_rmvpe_onnx()
    if is_f0:
        if rmvpe_path:
            import onnxruntime as ort
            sess_key = f"ort_rmvpe_{rmvpe_path}"
            if sess_key not in _LOADED_MODELS:
                so = ort.SessionOptions()
                so.log_severity_level = 3
                providers = _get_ort_providers()
                _LOADED_MODELS[sess_key] = ort.InferenceSession(rmvpe_path, sess_options=so, providers=providers)
            rmvpe_sess = _LOADED_MODELS[sess_key]
            out = rmvpe_sess.run(
                ["pitchf"],
                {
                    "waveform": audio_16k.reshape(1, -1).astype(np.float32),
                    "threshold": np.array([0.3], dtype=np.float32),
                },
            )
            pitchf = out[0].squeeze()
        else:
            pitchf = _extract_f0_autocorr(audio_16k, sr=16000, hop_length=160)

        # Shift pitch by semitones
        if abs(pitch_semitones) >= 0.05:
            pitchf = pitchf * (2.0 ** (pitch_semitones / 12.0))

        # Convert pitchf to coarse mel bins (1..255)
        f0_mel_min = 1127.0 * np.log(1.0 + 50.0 / 700.0)
        f0_mel_max = 1127.0 * np.log(1.0 + 1100.0 / 700.0)
        f0_mel = 1127.0 * np.log(1.0 + np.maximum(pitchf, 0.0) / 700.0)
        f0_coarse = (f0_mel - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0
        f0_coarse = np.clip(np.rint(f0_coarse), 1, 255).astype(np.int64)
        f0_coarse[pitchf <= 0] = 1
    else:
        f0_coarse = None
        pitchf = None

    # 5. Interpolate features 2x (from 20ms to 10ms frame rate)
    feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

    # Align sequence lengths
    if is_f0 and f0_coarse is not None:
        t_len = min(feats.shape[1], len(f0_coarse))
        feats = feats[:, :t_len, :]
        pitch_t = torch.tensor(f0_coarse[:t_len], dtype=torch.long, device=device).unsqueeze(0)
        pitchf_t = torch.tensor(pitchf[:t_len], dtype=torch.float32, device=device).unsqueeze(0)
        if is_half:
            pitchf_t = pitchf_t.half()
    else:
        t_len = feats.shape[1]
        pitch_t = None
        pitchf_t = None

    p_len_t = torch.tensor([t_len], dtype=torch.long, device=device)
    sid_t = torch.tensor([0], dtype=torch.long, device=device)
    if is_half:
        feats = feats.half()

    # 6. Synthesizer Inference
    with torch.no_grad():
        if is_f0:
            out = net.infer(feats, p_len_t, pitch_t, pitchf_t, sid_t)
        else:
            out = net.infer(feats, p_len_t, sid_t)

    out_audio = out[0][0, 0].cpu().float().numpy()
    out_wav = _write_wav_bytes(out_audio, sample_rate=tgt_sr)
    return out_wav, {
        "rvc_applied": True,
        "voice": model_files.name,
        "mode": "neural",
        "device": device_str,
        "sr": tgt_sr,
    }


# ==============================================================================
# Unified Entry Point: apply_rvc_conversion
# ==============================================================================

def apply_rvc_conversion(
    wav_bytes: bytes,
    rvc_voice: str,
    pitch_semitones: float = 0.0,
    index_rate: float = 0.75,
) -> tuple[bytes, dict[str, object]]:
    """Converts a WAV byte payload into the requested RVC voice.

    Uses Neural RVC v2 inference (PyTorch + CUDA/CPU + ONNX Runtime / TorchAudio + FAISS).
    If neural models or PyTorch are not available or inference fails, cleanly passes through
    the base audio without distortion.
    """
    if not rvc_voice or rvc_voice.strip().lower() in ("none", "off", "desactivado", "disabled", "false", ""):
        return wav_bytes, {"rvc_applied": False, "reason": "rvc_disabled"}

    models = discover_rvc_models()
    matched_model: Optional[RvcModelFiles] = None

    # Exact or case-insensitive match
    if rvc_voice in models:
        matched_model = models[rvc_voice]
    else:
        for k, v in models.items():
            if k.lower() == rvc_voice.strip().lower():
                matched_model = v
                break

    if not matched_model:
        logger.warning(
            f"[RVC] Voice model '{rvc_voice}' not found in models/rvc/. "
            f"Available: {list(models.keys())}. Returning base audio."
        )
        return wav_bytes, {
            "rvc_applied": False,
            "reason": "model_not_found",
            "voice": rvc_voice,
        }

    status = get_rvc_engine_status()

    # Neural RVC inference
    if status["neural_ready"]:
        with _RVC_LOCK:
            try:
                return _run_neural_rvc_inference(
                    wav_bytes=wav_bytes,
                    model_files=matched_model,
                    pitch_semitones=pitch_semitones,
                    index_rate=index_rate,
                )
            except Exception as exc:
                logger.error(f"[RVC] Neural inference error for '{matched_model.name}': {exc}")
                return wav_bytes, {
                    "rvc_applied": False,
                    "error": str(exc),
                    "voice": matched_model.name,
                }

    logger.warning(
        f"[RVC] Neural RVC inference not ready (missing torch or feature extractors). "
        f"Returning clean base audio."
    )
    return wav_bytes, {
        "rvc_applied": False,
        "reason": "neural_not_ready",
        "voice": matched_model.name,
    }
