"""Infer pack neural modules for RVC v1/v2 synthesis."""
try:
    from .models import (
        SynthesizerTrnMs256NSFsid,
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid_nono,
    )
    __all__ = [
        "SynthesizerTrnMs256NSFsid",
        "SynthesizerTrnMs768NSFsid",
        "SynthesizerTrnMs256NSFsid_nono",
        "SynthesizerTrnMs768NSFsid_nono",
    ]
except ImportError:
    __all__ = []
