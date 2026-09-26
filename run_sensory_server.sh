#!/usr/bin/env bash
# ==============================================================================
# 🌸 Aoi Sensory Hub Server (Puerto 8888)
# Unifica STT, TTS, Visión y OCR en 1 solo proceso
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "$DIR/../../aoi_2.5/.venv" ]; then
    source "$DIR/../../aoi_2.5/.venv/bin/activate"
fi

export AOI_SENSORY_PORT=${AOI_SENSORY_PORT:-8888}
export AOI_SENSORY_HOST=${AOI_SENSORY_HOST:-0.0.0.0}

echo "🌸 Iniciando Aoi Sensory Hub en $AOI_SENSORY_HOST:$AOI_SENSORY_PORT..."
exec python app.py
