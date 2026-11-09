#!/usr/bin/env bash
# ==============================================================================
# Sensory Server: Script de Descarga de Modelos
# Descarga los pesos ONNX de Whisper Base (STT) y Piper Voice (TTS)
# Soporta niveles de aprovisionamiento: --tier [light|full|all]
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$DIR/models"

TIER="${SENSORY_TIER:-full}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier)
      TIER="${2:-}"
      shift 2
      ;;
    --tier=*)
      TIER="${1#*=}"
      shift 1
      ;;
    -h|--help)
      echo "Uso: $0 [--tier light|full|all]"
      echo "  --tier light: Whisper Base + Piper Sharvard (~185 MB)"
      echo "  --tier full:  Whisper Base + Piper Sharvard + Piper Davefx (~230 MB)"
      echo "  --tier all:   Whisper Base + Whisper Small + Piper Sharvard + Piper Davefx"
      exit 0
      ;;
    *)
      echo "Argumento desconocido: $1" >&2
      exit 1
      ;;
  esac
done

TIER="$(echo "$TIER" | tr '[:upper:]' '[:lower:]')"
case "$TIER" in
  light|full|all) ;;
  *)
    echo "ERROR: Nivel de aprovisionamiento no válido: '$TIER'. Use 'light', 'full' o 'all'." >&2
    exit 1
    ;;
esac

mkdir -p "$MODELS_DIR/piper" "$MODELS_DIR/sherpa-onnx-whisper-base"

echo "=========================================================="
echo "Descargando Modelos para Sensory Server (Tier: $TIER)..."
echo "STT: Sherpa-ONNX Whisper Base | TTS: Piper Standalone ONNX"
echo "=========================================================="

# 1. Modelo STT: Sherpa-ONNX Whisper Base Multilingüe (~140 MB)
WHISPER_DIR="$MODELS_DIR/sherpa-onnx-whisper-base"
if [ ! -f "$WHISPER_DIR/base-encoder.int8.onnx" ]; then
    echo "Descargando Whisper Base ONNX..."
    TAR_FILE="$MODELS_DIR/whisper-base.tar.bz2"
    curl -L --progress-bar -o "$TAR_FILE" "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.tar.bz2"
    echo "Extrayendo Whisper Base..."
    tar -xjf "$TAR_FILE" -C "$MODELS_DIR"
    rm -f "$TAR_FILE"
    echo "Whisper Base listo."
else
    echo "Whisper Base ya esta instalado."
fi

# 2. Modelo TTS: Piper Voice en Español (Sharvard Medium ~45 MB - Voz principal)
PIPER_DIR="$MODELS_DIR/piper"
if [ ! -f "$PIPER_DIR/es_ES-sharvard-medium.onnx" ]; then
    echo "Descargando Piper Voice (es_ES-sharvard-medium)..."
    curl -L --progress-bar -o "$PIPER_DIR/es_ES-sharvard-medium.onnx" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx"
    curl -L --progress-bar -o "$PIPER_DIR/es_ES-sharvard-medium.onnx.json" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx.json"
    echo "Piper Sharvard Voice listo."
else
    echo "Piper Sharvard Voice ya esta instalado."
fi

# 3. Modelo TTS Opcional: Piper Voice en Español (Davefx Medium ~45 MB - Voz expresiva alternativa)
if [ "$TIER" = "full" ] || [ "$TIER" = "all" ]; then
    if [ ! -f "$PIPER_DIR/es_ES-davefx-medium.onnx" ]; then
        echo "Descargando Piper Voice Alternativa (es_ES-davefx-medium)..."
        curl -L --progress-bar -o "$PIPER_DIR/es_ES-davefx-medium.onnx" \
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx" || true
        curl -L --progress-bar -o "$PIPER_DIR/es_ES-davefx-medium.onnx.json" \
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json" || true
        echo "Piper Davefx Voice listo (si disponible)."
    fi
fi

# 4. Modelo STT de Alta Precisión Opcional: Sherpa-ONNX Whisper Small (~360 MB int8)
if [ "$TIER" = "all" ] || [ "${SENSORY_STT_MODEL:-}" = "small" ]; then
    WHISPER_SMALL_DIR="$MODELS_DIR/sherpa-onnx-whisper-small"
    mkdir -p "$WHISPER_SMALL_DIR"
    if [ ! -f "$WHISPER_SMALL_DIR/small-encoder.int8.onnx" ]; then
        echo "Descargando Whisper Small ONNX (Alta precisión en español)..."
        TAR_SMALL="$MODELS_DIR/whisper-small.tar.bz2"
        curl -L --progress-bar -o "$TAR_SMALL" "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.tar.bz2"
        echo "Extrayendo Whisper Small..."
        tar -xjf "$TAR_SMALL" -C "$MODELS_DIR"
        rm -f "$TAR_SMALL"
        echo "Whisper Small listo."
    else
        echo "Whisper Small ya esta instalado."
    fi
fi

echo "=========================================================="
echo "Modelos sensoriales listos en $MODELS_DIR (Tier: $TIER)"
echo "=========================================================="
