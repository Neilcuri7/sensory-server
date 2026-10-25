#!/usr/bin/env bash
# ==============================================================================
# 🌸 Aoi Sensory Hub: Script Automático de Descarga de Modelos
# Descarga los pesos ONNX de Whisper Base (STT) y Piper Voice (TTS)
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$DIR/models"
mkdir -p "$MODELS_DIR/piper" "$MODELS_DIR/sherpa-onnx-whisper-base"

echo "=========================================================="
echo "🌸 Descargando Modelos para Aoi Sensory Hub..."
echo "=========================================================="

# 1. Modelo STT: Sherpa-ONNX Whisper Base Multilingüe (~140 MB)
WHISPER_DIR="$MODELS_DIR/sherpa-onnx-whisper-base"
if [ ! -f "$WHISPER_DIR/base-encoder.int8.onnx" ]; then
    echo "🎙️ Descargando Whisper Base ONNX..."
    TAR_FILE="$MODELS_DIR/whisper-base.tar.bz2"
    curl -L --progress-bar -o "$TAR_FILE" "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.tar.bz2"
    echo "📦 Extrayendo Whisper Base..."
    tar -xjf "$TAR_FILE" -C "$MODELS_DIR"
    rm -f "$TAR_FILE"
    echo "✅ Whisper Base listo."
else
    echo "✅ Whisper Base ya está instalado."
fi

# 2. Modelo TTS: Piper Voice en Español (Sharvard Medium ~45 MB)
PIPER_DIR="$MODELS_DIR/piper"
if [ ! -f "$PIPER_DIR/es_ES-sharvard-medium.onnx" ]; then
    echo "🔊 Descargando Piper Voice (es_ES-sharvard-medium)..."
    curl -L --progress-bar -o "$PIPER_DIR/es_ES-sharvard-medium.onnx" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx"
    curl -L --progress-bar -o "$PIPER_DIR/es_ES-sharvard-medium.onnx.json" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx.json"
    echo "✅ Piper Voice listo."
else
    echo "✅ Piper Voice ya está instalado."
fi

# 3. Modelo TTS: Kokoro v1.0 ONNX Multilingüe (~88 MB int8 + ~27 MB voices)
mkdir -p "$MODELS_DIR/kokoro"
KOKORO_DIR="$MODELS_DIR/kokoro"
if [ ! -f "$KOKORO_DIR/kokoro-v1.0.int8.onnx" ]; then
    echo "🌸 Descargando Kokoro v1.0 ONNX (int8)..."
    curl -L --progress-bar -o "$KOKORO_DIR/kokoro-v1.0.int8.onnx" \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx"
    curl -L --progress-bar -o "$KOKORO_DIR/voices-v1.0.bin" \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
    echo "✅ Kokoro v1.0 ONNX listo."
else
    echo "✅ Kokoro v1.0 ONNX ya está instalado."
fi

echo "=========================================================="
echo "✨ Todos los modelos sensoriales están listos en $MODELS_DIR"
echo "=========================================================="
