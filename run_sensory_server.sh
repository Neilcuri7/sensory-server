#!/usr/bin/env bash
# ==============================================================================
# Sensory Server (Puerto 8888)
# Unifica STT, TTS, Visión y OCR en 1 solo proceso
# Soporta niveles de ejecución: --tier [light|full]
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

TIER="${SENSORY_TIER:-full}"
REMAINING_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier)
      TIER="${2:-full}"
      shift 2
      ;;
    --tier=*)
      TIER="${1#*=}"
      shift 1
      ;;
    *)
      REMAINING_ARGS+=("$1")
      shift 1
      ;;
  esac
done

export SENSORY_TIER="$(echo "$TIER" | tr '[:upper:]' '[:lower:]')"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export SENSORY_PORT=${SENSORY_PORT:-${AOI_SENSORY_PORT:-8888}}
export SENSORY_HOST=${SENSORY_HOST:-${AOI_SENSORY_HOST:-0.0.0.0}}
export SENSORY_CPU_THREADS=${SENSORY_CPU_THREADS:-6}

echo "Iniciando Sensory Server en $SENSORY_HOST:$SENSORY_PORT (Tier: $SENSORY_TIER, CPU Threads: $SENSORY_CPU_THREADS)..."
exec python app.py "${REMAINING_ARGS[@]}"
