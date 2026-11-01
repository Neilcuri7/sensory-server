# ⚡ Sensory Hub Server

> **Servidor Sensorial Unificado y Autónomo de Percepción Local**  
> Motor de alto rendimiento en FastAPI que consolida **STT (Reconocimiento de Voz)**, **TTS (Síntesis de Voz Neuronal)**, **Visión Multimodal** y **OCR (Detección de Texto y UI Grounding)** en un único proceso optimizado en el puerto **`8888`**.

---

## 🏛️ Arquitectura y Capacidades

| Módulo | Motor / Backend | Endpoint REST | Descripción |
| :--- | :--- | :--- | :--- |
| **🎙️ STT** | Sherpa-ONNX / Whisper-Base | `POST /v1/audio/transcriptions`<br/>`POST /stt` | Transcripción de audio OpenAI-compatible en $<150\text{ ms}$. |
| **🔊 TTS** | Kokoro ONNX / Piper ONNX | `POST /v1/audio/speech`<br/>`POST /tts`<br/>`GET /v1/audio/voices` | Síntesis neural local con voice blending, OLA pitch shift y soft-limiting. |
| **👁️ Visión** | Qwen2-VL / RapidOCR UI Grounding | `POST /v1/vision/inspect`<br/>`POST /v1/vision/locate-ui` | Inspección visual de capturas y localización de elementos de interfaz. |
| **🔍 OCR** | RapidOCR (ONNX Runtime) | `POST /v1/ocr/image`<br/>`POST /v1/ocr/base64` | Extracción de texto y bounding boxes `[xmin, ymin, xmax, ymax]` en $<30\text{ ms}$. |
| **🩺 Health** | Diagnóstico en tiempo real | `GET /health` | Estado del servicio, salud de modelos cargados y tier activo. |

---

## 🎛️ Tiers de Hardware y Aprovisionamiento

Sensory Hub permite seleccionar el nivel de modelos y consumo de recursos según la capacidad del dispositivo:

| Nivel | Modelos Incluidos | Tamaño | Hardware Recomendado | Comportamiento |
| :--- | :--- | :--- | :--- | :--- |
| **`light`** | Whisper Base STT + Piper TTS (Voz femenina española) | ~185 MB | CPU de bajo consumo, laptops sin GPU dedicada, entornos limitados en RAM. | Enrutamiento automático ultrarrápido a Piper (<80ms en CPU, 0 RAM overhead). Omite Kokoro sin advertencias. |
| **`full`** | Whisper Base STT + Piper TTS + **Kokoro v1.0 ONNX** (Español Anime) | ~300 MB | CPUs potentes (Intel Core i7 14ª Gen+) o GPUs NVIDIA (6 GB+ VRAM). | Activa síntesis Kokoro con voice blending neural (`ef_dora` + `jf_alpha`), OLA pitch shift y concurrencia multi-hilo. |

---

## 🚀 Inicio Rápido

### 1. Descargar Pesos de Modelos por Tier

```bash
# Para entornos ligeros / solo CPU:
./download_models.sh --tier light

# Para máxima calidad con voz neuronal Kokoro:
./download_models.sh --tier full
```

*(También soporta `--tier all` para descargar la totalidad de los modelos disponibles).*

---

### 2. Ejecutar el Servidor

#### Opción A: Con el Script de Arranque (Recomendado)
```bash
# Ejecución en tier ligero:
./run_sensory_server.sh --tier light

# Ejecución en tier completo:
./run_sensory_server.sh --tier full
```

#### Opción B: Con Nix (NixOS / Linux)
```bash
nix-shell --run "./run_sensory_server.sh --tier full"
```

#### Opción C: Con Entorno Virtual Python
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export SENSORY_TIER=full
python app.py
```

#### Opción D: Con Contenedor Docker
```bash
docker build -t sensory-hub-server .
docker run -d -p 8888:8888 --name sensory-hub-server sensory-hub-server
```

---

## 🎧 Características Avanzadas del Pipeline de Audio

1. **Voice Blending Neuronal (Kokoro ONNX):**
   - Interpolación en espacio latente: 70% `ef_dora` (español nativo con fonética y prosodia correcta) + 30% `jf_alpha` (timbre dulce).
   - Elimina la "traba de lengua" y las caídas fonéticas translingües.
2. **Pitch Shifting OLA (Windowed Overlap-Add):**
   - Modifica el tono mediante ventana Hann preservando de forma exacta la duración temporal del audio (`len(output) == len(samples)` en $<10\text{ ms}$).
   - Cero aceleración de cinta ni efecto ardilla.
3. **Soft-Limiting Anti-Clipping:**
   - Normalización suave de picos transitorios cuando $\text{peak} > 0.98$, previniendo distorsión armónica digital en consonantes oclusivas (`p`, `t`, `k`).
4. **Concurrencia Segura y Afinidad de CPU:**
   - Concurrencia real sin bloqueos globales para inferencias neuronales ONNX.
   - Aislamiento de exclusión mutua para librerías C no reentrantes (`libespeak-ng` / Piper).
   - Limitación de hilos intra-op a 4 núcleos para garantizar afinidad con P-Cores en procesadores de arquitectura híbrida.
   - Detección elástica de proveedores CUDA configurada para evitar sobreasignación de VRAM en GPUs de 6 GB (`kSameAsRequested`).

---

## 🧪 Verificación de Salud

```bash
curl http://localhost:8888/health
```

Respuesta esperada en `full` tier:
```json
{
  "status": "healthy",
  "service": "sensory-hub",
  "tier": "full",
  "port": 8888,
  "models": {
    "stt": true,
    "kokoro": true,
    "piper": true
  },
  "modules": {
    "stt": "Sherpa-ONNX / Whisper-Base",
    "tts": "Kokoro ONNX / Piper ONNX (100% Local Female Voices)",
    "vision": "RapidOCR ONNX UI Grounding",
    "ocr": "RapidOCR ONNX"
  }
}
```

---

## 📄 Licencia

MIT License.
