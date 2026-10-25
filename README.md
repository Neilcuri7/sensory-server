# 🌸 Aoi Sensory Hub Server

> **Servidor Sensorial Unificado y Autónomo de Percepción para Aoi Desktop Companion**  
> Consolida **STT (Reconocimiento de Voz)**, **TTS (Síntesis de Voz)**, **Visión Multimodal** y **OCR (Detección de Texto y UI Grounding)** en un único proceso FastAPI sobre el puerto **`8888`**.

---

## 🏛️ Arquitectura y Capacidades

| Módulo | Motor / Backend | Endpoint REST | Descripción |
| :--- | :--- | :--- | :--- |
| **🎙️ STT** | Sherpa-ONNX / Whisper-Base | `POST /v1/audio/transcriptions` | Transcripción de audio OpenAI-compatible en $<150\text{ ms}$. |
| **🔊 TTS** | Kokoro ONNX / Piper ONNX (100% Local) | `POST /v1/audio/speech` | Síntesis neural de voz 100% local (voces femeninas) en streaming. |
| **👁️ Visión** | Qwen2-VL-2B-Instruct | `POST /v1/vision/inspect`<br/>`POST /v1/vision/locate-ui` | Inspección visual de capturas y localización de elementos. |
| **🔍 OCR** | RapidOCR (ONNX Runtime) | `POST /v1/ocr/image`<br/>`POST /v1/ocr/base64` | Extracción de texto y bounding boxes `[xmin, ymin, xmax, ymax]` en $<30\text{ ms}$. |

---

## 🚀 Inicio Rápido

### 1. Descargar Pesos de Modelos (Primera vez)
```bash
./download_models.sh
```

### 2. Ejecutar con Nix (Recomendado en NixOS / Linux)
```bash
nix-shell --run "python app.py"
```

### 3. Ejecutar con Python Virtualenv
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run_sensory_server.sh
```

### 4. Ejecutar con Docker
```bash
docker build -t aoi-sensory-server .
docker run -d -p 8888:8888 --name aoi-sensory-server aoi-sensory-server
```

---

## 🧪 Verificación de Salud
```bash
curl http://localhost:8888/health
```

Respuesta esperada:
```json
{
  "status": "healthy",
  "service": "aoi-sensory-hub",
  "port": 8888,
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
MIT License - 2026 Aoi Project.
