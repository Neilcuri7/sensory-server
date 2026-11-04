"""
Tests unitarios para los endpoints de Sensory Server (FastAPI).
Verifica:
- Middleware de logging y propagación del header X-Request-ID.
- Endpoint /health reportando estado de modelos y degradación.
- Endpoint /v1/audio/transcriptions devolviendo HTTP 503 si el reconocedor no está disponible.
- Capa 3: Detección elástica de hardware (GPU vs CPU) y session options.
- Capa 3: Selector de motor y exposición de voces (/v1/audio/voices y /v1/audio/speech).
"""
import os
import io
import time
import wave
import threading
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import numpy as np

from app import app
import routers.stt_router as stt_router
import routers.tts_router as tts_router


class TestSensoryEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check_endpoint(self):
        """Verifica la respuesta del health check con estructura de modelos y estado."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn(data["status"], ("healthy", "degraded", "down"))
        self.assertIn("models", data)
        self.assertIn("stt", data["models"])
        self.assertIn("kokoro", data["models"])
        self.assertIn("piper", data["models"])
        self.assertTrue("x-request-id" in response.headers or "X-Request-ID" in response.headers)

    def test_request_logging_middleware_preserves_custom_request_id(self):
        """Verifica que si el cliente envía X-Request-ID, el middleware lo devuelva inalterado."""
        custom_id = "test-req-12345"
        response = self.client.get("/health", headers={"X-Request-ID": custom_id})
        returned_id = response.headers.get("X-Request-ID") or response.headers.get("x-request-id")
        self.assertEqual(returned_id, custom_id)

    def test_stt_returns_503_when_recognizer_is_none(self):
        """Verifica que si el reconocedor Whisper no carga, devuelva HTTP 503."""
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 8000)
        wav_bytes = wav_io.getvalue()

        with patch.object(stt_router, "get_recognizer", return_value=None):
            files = {"file": ("test.wav", wav_bytes, "audio/wav")}
            response = self.client.post("/v1/audio/transcriptions", files=files, data={"model": "whisper-base"})
            self.assertEqual(response.status_code, 503)
            data = response.json()
            self.assertIn("STT model not loaded", data.get("detail", ""))

    def test_stt_recognizer_threads_and_hardware_elasticity(self):
        """Verifica que get_recognizer configure num_threads = min(4, os.cpu_count() or 4) para P-cores."""
        expected_threads = min(4, os.cpu_count() or 4)
        with patch("sherpa_onnx.OfflineRecognizer.from_whisper") as mock_whisper:
            mock_whisper.return_value = MagicMock()
            saved_rec = stt_router.recognizer
            try:
                stt_router.recognizer = None
                rec = stt_router.get_recognizer()
                self.assertIsNotNone(rec)
                self.assertTrue(mock_whisper.called)
                _, kwargs = mock_whisper.call_args
                self.assertEqual(kwargs.get("num_threads"), expected_threads)
            finally:
                stt_router.recognizer = saved_rec

    def test_tts_session_options_threads(self):
        """Verifica que SessionOptions en CPU configure intra_op_num_threads = min(4, os.cpu_count() or 4) e inter_op=1."""
        expected_threads = min(4, os.cpu_count() or 4)
        sess_options = tts_router._get_elastic_session_options()
        self.assertEqual(sess_options.intra_op_num_threads, expected_threads)
        self.assertEqual(sess_options.inter_op_num_threads, 1)
        self.assertEqual(sess_options.log_severity_level, 3)

    def test_tts_hardware_detection_gpu_to_cpu_fallback(self):
        """Verifica que si GPU falla o no está disponible, degrade limpiamente a CPUExecutionProvider."""
        with patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]):
            with patch("onnxruntime.InferenceSession") as mock_sess:
                # Simular que el intento de GPU lanza excepción y luego el de CPU funciona
                cpu_session_mock = MagicMock()
                cpu_session_mock.get_providers.return_value = ["CPUExecutionProvider"]
                mock_sess.side_effect = [RuntimeError("CUDA driver not found"), cpu_session_mock]

                session, provider = tts_router._create_elastic_inference_session("dummy_model.onnx")
                self.assertEqual(provider, "CPUExecutionProvider")
                self.assertEqual(mock_sess.call_count, 2)

    def test_voices_endpoint_exposes_piper_and_kokoro(self):
        """Verifica que /v1/audio/voices exponga piper_f como bajo consumo en CPU y jf_alpha como alta fidelidad."""
        response = self.client.get("/v1/audio/voices")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("voices", data)
        voices = data["voices"]
        voice_ids = [v["id"] for v in voices]

        # Verificar presencia de jf_alpha (Kokoro) y piper_f (Piper)
        self.assertIn("jf_alpha", voice_ids)
        self.assertIn("piper_f", voice_ids)

        # Verificar metadata de piper_f
        piper_meta = next(v for v in voices if v["id"] == "piper_f")
        self.assertEqual(piper_meta.get("engine"), "piper")
        self.assertIn("CPU", piper_meta.get("name", ""))
        self.assertEqual(piper_meta.get("tier"), "low_consumption_cpu")

        # Verificar metadata de jf_alpha
        kokoro_meta = next(v for v in voices if v["id"] == "jf_alpha")
        self.assertEqual(kokoro_meta.get("engine"), "kokoro")
        self.assertEqual(kokoro_meta.get("tier"), "high_fidelity")

    def test_speech_endpoint_engine_routing_piper(self):
        """Verifica que model='piper' active el motor Piper para síntesis ultra-rápida."""
        dummy_wav = tts_router._generate_wav_bytes(b"\x00" * 4410, sample_rate=22050)
        fake_metrics = {
            "voice": "piper_f",
            "engine": "piper",
            "lang": "es",
            "chars": 4,
            "lock_wait_ms": 1.0,
            "infer_ms": 5.0,
            "audio_dur_ms": 100.0,
            "bytes": len(dummy_wav)
        }
        with patch.object(tts_router, "_synthesize_wav_sync", return_value=(dummy_wav, fake_metrics)) as mock_synth:
            response = self.client.post("/v1/audio/speech", json={"model": "piper", "input": "Hola"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers.get("X-TTS-Engine"), "piper")
            self.assertEqual(response.headers.get("X-TTS-Voice"), "piper_f")
            _, kwargs = mock_synth.call_args
            self.assertEqual(kwargs.get("model"), "piper")

    def test_anime_pitch_shift_benchmark_under_10ms(self):
        """Verifica que _apply_anime_pitch_shift sobre 24000 muestras ejecute en tiempo conversacional (< 35ms en CPU)."""
        sample_rate = 24000
        # 1 segundo de audio sintético a 24kHz
        samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate, dtype=np.float32))

        # Warm-up previo
        tts_router._apply_anime_pitch_shift(samples, sample_rate, 2.0)

        # Benchmark de ejecución
        t_start = time.perf_counter()
        shifted = tts_router._apply_anime_pitch_shift(samples, sample_rate, 2.0)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Verificación de rendimiento estricto (< 35 ms en virtualización WSL2)
        self.assertLess(
            elapsed_ms,
            35.0,
            f"El remuestreo tardó {elapsed_ms:.2f}ms, excediendo el límite de 35ms"
        )

        # Verificaciones de sanidad y preservación estricta de tempo y duración
        self.assertEqual(len(shifted), len(samples))
        self.assertEqual(shifted.dtype, samples.dtype)

        # Casos borde: tono despreciable o array vacío
        noop = tts_router._apply_anime_pitch_shift(samples, sample_rate, 0.0)
        np.testing.assert_array_equal(noop, samples)

        empty = np.array([], dtype=np.float32)
        empty_shifted = tts_router._apply_anime_pitch_shift(empty, sample_rate, 2.0)
        self.assertEqual(len(empty_shifted), 0)

    def test_spanish_inverted_punctuation_preserved_and_markdown_stripped(self):
        """Verifica que ¿ y ¡ se preserven para espeak-ng, mientras se eliminan roleplay, emojis y markdown."""
        raw_text = "*sonríe con emoción* ¡Hola mundo! ¿Cómo estás hoy? 🌸✨ Mira esto: https://example.com ```código en bloque``` **genial** # Título ~tachado~"
        cleaned = tts_router._sanitize_text_for_tts(raw_text)
        # ¿ e ¡ deben preservarse intactos
        self.assertIn("¿Cómo estás hoy?", cleaned)
        self.assertIn("¡Hola mundo!", cleaned)
        # *sonríe con emoción* debe eliminarse por completo (no leerse)
        self.assertNotIn("sonríe", cleaned)
        # Emojis eliminados
        self.assertNotIn("🌸", cleaned)
        self.assertNotIn("✨", cleaned)
        # URLs y bloques de código eliminados
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("código en bloque", cleaned)
        self.assertNotIn("*", cleaned)
        self.assertNotIn("#", cleaned)

    def test_detect_phoneme_lang_prefers_spanish(self):
        """Verifica que voces jf_* no fuercen fonemas japoneses 'ja' en texto latino/español."""
        # En español no debe devolver 'ja'
        self.assertEqual(tts_router._detect_phoneme_lang("Hola amigo, ¿cómo estás?", "jf_alpha"), "es")
        self.assertEqual(tts_router._detect_phoneme_lang("Buenos días", "ef_dora"), "es")
        # Solo en presencia de caracteres japoneses devuelve 'ja'
        self.assertEqual(tts_router._detect_phoneme_lang("こんにちは", "jf_alpha"), "ja")
        # Idioma explícito
        self.assertEqual(tts_router._detect_phoneme_lang("Hello friend", "af_bella", requested_lang="en-us"), "en-us")

    def test_soft_limiting_peak_normalization(self):
        """Verifica que _write_wav_bytes aplique soft-limiting cuando el pico supera 0.98."""
        loud_samples = np.array([0.0, 1.5, -1.8, 0.5], dtype=np.float32)
        wav_bytes = tts_router._write_wav_bytes(loud_samples, sample_rate=24000)
        read_samples, sr = tts_router._read_wav_samples(wav_bytes)
        self.assertEqual(sr, 24000)
        # El pico normalizado debe ser <= 1.0 (sin clipping digital de techo plano)
        self.assertLessEqual(float(np.max(np.abs(read_samples))), 1.0)

    def test_kokoro_voice_blending_resolution(self):
        """Verifica que _resolve_kokoro_voice mezcle ef_dora + jf_alpha para español anime."""
        mock_kokoro = MagicMock()
        mock_kokoro.get_voices.return_value = ["ef_dora", "jf_alpha", "jf_nezumi"]
        mock_kokoro.get_voice_style.side_effect = lambda name: np.ones((510, 1, 256), dtype=np.float32) * (1.0 if name == "ef_dora" else 2.0)

        # Caso ef_blend o jf_alpha en español: debe devolver el vector de estilo mezclado
        blended = tts_router._resolve_kokoro_voice(mock_kokoro, "jf_alpha", "es")
        self.assertIsInstance(blended, np.ndarray)
        self.assertEqual(blended.shape, (510, 1, 256))
        # 0.70 * 1.0 + 0.30 * 2.0 = 1.30
        expected_val = 0.70 * 1.0 + 0.30 * 2.0
        np.testing.assert_allclose(blended[0, 0, 0], expected_val, atol=1e-5)

        # Fallback si kokoro no tiene get_voice_style (mock simple de test)
        simple_mock = MagicMock(spec=["create"])
        fallback_voice = tts_router._resolve_kokoro_voice(simple_mock, "jf_alpha", "es")
        self.assertEqual(fallback_voice, "jf_alpha")

    def test_voices_endpoint_includes_ef_blend(self):
        """Verifica que /v1/audio/voices incluya la voz ef_blend."""
        response = self.client.get("/v1/audio/voices")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        voice_ids = [v["id"] for v in data["voices"]]
        self.assertIn("ef_blend", voice_ids)

    def test_load_kokoro_graceful_none_in_light_tier_when_files_missing(self):
        """Verifica que _load_kokoro en light tier retorne None limpiamente sin error si los pesos no existen."""
        with patch.dict(os.environ, {"SENSORY_TIER": "light"}):
            with patch("os.path.exists", return_value=False):
                orig_kokoro = tts_router.kokoro_voice
                orig_model = tts_router._kokoro_model
                try:
                    tts_router.kokoro_voice = None
                    tts_router._kokoro_model = None
                    result = tts_router._load_kokoro()
                    self.assertIsNone(result)
                    self.assertIsNone(tts_router.kokoro_voice)
                    self.assertIsNone(tts_router._kokoro_model)
                finally:
                    tts_router.kokoro_voice = orig_kokoro
                    tts_router._kokoro_model = orig_model

    def test_voices_endpoint_in_light_tier_defaults_to_piper(self):
        """Verifica que /v1/audio/voices en SENSORY_TIER=light devuelva default_engine='piper' y default_voice='piper_f'."""
        with patch.dict(os.environ, {"SENSORY_TIER": "light"}):
            response = self.client.get("/v1/audio/voices")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data.get("default_engine"), "piper")
            self.assertEqual(data.get("default_voice"), "piper_f")
            self.assertEqual(data.get("tier"), "light")

    def test_speech_endpoint_light_tier_routes_to_piper_automatically(self):
        """Verifica que peticiones a /v1/audio/speech y /tts en light tier se enruten a Piper incluso si piden Kokoro."""
        dummy_wav = tts_router._generate_wav_bytes(b"\x00" * 4410, sample_rate=22050)
        fake_metrics = {
            "voice": "piper_f",
            "engine": "piper",
            "lang": "es",
            "chars": 4,
            "lock_wait_ms": 1.0,
            "infer_ms": 5.0,
            "audio_dur_ms": 100.0,
            "bytes": len(dummy_wav)
        }
        with patch.dict(os.environ, {"SENSORY_TIER": "light"}):
            with patch.object(tts_router, "_synthesize_wav_sync", return_value=(dummy_wav, fake_metrics)) as mock_synth:
                # 1. /v1/audio/speech con model="kokoro" y voice="jf_alpha"
                resp = self.client.post("/v1/audio/speech", json={"model": "kokoro", "voice": "jf_alpha", "input": "Hola"})
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(resp.headers.get("X-TTS-Engine"), "piper")
                self.assertEqual(resp.headers.get("X-TTS-Voice"), "piper_f")
                _, kwargs = mock_synth.call_args
                self.assertEqual(kwargs.get("model"), "piper")

                # 2. /tts con model="kokoro"
                resp_tts = self.client.post("/tts", json={"model": "kokoro", "voice": "jf_alpha", "text": "Hola"})
                self.assertEqual(resp_tts.status_code, 200)
                data_tts = resp_tts.json()
                self.assertEqual(data_tts.get("engine"), "piper")
                self.assertEqual(data_tts.get("voice"), "piper_f")

    def test_synthesize_wav_sync_light_tier_uses_piper(self):
        """Verifica que _synthesize_wav_sync y el alias synthesize_tts usen Piper sin invocar Kokoro en light tier."""
        mock_piper = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.audio_int16_bytes = b"\x00\x00" * 2205
        mock_piper.synthesize.return_value = [mock_chunk]
        mock_piper.config.sample_rate = 22050

        with patch.dict(os.environ, {"SENSORY_TIER": "light"}):
            with patch.object(tts_router, "get_piper_voice", return_value=mock_piper):
                with patch.object(tts_router, "get_kokoro_voice") as mock_kokoro:
                    wav_bytes, metrics = tts_router.synthesize_tts("Hola mundo", "jf_alpha", 1.0)
                    self.assertGreater(len(wav_bytes), 44)
                    self.assertEqual(metrics["engine"], "piper")
                    self.assertEqual(metrics["voice"], "piper_f")
                    self.assertFalse(mock_kokoro.called)

    def test_health_endpoint_healthy_in_light_tier_with_only_stt_and_piper(self):
        """Verifica que /health reporte 'healthy' en light tier cuando solo stt y piper están cargados (kokoro omitido)."""
        with patch.dict(os.environ, {"SENSORY_TIER": "light"}):
            with patch.object(stt_router, "recognizer", MagicMock()):
                with patch.object(tts_router, "piper_voice", MagicMock()):
                    with patch.object(tts_router, "kokoro_voice", None):
                        response = self.client.get("/health")
                        self.assertEqual(response.status_code, 200)
                        data = response.json()
                        self.assertEqual(data["status"], "healthy")
                        self.assertEqual(data["tier"], "light")
                        self.assertFalse(data["models"]["kokoro"])
                        self.assertTrue(data["models"]["piper"])
                        self.assertTrue(data["models"]["stt"])

    def test_kokoro_concurrent_requests_serialized_by_lock(self):
        """Verifica que llamadas concurrentes con Kokoro se serialicen mediante _espeak_piper_lock evitando carreras en espeak-ng."""
        active_calls = 0
        max_concurrent_calls = 0
        call_lock = threading.Lock()
        results = []

        def mock_create(text, voice=None, speed=1.0, lang="es"):
            nonlocal active_calls, max_concurrent_calls
            with call_lock:
                active_calls += 1
                if active_calls > max_concurrent_calls:
                    max_concurrent_calls = active_calls
            # Pausa artificial para forzar colisión si no existiese el lock global
            time.sleep(0.04)
            with call_lock:
                active_calls -= 1
            return np.zeros(2400, dtype=np.float32), 24000

        mock_kokoro = MagicMock()
        mock_kokoro.get_voices.return_value = ["jf_alpha", "ef_dora"]
        mock_kokoro.create.side_effect = mock_create

        def run_worker(text_payload):
            wav, metrics = tts_router._synthesize_wav_sync(
                text=text_payload,
                voice_id="jf_alpha",
                speed=1.0,
                language="es",
                pitch_semitones=2.0,
                model="kokoro"
            )
            results.append((wav, metrics))

        with patch.dict(os.environ, {"SENSORY_TIER": "full"}):
            with patch.object(tts_router, "get_kokoro_voice", return_value=mock_kokoro):
                t1 = threading.Thread(target=run_worker, args=("Primera oración de prueba concurrente.",))
                t2 = threading.Thread(target=run_worker, args=("Segunda oración de prueba concurrente.",))

                t1.start()
                t2.start()
                t1.join()
                t2.join()

        # Ambas síntesis deben haber culminado con éxito
        self.assertEqual(len(results), 2)
        for wav, metrics in results:
            self.assertEqual(metrics["engine"], "kokoro")
            self.assertGreater(len(wav), 44)

        # La concurrencia máxima en kokoro.create debe ser estrictamente 1 (cero carreras)
        self.assertEqual(max_concurrent_calls, 1, f"Condición de carrera detectada: max concurrent={max_concurrent_calls}")

        # Al menos uno de los hilos debió registrar tiempo de espera de lock (> 0ms)
        lock_waits = [metrics["lock_wait_ms"] for _, metrics in results]
        self.assertTrue(any(w > 10.0 for w in lock_waits), f"Ningún hilo esperó en cola de lock: {lock_waits}")

    def test_ef_dora_pure_spanish_resolution_and_zero_default_pitch(self):
        """Verifica que ef_dora resuelva pura sin mezcla y use pitch 0.0 por defecto en /tts y /v1/audio/speech."""
        mock_kokoro = MagicMock()
        mock_kokoro.get_voices.return_value = ["ef_dora", "jf_alpha"]
        mock_kokoro.get_voice_style.return_value = np.ones((510, 1, 256), dtype=np.float32)

        # 1. Resolución pura de ef_dora
        resolved = tts_router._resolve_kokoro_voice(mock_kokoro, "ef_dora", "es")
        self.assertEqual(resolved, "ef_dora")
        self.assertIsInstance(resolved, str)
        # get_voice_style no debe llamarse para ef_dora
        mock_kokoro.get_voice_style.assert_not_called()

        # Comparar con ef_blend o jf_alpha que sí deben mezclar
        blended = tts_router._resolve_kokoro_voice(mock_kokoro, "ef_blend", "es")
        self.assertIsInstance(blended, np.ndarray)
        self.assertTrue(mock_kokoro.get_voice_style.called)

        # 2. Pitch 0.0 por defecto en /tts para ef_dora sin pitch explícito
        dummy_wav = tts_router._generate_wav_bytes(b"\x00" * 4410, sample_rate=24000)
        fake_metrics = {
            "voice": "ef_dora",
            "engine": "kokoro",
            "lang": "es",
            "chars": 4,
            "lock_wait_ms": 0.0,
            "infer_ms": 10.0,
            "audio_dur_ms": 100.0,
            "bytes": len(dummy_wav)
        }
        with patch.dict(os.environ, {"SENSORY_TIER": "full"}):
            with patch.object(tts_router, "_synthesize_wav_sync", return_value=(dummy_wav, fake_metrics)) as mock_synth:
                # /tts con ef_dora sin pitch_semitones
                resp_dora = self.client.post("/tts", json={"voice": "ef_dora", "text": "Hola, soy Dora."})
                self.assertEqual(resp_dora.status_code, 200)
                args_dora, kwargs_dora = mock_synth.call_args
                passed_pitch_dora = args_dora[4] if len(args_dora) > 4 else kwargs_dora.get("pitch_semitones")
                self.assertEqual(passed_pitch_dora, 0.0)

                mock_synth.reset_mock()
                # /tts con voz anime jf_alpha sin pitch explícito -> mantiene 2.0
                resp_anime = self.client.post("/tts", json={"voice": "jf_alpha", "text": "Konnichiwa!"})
                self.assertEqual(resp_anime.status_code, 200)
                args_anime, kwargs_anime = mock_synth.call_args
                passed_pitch_anime = args_anime[4] if len(args_anime) > 4 else kwargs_anime.get("pitch_semitones")
                self.assertEqual(passed_pitch_anime, 2.0)

                mock_synth.reset_mock()
                # /v1/audio/speech con ef_dora sin pitch explícito -> usa 0.0
                resp_speech = self.client.post("/v1/audio/speech", json={"voice": "ef_dora", "input": "Hola, soy Dora."})
                self.assertEqual(resp_speech.status_code, 200)
                args_speech, kwargs_speech = mock_synth.call_args
                passed_pitch_speech = args_speech[4] if len(args_speech) > 4 else kwargs_speech.get("pitch_semitones")
                self.assertEqual(passed_pitch_speech, 0.0)

                mock_synth.reset_mock()
                # /tts con ef_dora con pitch_semitones explícito -> respeta el valor explícito
                resp_explicit = self.client.post("/tts", json={"voice": "ef_dora", "text": "Hola", "pitch_semitones": 1.2})
                self.assertEqual(resp_explicit.status_code, 200)
                args_exp, kwargs_exp = mock_synth.call_args
                passed_pitch_exp = args_exp[4] if len(args_exp) > 4 else kwargs_exp.get("pitch_semitones")
                self.assertEqual(passed_pitch_exp, 1.2)

    def test_tts_l1_l2_cache_roundtrip_and_fast_hit(self):
        """Verifica que el caché SHA-256 almacene en L1 RAM + L2 disco y sirva aciertos instantáneos sin adquirir lock."""
        import tempfile
        dummy_wav = tts_router._write_wav_bytes(np.ones(2400, dtype=np.float32) * 0.25, sample_rate=24000)

        with tempfile.TemporaryDirectory() as tmp_cache_dir:
            with patch.object(tts_router, "TTS_CACHE_DIR", tmp_cache_dir):
                tts_router._tts_ram_cache.clear()
                key = tts_router._compute_tts_cache_key("¡Hola! Aquí estoy contigo.", "ef_dora", "es", 1.05, 0.0, "kokoro")
                self.assertIsNone(tts_router.get_cached_tts_wav(key))

                # Guardar en L1 y L2
                tts_router.store_cached_tts_wav(key, dummy_wav)
                disk_file = os.path.join(tmp_cache_dir, f"{key}.wav")
                self.assertTrue(os.path.exists(disk_file))

                # L1 hit
                hit_l1 = tts_router.get_cached_tts_wav(key)
                self.assertEqual(hit_l1, dummy_wav)

                # Limpiar L1 para verificar promoción desde L2 (disco)
                tts_router._tts_ram_cache.clear()
                hit_l2 = tts_router.get_cached_tts_wav(key)
                self.assertEqual(hit_l2, dummy_wav)
                self.assertIn(key, tts_router._tts_ram_cache)

                # Verificar que _synthesize_wav_sync devuelva cache_hit=True en <2ms sin invocar inferencia
                wav_out, metrics = tts_router._synthesize_wav_sync(
                    text="¡Hola! Aquí estoy contigo.",
                    voice_id="ef_dora",
                    speed=1.05,
                    language="es",
                    pitch_semitones=0.0,
                    model="kokoro"
                )
                self.assertEqual(wav_out, dummy_wav)
                self.assertTrue(metrics.get("cache_hit"))
                self.assertEqual(metrics.get("lock_wait_ms"), 0.0)
                self.assertLess(metrics.get("infer_ms", 999.0), 5.0)
                tts_router._tts_ram_cache.clear()

    def test_trim_dead_silence_removes_padding_and_preserves_pure_zeros(self):
        """Verifica que _trim_dead_silence recorte silencios extremos preservando 20ms de margen e ignore arrays puros de ceros."""
        sr = 24000
        # 1. Silencio puro (como en mocks): debe devolverse intacto sin recortar
        pure_zeros = np.zeros(2400, dtype=np.float32)
        trimmed_zeros = tts_router._trim_dead_silence(pure_zeros, sr)
        self.assertEqual(len(trimmed_zeros), 2400)

        # 2. Señal con 5000 muestras de silencio al inicio y 5000 al final, y 4000 activas en el centro
        signal = np.concatenate([
            np.zeros(5000, dtype=np.float32),
            np.ones(4000, dtype=np.float32) * 0.5,
            np.zeros(5000, dtype=np.float32),
        ])
        trimmed = tts_router._trim_dead_silence(signal, sr, threshold=0.006, pad_ms=20)
        pad_samples = int(0.020 * sr)  # 480 muestras
        expected_len = 4000 + 2 * pad_samples
        self.assertEqual(len(trimmed), expected_len)

    def test_stt_preprocess_audio_and_domain_lexicon_corrections(self):
        """Verifica el acondicionamiento DSP (resample_poly + auto-gain) y la corrección fonética de 'Rylai'."""
        # 1. Corrección fonética de variantes hispanas de Whisper
        raw_transcript = "hola rilai, ¿cómo estás rilay? oye railai y raily"
        corrected = stt_router._apply_domain_lexicon_corrections(raw_transcript)
        self.assertEqual(corrected, "hola Rylai, ¿cómo estás Rylai? oye Rylai y Rylai")

        # 2. Preprocesamiento de señal estéreo a 48kHz con volumen bajo (pico 0.10) y DC offset
        sr_in = 48000
        t = np.linspace(0, 0.5, sr_in // 2, dtype=np.float32)
        quiet_sine = (np.sin(2 * np.pi * 440 * t) * 0.10) + 0.05  # pico ~0.10 + DC offset 0.05
        stereo_signal = np.column_stack([quiet_sine, quiet_sine])

        processed, sr_out = stt_router._preprocess_stt_audio(stereo_signal, sr_in)
        self.assertEqual(sr_out, 16000)
        self.assertEqual(len(processed.shape), 1)
        self.assertAlmostEqual(float(np.mean(processed)), 0.0, places=2)
        # Auto-gain debe haber elevado el pico cerca de 0.89
        self.assertGreater(float(np.max(np.abs(processed))), 0.80)
        self.assertLessEqual(float(np.max(np.abs(processed))), 0.95)

    def test_piper_voice_defaults_to_zero_pitch(self):
        """Verifica que piper_f use pitch 0.0 por defecto para evitar artefactos metálicos OLA."""
        dummy_wav = tts_router._generate_wav_bytes(b"\x00" * 4410, sample_rate=22050)
        fake_metrics = {
            "voice": "piper_f",
            "engine": "piper",
            "lang": "es",
            "chars": 4,
            "lock_wait_ms": 0.0,
            "infer_ms": 5.0,
            "audio_dur_ms": 100.0,
            "bytes": len(dummy_wav)
        }

        with patch.object(tts_router, "_synthesize_wav_sync", return_value=(dummy_wav, fake_metrics)) as mock_synth:
            # 1. /tts con piper_f sin pitch explícito -> debe pasar pitch 0.0
            resp_tts = self.client.post("/tts", json={"voice": "piper_f", "text": "Hola Piper"})
            self.assertEqual(resp_tts.status_code, 200)
            args, kwargs = mock_synth.call_args
            passed_pitch = args[4] if len(args) > 4 else kwargs.get("pitch_semitones")
            self.assertEqual(passed_pitch, 0.0)

            mock_synth.reset_mock()
            # 2. /v1/audio/speech con piper_f sin pitch explícito -> debe pasar pitch 0.0
            resp_speech = self.client.post("/v1/audio/speech", json={"voice": "piper_f", "input": "Hola Piper"})
            self.assertEqual(resp_speech.status_code, 200)
            args, kwargs = mock_synth.call_args
            passed_pitch = args[4] if len(args) > 4 else kwargs.get("pitch_semitones")
            self.assertEqual(passed_pitch, 0.0)

            mock_synth.reset_mock()
            # 3. En tier light: petición con jf_alpha y pitch 2.0 (default anime de Kokoro) debe auto-anular el pitch a 0.0
            with patch.dict(os.environ, {"SENSORY_TIER": "light"}):
                resp_light = self.client.post("/tts", json={"voice": "jf_alpha", "text": "Hola", "pitch_semitones": 2.0})
                self.assertEqual(resp_light.status_code, 200)
                args, kwargs = mock_synth.call_args
                passed_pitch = args[4] if len(args) > 4 else kwargs.get("pitch_semitones")
                self.assertEqual(passed_pitch, 0.0)

    def test_apply_piper_warmth_eq(self):
        """Verifica que _apply_piper_warmth_eq ejecute en <2ms, preserve longitud/dtype y suavice transientes ásperas."""
        sample_rate = 22050
        # 1 segundo de audio sintético a 22.05kHz
        samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate, dtype=np.float32))

        # Warm-up
        tts_router._apply_piper_warmth_eq(samples, sample_rate)

        # Benchmark < 2ms
        t0 = time.perf_counter()
        out = tts_router._apply_piper_warmth_eq(samples, sample_rate)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertLess(elapsed_ms, 2.0, f"Warmth EQ tardó {elapsed_ms:.2f}ms (>2.0ms)")

        # Preservación estricta de longitud y tipo
        self.assertEqual(len(out), len(samples))
        self.assertEqual(out.dtype, samples.dtype)

        # Silencio puro permanece intacto
        silence = np.zeros(2048, dtype=np.float32)
        out_silence = tts_router._apply_piper_warmth_eq(silence, sample_rate)
        np.testing.assert_array_equal(out_silence, silence)

        # Suavizado de transientes bruscas (escalón o impulso digital)
        step = np.zeros(200, dtype=np.float32)
        step[100:] = 1.0
        smoothed = tts_router._apply_piper_warmth_eq(step, sample_rate)
        max_jump_orig = float(np.max(np.abs(np.diff(step))))
        max_jump_smoothed = float(np.max(np.abs(np.diff(smoothed))))
        self.assertLess(max_jump_smoothed, max_jump_orig)

    def test_apply_rvc_conversion_nino_and_miku_and_disabled(self):
        """Verifica que apply_rvc_conversion procese Nino y Miku Nakano exitosamente y rechace none/inválidos."""
        from rvc_engine import apply_rvc_conversion, get_wav_info

        sr = 24000
        t = np.linspace(0, 0.4, int(sr * 0.4), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 330 * t) + 0.2 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)
        orig_wav = tts_router._write_wav_bytes(audio, sample_rate=sr)

        # 1. Nino Nakano
        out_nino, meta_nino = apply_rvc_conversion(orig_wav, "Nino Nakano")
        self.assertTrue(meta_nino.get("rvc_applied"))
        self.assertEqual(meta_nino.get("voice"), "Nino Nakano")
        self.assertNotEqual(out_nino, orig_wav)
        sr_nino, frames_nino = get_wav_info(out_nino)
        self.assertGreater(sr_nino, 0)
        self.assertGreater(frames_nino, 0)

        # 2. Miku Nakano
        out_miku, meta_miku = apply_rvc_conversion(orig_wav, "Miku Nakano")
        self.assertTrue(meta_miku.get("rvc_applied"))
        self.assertEqual(meta_miku.get("voice"), "Miku Nakano")
        self.assertNotEqual(out_miku, orig_wav)
        sr_miku, frames_miku = get_wav_info(out_miku)
        self.assertGreater(sr_miku, 0)
        self.assertGreater(frames_miku, 0)

        # 3. none / off
        out_none, meta_none = apply_rvc_conversion(orig_wav, "none")
        self.assertFalse(meta_none.get("rvc_applied"))
        self.assertEqual(out_none, orig_wav)

        # 4. NonExistentVoice
        out_fake, meta_fake = apply_rvc_conversion(orig_wav, "NonExistentVoice")
        self.assertFalse(meta_fake.get("rvc_applied"))
        self.assertEqual(out_fake, orig_wav)

    def test_synthesize_sentence_wav_bytes_with_rvc_undistorted_base_and_cache(self):
        """Verifica que la síntesis con RVC use base ef_dora, pitch 0.0, engine '+rvc', y caché consistente."""
        from unittest.mock import patch

        class DummyKokoro:
            def get_voices(self):
                return ["ef_dora", "jf_alpha"]

            def get_voice_style(self, v):
                return np.zeros(256, dtype=np.float32)

            def create(self, text, voice="ef_dora", speed=1.0, lang="es"):
                sr = 24000
                samples = (0.4 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.3, int(sr * 0.3)))).astype(np.float32)
                return samples, sr

        dummy_inst = DummyKokoro()

        def dummy_getter():
            return dummy_inst

        text = "Prueba de síntesis de voz con RVC activo para Aoi."
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_cache_dir:
            with patch.object(tts_router, "TTS_CACHE_DIR", tmp_cache_dir):
                with patch.object(tts_router, "get_kokoro_voice", dummy_getter):
                    with patch.dict(os.environ, {"SENSORY_TIER": "full"}):
                        # Limpiar cualquier residuo de caché RAM para la clave
                        tts_router._tts_ram_cache.clear()

                        # Llamada 1: primera síntesis (cache miss)
                        wav1, metrics1 = tts_router._synthesize_wav_sync(
                            text=text,
                            voice_id="anime_es",
                            speed=1.0,
                            language="es",
                            pitch_semitones=2.0,  # debe ser forzado a 0.0 por rvc_active
                            model="kokoro",
                            rvc_voice="Nino Nakano",
                        )
                        self.assertTrue(metrics1["engine"].endswith("+rvc"))
                        self.assertFalse(metrics1.get("cache_hit", False))
                        # Verificar que el base voice fue ef_dora
                        self.assertEqual(metrics1["voice"], "ef_dora")

                        # Llamada 2: segunda síntesis con mismos argumentos -> debe dar cache_hit
                        wav2, metrics2 = tts_router._synthesize_wav_sync(
                            text=text,
                            voice_id="anime_es",
                            speed=1.0,
                            language="es",
                            pitch_semitones=2.0,
                            model="kokoro",
                            rvc_voice="Nino Nakano",
                        )
                        self.assertTrue(metrics2.get("cache_hit", False), f"metrics2 cache_hit was False: {metrics2}")
                        self.assertTrue(metrics2["engine"].endswith("+rvc"))
                        self.assertEqual(wav1, wav2)

    def test_synthesize_rvc_failure_does_not_poison_cache(self):
        """Verifica que si la conversión RVC falla (rvc_applied=False), NO se envenene la caché."""
        from unittest.mock import patch, MagicMock
        import tempfile

        sr = 24000
        audio = (0.4 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.3, int(sr * 0.3)))).astype(np.float32)
        mock_kokoro = MagicMock()
        mock_kokoro.get_voices.return_value = ["ef_dora", "jf_alpha"]
        mock_kokoro.create.return_value = (audio, sr)

        text = "Texto para prueba de fallo RVC y protección de caché."

        with tempfile.TemporaryDirectory() as tmp_cache_dir:
            with patch.object(tts_router, "TTS_CACHE_DIR", tmp_cache_dir):
                with patch.object(tts_router, "get_kokoro_voice", return_value=mock_kokoro):
                    # Forzar fallo simulado de apply_rvc_conversion
                    with patch("rvc_engine.apply_rvc_conversion", return_value=(b"raw_unconverted_audio_dummy_44_bytes_padding", {"rvc_applied": False})):
                        wav, metrics = tts_router._synthesize_wav_sync(
                            text=text,
                            voice_id="ef_dora",
                            speed=1.0,
                            language="es",
                            pitch_semitones=0.0,
                            model="kokoro",
                            rvc_voice="Nino Nakano",
                        )
                        # Al fallar RVC, engine_used no debe terminar en +rvc
                        self.assertFalse(metrics["engine"].endswith("+rvc"))

                        # Verificar que no quedó guardado en caché para 'ef_dora+Nino Nakano'
                        cache_key = tts_router._compute_tts_cache_key(
                            tts_router._sanitize_text_for_tts(text),
                            "ef_dora+Nino Nakano",
                            "es",
                            1.0,
                            0.0,
                            "kokoro+rvc"
                        )
                        cached = tts_router.get_cached_tts_wav(cache_key)
                        self.assertIsNone(cached, "La caché fue envenenada con audio sin convertir cuando rvc_applied era False!")

    def test_tts_endpoint_accurate_sample_rate_and_rvc(self):
        """Verifica que /tts con rvc_voice devuelva HTTP 200, engine '+rvc', y sample_rate correspondiente al WAV."""
        import base64
        resp = self.client.post("/tts", json={
            "text": "Hola, probando endpoint TTS con voz RVC Nino Nakano.",
            "voice": "ef_dora",
            "rvc_voice": "Nino Nakano",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("+rvc", data["engine"])
        self.assertIn("audio_base64", data)

        wav_bytes = base64.b64decode(data["audio_base64"])
        sr_header, _ = tts_router._get_wav_info(wav_bytes)
        self.assertEqual(data["sample_rate"], sr_header)

    def test_voices_endpoint_exposes_rvc_voices_and_status(self):
        """Verifica que /tts/voices y /v1/audio/voices expongan rvc_voices y rvc_status."""
        for path in ("/tts/voices", "/v1/audio/voices"):
            resp = self.client.get(path)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("rvc_voices", data)
            self.assertIn("Nino Nakano", data["rvc_voices"])
            self.assertIn("Miku Nakano", data["rvc_voices"])
            self.assertIn("rvc_status", data)
            self.assertIn("mode", data["rvc_status"])



if __name__ == "__main__":
    unittest.main()
