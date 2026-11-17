"""
speech_utils.py — Local Speech-to-Text (STT) and Text-to-Speech (TTS).

STT:  Uses faster-whisper to transcribe audio locally.
      The Whisper model is loaded ONCE and cached in Streamlit session state
      so it is not reloaded on every UI rerun.

TTS:  Uses pyttsx3 with the espeak-ng backend (Linux).
      If pyttsx3 cannot initialise (e.g., headless server, WSL without audio),
      TTS degrades gracefully — the interview continues as text-only.

Neither module sends audio or text to any external API.

Data flow (STT):
  app.py receives WAV bytes from the browser microphone component
  → speech_utils.transcribe_audio(audio_bytes) is called
  → bytes are written to a temp file
  → faster-whisper transcribes the file
  → plain-text transcript is returned to app.py

Data flow (TTS):
  app.py calls speech_utils.speak_text(question_text)
  → pyttsx3 engine speaks the text through the local audio device
  → returns True/False so the caller knows if TTS worked
"""

import io
import os
import logging
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

# ── STT ───────────────────────────────────────────────────────────────────────

# We keep a module-level reference to the loaded Whisper model so that
# Streamlit reruns do not trigger a fresh model load (which is slow).
_whisper_model = None
_whisper_model_size: Optional[str] = None


def get_whisper_model(model_size: str = "base"):
    """Return a cached WhisperModel instance, loading it on first call.

    Args:
        model_size: One of "tiny", "base", "small", "medium", "large-v2".
                    Defaults to "base" (≈145 MB, good for laptops).

    Returns:
        A faster_whisper.WhisperModel instance.

    Raises:
        RuntimeError: if the model cannot be loaded (e.g., no internet for
                      first download, or out of memory).
    """
    global _whisper_model, _whisper_model_size

    # If the requested model size changed, reload.
    if _whisper_model is None or _whisper_model_size != model_size:
        logger.info("Loading Whisper model '%s' …", model_size)
        try:
            from faster_whisper import WhisperModel  # lazy import

            # Use CUDA if an NVIDIA GPU is available; otherwise fall back to CPU.
            # int8 quantisation on CPU makes inference much faster.
            device = "cuda" if _cuda_available() else "cpu"
            compute = "float16" if device == "cuda" else "int8"

            _whisper_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute,
            )
            _whisper_model_size = model_size
            logger.info("Whisper model loaded on %s (%s)", device, compute)
        except Exception as exc:
            logger.error("Failed to load Whisper model: %s", exc)
            raise RuntimeError(f"Could not load Whisper '{model_size}' model: {exc}") from exc

    return _whisper_model


def _cuda_available() -> bool:
    """Check whether a CUDA-capable GPU is usable by PyTorch/CTranslate2."""
    try:
        import ctranslate2  # faster-whisper depends on this
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def transcribe_audio(audio_bytes: bytes, model_size: str = "base", language: str = "en") -> str:
    """Transcribe raw WAV audio bytes using local faster-whisper.

    Args:
        audio_bytes: Raw audio data (WAV format expected from the browser).
        model_size:  Whisper model size to use.
        language:    BCP-47 language code. "en" for English.

    Returns:
        The transcribed text as a single string, stripped of leading/trailing
        whitespace. Returns an empty string if transcription produces nothing.

    Raises:
        RuntimeError: propagated from get_whisper_model() if the model fails.
        ValueError:   if audio_bytes is empty or None.
    """
    if not audio_bytes:
        raise ValueError("No audio data received. Please record a non-empty answer.")

    model = get_whisper_model(model_size)

    # Write audio bytes to a temporary file because faster-whisper expects a
    # file path (or a NumPy array), not raw bytes.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, _info = model.transcribe(
            tmp_path,
            language=language,
            beam_size=5,
            vad_filter=True,         # skip silent sections
            vad_parameters={"min_silence_duration_ms": 500},
        )
        # Concatenate all segment texts into one string.
        transcript = " ".join(seg.text.strip() for seg in segments).strip()
        logger.info("Transcription complete: %d chars", len(transcript))
        return transcript
    except Exception as exc:
        logger.error("Transcription failed: %s", exc)
        raise RuntimeError(f"Speech recognition failed: {exc}") from exc
    finally:
        # Always clean up the temp file.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def whisper_model_status(model_size: str = "base") -> dict:
    """Return a status dict for the setup-check display.

    Returns:
        {"status": "ok"|"error", "message": str, "device": str}
    """
    try:
        model = get_whisper_model(model_size)
        device = "cuda" if _cuda_available() else "cpu"
        return {"status": "ok", "message": f"Model '{model_size}' ready on {device}", "device": device}
    except RuntimeError as e:
        return {"status": "error", "message": str(e), "device": "unknown"}


# ── TTS ───────────────────────────────────────────────────────────────────────

_tts_engine = None          # cached pyttsx3 engine
_tts_available: Optional[bool] = None   # None = not yet tested


def _init_tts():
    """Attempt to initialise the pyttsx3 engine once.

    In WSL or headless environments, pyttsx3 may fail to find an audio device.
    We catch that and set _tts_available=False so the rest of the app can
    degrade gracefully.
    """
    global _tts_engine, _tts_available

    if _tts_available is not None:
        return  # already initialised (or already known to have failed)

    try:
        import pyttsx3
        engine = pyttsx3.init()

        # Configure voice properties for a clear interview-style delivery.
        engine.setProperty("rate", 160)    # words per minute (default ~200)
        engine.setProperty("volume", 0.9)  # 0.0–1.0

        _tts_engine = engine
        _tts_available = True
        logger.info("pyttsx3 TTS engine initialised successfully")
    except Exception as exc:
        logger.warning("TTS init failed (text-only mode): %s", exc)
        _tts_engine = None
        _tts_available = False


def is_tts_available() -> bool:
    """Return True if TTS can produce speech on this machine."""
    _init_tts()
    return bool(_tts_available)


def speak_text(text: str) -> bool:
    """Speak the given text aloud using the local TTS engine.

    Args:
        text: The text to speak. Should be reasonably short (one question).

    Returns:
        True if speech was produced, False if TTS is unavailable or failed.

    Note:
        pyttsx3 is synchronous — this call blocks until speech finishes.
        Because Streamlit reruns re-execute Python top-to-bottom, we rely on
        the caller (app.py) to gate TTS with a session-state flag so the same
        question is not repeated on every rerun.
    """
    _init_tts()

    if not _tts_available or _tts_engine is None:
        logger.debug("TTS unavailable — skipping speech for: %s", text[:60])
        return False

    try:
        _tts_engine.say(text)
        _tts_engine.runAndWait()
        return True
    except Exception as exc:
        logger.warning("TTS speak_text failed: %s", exc)
        return False


def tts_status() -> dict:
    """Return a status dict for the setup-check display."""
    _init_tts()
    if _tts_available:
        return {"status": "ok", "message": "pyttsx3 TTS engine ready"}
    else:
        return {
            "status": "warning",
            "message": (
                "TTS unavailable — likely no audio device (WSL/headless). "
                "Interview works in text-only mode. "
                "Install espeak-ng: sudo apt install espeak-ng"
            ),
        }
