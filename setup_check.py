"""
setup_check.py — Environment and dependency verification script.

Run this before starting the application to confirm everything is configured:

    python setup_check.py

Each check is classified as:
  PASS    — fully working
  WARN    — degraded but non-fatal (e.g., TTS unavailable in headless env)
  FAIL    — blocking issue that must be fixed before the app works
"""

import sys
import os
import importlib
import sqlite3
import subprocess

# ── Helpers ───────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}[PASS]{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}[WARN]{RESET} {msg}")
def fail(msg): print(f"  {RED}[FAIL]{RESET} {msg}")

def header(title):
    print(f"\n{BOLD}{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}{RESET}")

issues = []   # collect blocking failures

# ── 1. Python version ─────────────────────────────────────────────────────────
header("1. Python Environment")
major, minor = sys.version_info[:2]
version_str = f"Python {major}.{minor}.{sys.version_info[2]}"
if major == 3 and minor >= 11:
    ok(f"{version_str} (compatible)")
elif major == 3 and minor == 10:
    warn(f"{version_str} — 3.11+ recommended; should still work")
else:
    fail(f"{version_str} — Python 3.11 or 3.12 required")
    issues.append("Python version")

# ── 2. Required packages ──────────────────────────────────────────────────────
header("2. Python Packages")
REQUIRED_PACKAGES = {
    "streamlit":               "streamlit",
    "requests":                "requests",
    "dotenv":                  "python-dotenv",
    "fitz":                    "PyMuPDF",
    "docx":                    "python-docx",
    "faster_whisper":          "faster-whisper",
    "pyttsx3":                 "pyttsx3",
    "streamlit_mic_recorder":  "streamlit-mic-recorder",
    "numpy":                   "numpy",
    "ollama":                  "ollama",
}

for import_name, pip_name in REQUIRED_PACKAGES.items():
    try:
        importlib.import_module(import_name)
        ok(f"{pip_name}")
    except ImportError:
        fail(f"{pip_name} — install with: pip install {pip_name}")
        issues.append(pip_name)

# ── 3. Ollama connectivity ────────────────────────────────────────────────────
header("3. Ollama Service")
try:
    from dotenv import load_dotenv
    load_dotenv()
    import requests

    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "gemma2:2b")

    resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    resp.raise_for_status()
    models = [m["name"] for m in resp.json().get("models", [])]
    ok(f"Ollama is running at {OLLAMA_BASE_URL}")

    if any(m == OLLAMA_MODEL or m.startswith(OLLAMA_MODEL.split(":")[0]) for m in models):
        ok(f"Model '{OLLAMA_MODEL}' is available")
    else:
        fail(f"Model '{OLLAMA_MODEL}' not found. Pull it: ollama pull {OLLAMA_MODEL}")
        fail(f"Available models: {', '.join(models) if models else 'none'}")
        issues.append(f"Ollama model '{OLLAMA_MODEL}'")

except Exception as e:
    fail(f"Cannot reach Ollama: {e}")
    fail("Start Ollama with: ollama serve")
    issues.append("Ollama service")

# ── 4. Whisper STT ────────────────────────────────────────────────────────────
header("4. Whisper STT (faster-whisper)")
whisper_model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
try:
    from faster_whisper import WhisperModel

    ok(f"faster-whisper imported successfully")
    print(f"  {YELLOW}[INFO]{RESET} Skipping model load (takes 5-30s). "
          f"Will load '{whisper_model_size}' model on first use.")

    # Check CUDA
    try:
        import ctranslate2
        cuda_count = ctranslate2.get_cuda_device_count()
        if cuda_count > 0:
            ok(f"CUDA available ({cuda_count} device(s)) — will use GPU for faster STT")
        else:
            warn("No CUDA GPU — will use CPU (int8). This is fine for 'base' model.")
    except Exception:
        warn("Could not check CUDA — assuming CPU.")

except ImportError as e:
    fail(f"faster-whisper not available: {e}")
    issues.append("faster-whisper")

# ── 5. TTS ────────────────────────────────────────────────────────────────────
header("5. Text-to-Speech (pyttsx3 / espeak-ng)")
try:
    import pyttsx3
    ok("pyttsx3 imported successfully")

    # Check for espeak-ng system binary
    result = subprocess.run(["which", "espeak-ng"], capture_output=True, text=True)
    if result.returncode == 0:
        ok(f"espeak-ng found at {result.stdout.strip()}")
    else:
        warn("espeak-ng not found in PATH")
        warn("Install with: sudo apt install espeak-ng")
        warn("TTS will be unavailable — interview continues in text mode")

    # Try actually initialising the engine
    try:
        engine = pyttsx3.init()
        engine.stop()
        ok("pyttsx3 engine initialised successfully")
    except Exception as e:
        warn(f"pyttsx3 engine init failed (likely no audio device): {e}")
        warn("This is expected in WSL / headless environments.")
        warn("Interview will work in text-only mode.")

except ImportError:
    warn("pyttsx3 not installed — TTS unavailable")
    warn("Install with: pip install pyttsx3")

# ── 6. Database ───────────────────────────────────────────────────────────────
header("6. SQLite Database")
db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
try:
    os.makedirs(db_dir, exist_ok=True)
    ok(f"data/ directory writable: {db_dir}")

    # Test database creation
    test_db = os.path.join(db_dir, "test_check.db")
    conn = sqlite3.connect(test_db)
    conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    row = conn.execute("SELECT id FROM t").fetchone()
    conn.close()
    os.unlink(test_db)

    if row and row[0] == 1:
        ok("SQLite read/write working correctly")
    else:
        fail("SQLite test failed — unexpected data")
        issues.append("SQLite")
except Exception as e:
    fail(f"SQLite error: {e}")
    issues.append("SQLite")

# ── 7. Microphone (browser-based) ─────────────────────────────────────────────
header("7. Microphone (browser-based recording)")
try:
    from streamlit_mic_recorder import mic_recorder
    ok("streamlit-mic-recorder imported successfully")
    print(f"  {YELLOW}[INFO]{RESET} Browser microphone access is required at runtime.")
    print(f"  {YELLOW}[INFO]{RESET} Grant permission in your browser when prompted.")
    print(f"  {YELLOW}[INFO]{RESET} In WSL, the browser accesses the Windows microphone directly.")
except ImportError:
    warn("streamlit-mic-recorder not installed")
    warn("Install: pip install streamlit-mic-recorder")
    warn("The app will fall back to text-only input mode.")

# ── 8. System audio (for TTS playback) ────────────────────────────────────────
header("8. System Audio")
result = subprocess.run(["aplay", "-l"], capture_output=True, text=True)
if result.returncode == 0 and result.stdout.strip():
    ok("Audio output devices found")
else:
    warn("No audio output detected (common in WSL)")
    warn("TTS will be disabled — interview works in text mode")

# ── 9. Project files ──────────────────────────────────────────────────────────
header("9. Project Files")
required_files = ["app.py", "interview_engine.py", "speech_utils.py", "database.py",
                  "requirements.txt", ".env.example"]
base_dir = os.path.dirname(os.path.abspath(__file__))
for fname in required_files:
    fpath = os.path.join(base_dir, fname)
    if os.path.exists(fpath):
        ok(f"{fname} exists")
    else:
        fail(f"{fname} missing!")
        issues.append(fname)

# Check .env file
env_path = os.path.join(base_dir, ".env")
if os.path.exists(env_path):
    ok(".env exists (environment configured)")
else:
    warn(".env not found — copy .env.example to .env and configure it")
    warn("Using defaults: OLLAMA_MODEL=gemma2:2b, WHISPER_MODEL_SIZE=base")

# ── Summary ───────────────────────────────────────────────────────────────────
header("Summary")
if not issues:
    print(f"\n  {GREEN}{BOLD}✅ All checks passed! You can start the app:{RESET}")
    print(f"\n     streamlit run app.py\n")
else:
    print(f"\n  {RED}{BOLD}❌ {len(issues)} blocking issue(s) found:{RESET}")
    for issue in issues:
        print(f"     • {issue}")
    print(f"\n  Fix the issues above, then run: streamlit run app.py\n")
