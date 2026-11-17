#!/usr/bin/env bash
# setup.sh — One-step environment setup for Voice Interview Simulator
#
# This script:
#   1. Checks Python version
#   2. Creates a virtual environment (if not already present)
#   3. Installs Python dependencies
#   4. Copies .env.example to .env (if .env doesn't exist)
#   5. Runs setup_check.py to verify the environment
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# NOTE: This script does NOT use sudo. Install system packages manually first:
#   sudo apt install -y python3 python3-venv python3-pip \
#       build-essential portaudio19-dev espeak-ng ffmpeg

set -e   # exit on first error

BOLD="\033[1m"
GREEN="\033[92m"
YELLOW="\033[93m"
RED="\033[91m"
RESET="\033[0m"

info()  { echo -e "${BOLD}[INFO]${RESET}  $*"; }
ok()    { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}   $*"; }
fail()  { echo -e "${RED}[FAIL]${RESET}   $*"; exit 1; }

echo -e "${BOLD}======================================================"
echo -e "  Voice Interview Simulator — Environment Setup"
echo -e "======================================================${RESET}"
echo ""

# ── 1. Python version ─────────────────────────────────────────────────────────
info "Checking Python version…"
if ! command -v python3 &>/dev/null; then
    fail "python3 not found. Install with: sudo apt install python3"
fi
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
ok "Python $PY_VERSION found"
if python3 -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)"; then
    ok "Python version is compatible"
else
    fail "Python 3.10+ required. Current: $PY_VERSION"
fi

# ── 2. Virtual environment ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

if [ -d "$VENV_DIR" ]; then
    ok "Virtual environment already exists at $VENV_DIR"
else
    info "Creating virtual environment at $VENV_DIR…"
    python3 -m venv "$VENV_DIR"
    ok "Virtual environment created"
fi

# Activate
source "$VENV_DIR/bin/activate"
ok "Virtual environment activated"

# ── 3. Upgrade pip ────────────────────────────────────────────────────────────
info "Upgrading pip…"
pip install --upgrade pip --quiet
ok "pip upgraded"

# ── 4. Install dependencies ───────────────────────────────────────────────────
info "Installing Python dependencies (this may take 2-5 minutes on first run)…"
pip install -r "$SCRIPT_DIR/requirements.txt"
ok "Dependencies installed"

# ── 5. Environment configuration ─────────────────────────────────────────────
ENV_FILE="$SCRIPT_DIR/.env"
ENV_EXAMPLE="$SCRIPT_DIR/.env.example"

if [ -f "$ENV_FILE" ]; then
    ok ".env already exists — skipping copy"
else
    if [ -f "$ENV_EXAMPLE" ]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        ok ".env created from .env.example"
        warn "Edit .env to set your OLLAMA_MODEL if needed"
    else
        warn ".env.example not found — please create .env manually"
    fi
fi

# ── 6. Data directory ─────────────────────────────────────────────────────────
mkdir -p "$SCRIPT_DIR/data"
ok "data/ directory ready"

# ── 7. Ollama check ───────────────────────────────────────────────────────────
echo ""
info "Checking Ollama…"
if command -v ollama &>/dev/null; then
    ok "Ollama is installed ($(ollama --version 2>/dev/null || echo 'version unknown'))"
    if curl -s --max-time 3 http://localhost:11434/api/tags &>/dev/null; then
        ok "Ollama service is running"
    else
        warn "Ollama service is NOT running"
        warn "Start it with: ollama serve"
    fi
else
    warn "Ollama is not installed"
    warn "Install from: https://ollama.com/download/linux"
    warn "Or run: curl -fsSL https://ollama.com/install.sh | sh"
fi

# ── 8. Verification ───────────────────────────────────────────────────────────
echo ""
info "Running setup verification…"
python "$SCRIPT_DIR/setup_check.py" || true   # don't fail the script on check errors

echo ""
echo -e "${BOLD}======================================================"
echo -e "  Setup complete!"
echo -e "======================================================"
echo -e "${RESET}"
echo "To start the application:"
echo ""
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""
echo "Then open http://localhost:8501 in your browser."
echo ""
