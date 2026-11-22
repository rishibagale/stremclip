#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "   StreamClip 1-Click Setup (Unix/Mac)  "
echo "========================================"

# 1. Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "[!] FFmpeg not found! Attempting installation..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew install ffmpeg
        else
            echo "[X] Homebrew not found. Please install Homebrew or install FFmpeg manually."
            exit 1
        fi
    elif command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y ffmpeg
    else
        echo "[X] Please install FFmpeg using your system package manager."
        exit 1
    fi
else
    echo "[OK] Found FFmpeg: $(which ffmpeg)"
fi

# 2. Check or install uv
if ! command -v uv &> /dev/null; then
    echo "[*] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "[OK] Found uv"
fi

# 3. Sync dependencies
echo "[*] Installing StreamClip dependencies and Web UI..."
uv sync --extra ui --extra whisper --link-mode copy

echo "========================================"
echo " Setup Complete! Launching StreamClip..."
echo "========================================"
uv run streamclip ui
