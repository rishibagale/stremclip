# StreamClip 1-Click Setup Script for Windows
$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   StreamClip 1-Click Setup (Windows)   " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. Check Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[!] Python not found. Please install Python 3.11+ (winget install Python.Python.3.11)" -ForegroundColor Yellow
} else {
    $pyVer = python --version
    Write-Host "[OK] Found $pyVer" -ForegroundColor Green
}

# 2. Check FFmpeg
$ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpegCmd) {
    Write-Host "[!] FFmpeg not found! Attempting automated installation via winget..." -ForegroundColor Yellow
    try {
        winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
        Write-Host "[OK] FFmpeg installed successfully! You may need to restart your terminal." -ForegroundColor Green
    } catch {
        Write-Host "[X] Could not auto-install FFmpeg. Please install manually: winget install Gyan.FFmpeg" -ForegroundColor Red
    }
} else {
    Write-Host "[OK] Found FFmpeg: $($ffmpegCmd.Source)" -ForegroundColor Green
}

# 3. Check or install uv
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCmd) {
    Write-Host "[*] Installing uv (ultra-fast package manager)..." -ForegroundColor Cyan
    irm https://astral.sh/uv/install.ps1 | iex
    $env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
} else {
    Write-Host "[OK] Found uv" -ForegroundColor Green
}

# 4. Sync dependencies
Write-Host "[*] Installing StreamClip dependencies and Web UI..." -ForegroundColor Cyan
$env:UV_LINK_MODE = "copy"
uv sync --extra ui --extra whisper --link-mode copy

Write-Host "========================================" -ForegroundColor Green
Write-Host " Setup Complete! Launching StreamClip..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
uv run streamclip ui
