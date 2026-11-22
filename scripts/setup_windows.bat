@echo off
setlocal enabledelayedexpansion

echo ========================================
echo    StreamClip Setup (Command Prompt)
echo ========================================

REM 1. Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python not found. Installing Python 3.11 via winget...
    winget install Python.Python.3.11 --accept-package-agreements --accept-source-agreements
) else (
    for /f "tokens=*" %%v in ('python --version') do echo [OK] Found %%v
)

REM 2. Check FFmpeg
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] FFmpeg not found. Installing FFmpeg via winget...
    winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
) else (
    echo [OK] Found FFmpeg
)

REM 3. Check or install uv
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Installing uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
) else (
    echo [OK] Found uv
)

REM 4. Sync dependencies and run UI
echo [*] Installing dependencies and Web UI...
set UV_LINK_MODE=copy
uv sync --extra ui --extra whisper --link-mode copy

echo ========================================
echo  Setup Complete! Launching Web UI...
echo ========================================
uv run streamclip ui
pause
