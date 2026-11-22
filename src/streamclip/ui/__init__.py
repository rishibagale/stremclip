"""StreamClip Web UI module."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def launch_ui(
    host: str = "localhost",
    port: int = 8501,
    no_browser: bool = False,
) -> int:
    """Launch Streamlit web dashboard."""
    app_path = Path(__file__).parent / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        "true" if no_browser else "false",
        "--theme.base",
        "dark",
    ]
    env = os.environ.copy()
    try:
        proc = subprocess.run(cmd, env=env)
        return proc.returncode
    except KeyboardInterrupt:
        return 0
