# streamclip

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-blue)](http://mypy-lang.org/)

**streamclip** is a high-performance, multimodal livestream VOD highlight extractor designed for Python 3.11+. It ingests full livestream VODs and chat replay streams, identifies peak action moments across audio and chat, enriches candidate windows with visual motion analysis and speech transcription, and renders vertical 1080x1920 highlight clips ready for TikTok, YouTube Shorts, and Instagram Reels.

Initial target: **Grand Theft Auto VI (GTA VI)** livestream broadcasts.

---

## Features

- **Multimodal Signal Fusion**: Synchronously fuses per-second audio RMS energy, spectral flux, and smoothed hype-weighted chat velocity into a unified highlight signal.
- **Dynamic Candidate Windowing**: Snaps start boundaries to speech/silence points, tracks post-peak energy decay, clamps durations to mobile-friendly windows (15–60s), and merges overlapping moments.
- **Candidate-Gated Enrichment**: Decodes visual frames and transcribes speech strictly on candidate windows (+ margin), avoiding full-VOD processing overhead.
- **Standalone Quality Reranking**: Evaluates hook strength, climax, and viral potential via Claude 3.5 Sonnet (with retries and strict JSON output) or component-based heuristic scoring with greedy Non-Maximum Suppression (NMS) deduplication.
- **Vertical Rendering Engine**: Generates 1080x1920 vertical video layouts (`stack`, `gameplay_only`, `facecam_only`), burns in ASS subtitles with active word-level highlighting and mobile-safe margins, applies EBU R128 loudness normalization, and encodes with NVENC or x264.
- **Full Resumability**: Caches intermediate timeseries, transcription slices, visual scores, and rendered video files keyed by content and config hash.

---

## Installation

### Prerequisites
- **Python**: `>= 3.11`
- **FFmpeg**: `ffmpeg` and `ffprobe` must be installed on your system PATH with `libass` support.

### Installing with uv or pip
```bash
# Clone the repository
git clone https://github.com/streamclip/streamclip.git
cd streamclip

# Install core package using uv
uv sync

# Or with optional extras:
# [whisper]  - faster-whisper speech transcription
# [audio-ml] - PyTorch + torchaudio for YAMNet audio classification
# [vision]   - Computer vision tools
uv pip install -e ".[whisper,audio-ml]"
```

---

## Quickstart

### 1. Run End-to-End Extraction Pipeline
Extract, rerank, and render highlight clips from a VOD and chat log:
```bash
streamclip run vod.mp4 --chat chat.csv --out-dir output/
```

### 2. Dry Run & LLM Cost Estimation
Inspect the candidate detection plan, highlight windows, and estimated Anthropic API cost without encoding video:
```bash
streamclip run vod.mp4 --chat chat.csv --dry-run
```

### 3. Step-by-Step Workflow: Detect then Render
```bash
# Step 1: Detect highlight candidate windows (outputs candidates.json)
streamclip detect vod.mp4 --chat chat.csv -o candidates.json

# Step 2: Render vertical highlight clips from candidates.json
streamclip render candidates.json --video vod.mp4 --layout stack -o output/
```

### 4. Evaluate Against Ground Truth
Evaluate candidate highlights against manual ground truth labels (`labels.json`):
```bash
streamclip eval -l labels.json -v vod.mp4 -c chat.csv -k 5
```

### 5. Tune Hyperparameters
Run random or grid search optimization over fusion weights and peak gap constraints:
```bash
streamclip tune -l labels.json -v vod.mp4 -c chat.csv -o configs/tuned.yaml -m random --trials 30
```

---

## How Detection Works

```
VOD & Chat Ingest
      │
      ├──> Audio Signal (16kHz RMS + Spectral Flux) ──┐
      └──> Chat Signal (Messages/Sec + Hype Weights)  ──┴─> Multimodal Fusion (Rolling Z-Score)
                                                                      │
                                                               Peak Detection
                                                                      │
                                                            Candidate Windowing
                                                      (Boundary Snapping & Decay)
                                                                      │
                                                           Candidate Windows Only
                                                      ┌───────────────┴───────────────┐
                                                      ▼                               ▼
                                            Motion Visual Scoring             Whisper STT
                                                      └───────────────┬───────────────┘
                                                                      │
                                                          LLM / Heuristic Rerank
                                                                      │
                                                            Greedy NMS Dedupe
                                                                      │
                                                          Vertical Video Render
                                                      (1080x1920 + ASS Captions)
```

1. **Signal Extraction**:
   - **Audio Track**: Extracts 16 kHz mono audio, computing per-second Root-Mean-Square (RMS) amplitude and spectral flux (spectral energy difference between consecutive frames).
   - **Chat Track**: Computes per-second message density, applying regex hype bonuses (e.g. `LUL`, `KEKW`, `POG`, `WTF`, `CLIP IT`, caps ratio, exclamation marks, and cross-user repeated spam).
2. **Multimodal Signal Fusion**:
   - Computes rolling baseline z-scores ($z = (x - \mu) / \sigma$ or robust median/MAD) over a 300s window.
   - Scales track contributions by track activity confidence (e.g. low chat volume automatically de-weights chat).
   - Convolves with a 1D Gaussian kernel for smoothing and identifies peaks via `scipy.signal.find_peaks`.
3. **Candidate Windowing**:
   - Extends backwards by `pre_roll_s` before the peak, snapping the start to the nearest audio silence point or speech word boundary.
   - Extends forward past the peak until the fused signal decays back near baseline plus `post_roll_s`.
   - Clamps duration to `[clip_min_s, clip_max_s]` (default 15–60s) and merges overlapping windows.
4. **Candidate-Gated Enrichment**:
   - Fast ffmpeg input seeking samples 1 frame per 1–2s within candidate windows to calculate frame-difference motion and scene cuts.
   - Targeted Whisper speech-to-text transcribes only the candidate windows.
5. **Reranking & Deduplication**:
   - Claude 3.5 Sonnet (or Heuristic component scoring) evaluates hook quality, viral title, and punchline reason.
   - Greedy Non-Maximum Suppression (NMS) suppresses near-duplicate clips ($IoU \ge 0.5$ or peak distance $< 10.0$s).
6. **Vertical Rendering**:
   - Filtergraph creates vertical 1080x1920 output with selected layout (`gameplay_only`, `stack`, `facecam_only`).
   - Burns in ASS word-level highlighted subtitles with mobile-safe margins (420px vertical padding).
   - Normalizes audio loudness to -16 LUFS with EBU R128 (`loudnorm`) and faststart flags.

---

## Configuration Reference

`streamclip` configurations are specified in YAML:

```yaml
# Weights for multimodal signal fusion
fusion_weights:
  chat: 0.5        # Weight given to chat message density and hype
  audio: 0.3       # Weight given to audio RMS energy and spectral flux
  vision: 0.2      # Weight given to visual motion intensity

# Peak detection parameters
peak_height: 1.0       # Minimum fused z-score height to register as a highlight peak
min_peak_gap_s: 20.0   # Minimum separation in seconds between distinct peaks

# Candidate window duration constraints
clip_min_s: 15.0       # Minimum clip duration in seconds
clip_max_s: 60.0       # Maximum clip duration in seconds (TikTok/Shorts limit)
pre_roll_s: 15.0       # Anticipation lead-up duration before the peak moment
post_roll_s: 4.0        # Buffer duration after signal decays back to baseline

# Selection and resource budgets
top_k: 20                  # Maximum number of highlight clips to select
vision_budget_windows: 40  # Maximum number of candidate windows to score with vision
```

---

## Evaluation Workflow

To benchmark candidate detection accuracy against ground truth annotations:

1. Prepare `labels.json` with ground truth highlight intervals:
   ```json
   [
     {"t_start": 50.0, "t_end": 70.0},
     {"t_start": 170.0, "t_end": 190.0},
     {"t_start": 290.0, "t_end": 310.0}
   ]
   ```
2. Run evaluation:
   ```bash
   streamclip eval --labels labels.json --video vod.mp4 --chat chat.csv --top-k 5
   ```
3. Metrics computed:
   - **Recall@K**: Proportion of ground truth moments detected in top-K clips (hit criteria: $\ge 50\%$ window overlap or peak inside ground truth).
   - **Precision@K**: Proportion of top-K candidate clips that represent true highlights.
   - **Mean Time Offset**: Average temporal distance (in seconds) between candidate peaks and ground truth centers.

---

## Known Limits & Considerations

- **FFmpeg Dependency**: FFmpeg must be installed on your system with `libass` and `loudnorm` filter support.
- **NVENC Hardware Acceleration**: Hardware-accelerated H.264 encoding requires a compatible NVIDIA GPU and driver. If unavailable, `streamclip` automatically falls back to CPU-based `libx264`.
- **Chat Replay Formats**: Auto-detects TwitchDownloader JSON, chat-downloader JSON, and CSV (`t,user,text`). Non-standard JSON formats may require conversion.
- **Whisper Speech Transcription**: Requires optional dependency `[whisper]` (`pip install 'streamclip[whisper]'`). If not installed, rendering proceeds without speech captions.

---

## Development & Testing

```bash
# Run pytest test suite
make test

# Run linting and code formatting checks
make lint

# Run strict mypy type checking
make type
```
