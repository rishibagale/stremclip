"""StreamClip Web UI Dashboard (Streamlit).

SupoClip-inspired intuitive studio for livestream VOD ingestion,
real-time download HUD (speed, %, ETA), transparent clip discovery,
4D virality cards gallery, auto-facecam framing, and 1080x1920 vertical rendering.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import streamlit as st

from streamclip.ingest.video import DownloadProgress, download_with_progress


def check_dependencies() -> dict[str, bool]:
    """Check availability of external and optional tools."""
    return {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "ytdlp": shutil.which("yt-dlp") is not None,
    }


def run_app() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(
        page_title="StreamClip — Viral Shorts Creator",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom styling inspired by SupoClip & OpusClip dark theme
    st.markdown(
        """
        <style>
        .hero-box {
            background: linear-gradient(135deg, #1e1e2e 0%, #252538 100%);
            border: 1px solid #3b3b54;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
        }
        .hud-card {
            background-color: #1a1a2e;
            border: 1px solid #4f46e5;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 16px;
        }
        .clip-card {
            background-color: #1e1e2f;
            border: 1px solid #312e81;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 16px;
            transition: transform 0.2s;
        }
        .clip-card:hover {
            border-color: #6366f1;
        }
        .score-pill {
            background-color: #312e81;
            color: #a5b4fc;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
        }
        .explainer-box {
            background-color: #13131f;
            border-left: 4px solid #10b981;
            border-radius: 6px;
            padding: 14px 18px;
            margin: 15px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state
    if "candidates" not in st.session_state:
        st.session_state.candidates = []
    if "words" not in st.session_state:
        st.session_state.words = []
    if "selected_candidate_idx" not in st.session_state:
        st.session_state.selected_candidate_idx = 0
    if "active_video_path" not in st.session_state:
        st.session_state.active_video_path = "carry_clip1_hd.mp4"
    if "download_status" not in st.session_state:
        st.session_state.download_status = None

    deps = check_dependencies()

    # Sidebar: System telemetry & output location
    with st.sidebar:
        st.markdown("### 🎬 **StreamClip Studio**")
        st.caption("Open-source OpusClip & SupoClip alternative")
        st.divider()

        st.markdown("#### ⚙️ **System Engine**")
        if deps["ffmpeg"]:
            st.success("FFmpeg: Ready", icon="✅")
        else:
            st.error("FFmpeg: Missing", icon="❌")
            st.caption("Install via: `winget install Gyan.FFmpeg`")

        if deps["ytdlp"]:
            st.success("yt-dlp: Ready", icon="✅")
        else:
            st.warning("yt-dlp: Not on PATH", icon="⚠️")

        st.divider()
        st.markdown("#### 📁 **Where Clips Are Saved**")
        output_dir = st.text_input(
            "Output Directory on Disk",
            value=str(Path("output/web_renders").resolve()),
            help="Full path on your computer where rendered clips are saved.",
        )
        st.caption(f"📂 Folder: `{output_dir}`")

        st.divider()
        st.markdown("#### 🎛️ **Default Settings**")
        default_layout = st.selectbox(
            "Layout",
            options=["stack", "gameplay_only"],
            index=0,
            help="stack: Top Facecam / Bottom Gameplay; gameplay_only: Full frame",
        )
        default_facecam = st.selectbox(
            "Facecam Detection",
            options=["auto", "bottom-left", "bottom-right", "top-left", "top-right"],
            index=0,
            help="auto: automatically tracks streamer webcam across frames",
        )

    # MAIN CONTENT

    # Section 1: Hero Stream / VOD Link Ingestion Bar
    st.markdown(
        """
        <div class="hero-box">
            <h2 style='margin-top:0;'>🎥 Where to Paste Your Stream or Video Link</h2>
            <p style='color:#a1a1aa; margin-bottom:12px;'>
                Paste any live stream or VOD URL to automatically find high-retention highlights,
                detect your webcam, and burn viral hook banners and word-level captions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_hero_input, col_hero_btn = st.columns([4, 1])

    with col_hero_input:
        input_mode = st.radio(
            "Source Type",
            options=["🌐 Stream URL (YouTube / Twitch / Kick)", "📁 Local Video File on Disk"],
            horizontal=True,
        )

        if "Stream URL" in input_mode:
            stream_url = st.text_input(
                "Paste Video / Stream Link Here",
                placeholder="https://www.youtube.com/watch?v=... or https://twitch.tv/videos/...",
                label_visibility="collapsed",
            )
            st.caption(
                "Supported: 🔴 **YouTube Streams/VODs** | 💜 **Twitch Streams/VODs** | 🟢 **Kick**"
            )
        else:
            stream_url = ""
            local_path = st.text_input(
                "Local Video File Path",
                value=st.session_state.active_video_path,
                placeholder="e.g. carry_clip1_hd.mp4 or C:/Videos/stream.mp4",
                label_visibility="collapsed",
            )
            st.caption(
                "Supported formats: **.mp4**, **.mkv**, **.mov**, **.ts** (local processing)"
            )

    with col_hero_btn:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        find_btn = st.button("⚡ Find Viral Clips", type="primary", use_container_width=True)

    # Demo Quick-Loader
    col_demo1, col_demo2 = st.columns([3, 1])
    with col_demo1:
        st.caption("💡 Want a fast test? Click below to instantly load demo highlight candidates:")
    with col_demo2:
        if st.button("⚡ Load Demo VOD (CarryisLive)", use_container_width=True):
            cand_files = ["hd_clip1_candidate.json", "hd_candidates.json", "candidates.json"]
            for cf in cand_files:
                p = Path(cf)
                if p.is_file():
                    try:
                        with open(p, encoding="utf-8") as f:
                            st.session_state.candidates = json.load(f)
                            st.session_state.active_video_path = "carry_clip1_hd.mp4"
                            st.success(
                                f"Loaded {len(st.session_state.candidates)} demo highlights!"
                            )
                            break
                    except Exception:
                        pass
            if Path("clip1_words.json").is_file():
                try:
                    with open("clip1_words.json", encoding="utf-8") as f:
                        st.session_state.words = json.load(f)
                except Exception:
                    pass

    # SECTION 2: Real-Time Download HUD (Speed, %, ETA)
    if find_btn and "Stream URL" in input_mode and stream_url:
        st.markdown("### 📥 Live Download & Ingestion HUD")
        hud_placeholder = st.empty()
        progress_bar = st.progress(0.0)

        dest_dir = Path("output/downloads")
        dest_dir.mkdir(parents=True, exist_ok=True)
        target_video = dest_dir / "downloaded_stream.mp4"

        def update_progress(prog: DownloadProgress) -> None:
            pct = min(100.0, max(0.0, prog.percent)) / 100.0
            progress_bar.progress(pct)
            with hud_placeholder.container():
                c_pct, c_spd, c_size, c_eta = st.columns(4)
                c_pct.metric("Download Progress", f"{prog.percent:.1f}%")
                c_spd.metric("Speed (MB/s)", prog.speed_str)
                c_size.metric("Total Size", prog.total_str)
                c_eta.metric("Estimated Time Left", prog.eta_str)

        try:
            with st.spinner("Connecting to stream and starting high-speed acquisition..."):
                download_with_progress(stream_url, target_video, progress_callback=update_progress)
            st.success(f"Download complete! Saved to `{target_video}`")
            st.session_state.active_video_path = str(target_video)
        except Exception as e:
            st.error(f"Download error: {e}")

    elif find_btn and "Local Video" in input_mode:
        if Path(local_path).is_file():
            st.session_state.active_video_path = local_path
            st.success(f"Loaded local video: `{local_path}`")
        else:
            st.error(f"File not found: `{local_path}`")

    st.divider()

    # SECTION 3: Explainer Box — Where It Finds Clips & How It Works
    with st.expander(
        "💡 **How StreamClip Discovers Clips & Where They Are Saved**", expanded=False
    ):
        st.markdown(
            f"""
            <div class="explainer-box">
                <h4>🧠 How the Multimodal Detection Pipeline Works</h4>
                <ul>
                    <li>🔊 <b>Audio Climax Engine</b>: Analyzes 1 Hz RMS volume to find
                    screams, laughter, and decibel spikes.</li>
                    <li>💬 <b>Chat Velocity Spikes</b>: Parses chat replay messages/sec to spot
                    viewer hype and emote bursts.</li>
                    <li>🎮 <b>Visual Motion</b>: Detects rapid scene changes and action.</li>
                    <li>📊 <b>4D Virality Model (SupoClip)</b>: Evaluates each moment across:
                        <b>Hook</b> (0-100), <b>Engagement</b> (0-100), <b>Value</b> (0-100),
                        and <b>Shareability</b> (0-100).
                    </li>
                    <li>👤 <b>Automatic Facecam Framing</b>: Automatically tracks webcam box.</li>
                </ul>
                <hr>
                <h4>📁 Where Your Rendered Clips Are Stored</h4>
                <p>All finished vertical 1080x1920 MP4 clips are written to:</p>
                <code>{output_dir}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # SECTION 4: Discovered Clips Grid (SupoClip / OpusClip Style)
    st.markdown("### 🏆 Discovered Highlight Clips")

    if not st.session_state.candidates:
        st.info(
            "Paste a stream link above or click **'⚡ Load Demo VOD'** to view discovered clips."
        )
    else:
        candidates = st.session_state.candidates
        st.write(f"Showing **{len(candidates)}** viral highlight moments sorted by virality score:")

        # Render in 3-column responsive card grid
        cols = st.columns(3)
        for i, cand in enumerate(candidates):
            col = cols[i % 3]
            start_s = float(cand.get("start", 0.0))
            end_s = float(cand.get("end", 30.0))
            dur_s = end_s - start_s
            hook_title = cand.get("hook_title") or "WAIT FOR THE SCREAM! 💀"
            score = float(cand.get("score", 2.5))
            hook_score = float(cand.get("hook_score", 92.0))
            eng_score = float(cand.get("engagement_score", 94.0))

            with col:
                v_score = min(100.0, score * 20)
                st.markdown(
                    f"""
                    <div class="clip-card">
                        <div style="display:flex; justify-content:space-between;">
                            <span class="score-pill">Clip #{i + 1}</span>
                            <span style="color:#ef4444; font-weight:bold;">
                                🔥 Virality: {v_score:.0f} / 100
                            </span>
                        </div>
                        <h4 style="margin:10px 0 4px 0;">{hook_title}</h4>
                        <p style="color:#a1a1aa; font-size:13px; margin:0;">
                            ⏱️ <b>{start_s:.1f}s - {end_s:.1f}s</b> (Duration: {dur_s:.1f}s)
                        </p>
                        <div style="margin:10px 0; font-size:12px; color:#cbd5e1;">
                            🎣 Hook: <b>{hook_score:.0f}%</b> │ 💬 Chat: <b>{eng_score:.0f}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    if st.button(
                        f"✂ Customize #{i + 1}", key=f"cust_{i}", use_container_width=True
                    ):
                        st.session_state.selected_candidate_idx = i
                        st.toast(f"Selected Clip #{i + 1} for editing in Tab below!")
                with b_col2:
                    if st.button(
                        f"🚀 Render #{i + 1}",
                        key=f"rnd_{i}",
                        type="primary",
                        use_container_width=True,
                    ):
                        st.session_state.selected_candidate_idx = i
                        # Execute render
                        target_out_dir = Path(output_dir)
                        target_out_dir.mkdir(parents=True, exist_ok=True)
                        temp_cand_path = target_out_dir / f"clip_{i + 1}_cand.json"
                        with open(temp_cand_path, "w", encoding="utf-8") as f:
                            json.dump([cand], f)

                        cmd = [
                            "streamclip",
                            "render",
                            str(temp_cand_path),
                            "--video",
                            st.session_state.active_video_path,
                            "--layout",
                            default_layout,
                            "--facecam",
                            default_facecam,
                            "--out-dir",
                            str(target_out_dir),
                            "--top-k",
                            "1",
                        ]
                        if Path("clip1_words.json").is_file():
                            cmd.extend(["--captions", "clip1_words.json"])

                        with st.spinner(f"Encoding Clip #{i + 1} to 1080x1920 vertical format..."):
                            res = subprocess.run(cmd, capture_output=True, text=True)
                            if res.returncode == 0:
                                st.success(
                                    f"Rendered Clip #{i + 1} successfully! Check player below."
                                )
                            else:
                                st.error(f"Render error: {res.stderr}")

    st.divider()

    # SECTION 5: Framing, Hook Banner & Boundary Editor
    st.markdown("### ✂️ Clip Customizer: Framing, Hook Banner & Subtitles")
    if st.session_state.candidates:
        curr_idx = st.session_state.selected_candidate_idx
        active_cand = st.session_state.candidates[curr_idx]

        st.caption(f"Editing **Clip #{curr_idx + 1}**")

        edit_col1, edit_col2 = st.columns(2)
        with edit_col1:
            st.markdown("#### 1. Viral Hook Banner (Top 3.5s)")
            curr_title = active_cand.get("hook_title") or "WAIT FOR THE SCREAM! 💀"
            new_title = st.text_input("Hook Title Text", value=curr_title)
            active_cand["hook_title"] = new_title

            preset_titles = [
                "WAIT FOR THE SCREAM! 💀",
                "HE CANNOT BE SERIOUS 😭",
                "BRO REALLY DID THIS... 💀",
                "WAIT TILL THE END! 🔥",
                "UNBELIEVABLE CLUTCH! 🏆",
            ]
            chosen_preset = st.selectbox(
                "Quick Title Presets", options=["-- Select Preset --", *preset_titles]
            )
            if chosen_preset != "-- Select Preset --":
                active_cand["hook_title"] = chosen_preset
                st.rerun()

        with edit_col2:
            st.markdown("#### 2. Sentence-Aware Boundary Snapping")
            s_val = st.number_input(
                "Clip Start Time (s)", value=float(active_cand.get("start", 0.0)), step=0.5
            )
            e_val = st.number_input(
                "Clip End Time (s)", value=float(active_cand.get("end", 36.0)), step=0.5
            )
            active_cand["start"] = s_val
            active_cand["end"] = e_val

            if st.session_state.words:
                if st.button("✨ Auto-Snap Boundaries to Sentence Pauses"):
                    try:
                        from streamclip.models import Candidate, Word
                        from streamclip.windowing import refine_candidate_with_transcript

                        c_obj = Candidate(
                            start=s_val,
                            end=e_val,
                            peak_t=active_cand.get("peak_t", (s_val + e_val) / 2),
                            score=active_cand.get("score", 1.0),
                        )
                        w_objs = [Word(**w) for w in st.session_state.words]
                        refined = refine_candidate_with_transcript(c_obj, w_objs)
                        active_cand["start"] = refined.start
                        active_cand["end"] = refined.end
                        st.success(
                            f"Snapped! New range: {refined.start:.2f}s -> {refined.end:.2f}s"
                        )
                        st.rerun()
                    except Exception as e:
                        st.warning(f"Could not snap: {e}")

    st.divider()

    # SECTION 6: In-Browser Player & Output Gallery
    st.markdown("### 🎬 Rendered Vertical Shorts Player")
    out_dir_path = Path(output_dir)
    rendered_files: list[Path] = []
    if out_dir_path.is_dir():
        rendered_files.extend(list(out_dir_path.glob("*.mp4")))
    if Path("output/auto_test").is_dir():
        rendered_files.extend(list(Path("output/auto_test").glob("*.mp4")))

    valid_renders = [p for p in rendered_files if p.stat().st_size > 1000]

    if valid_renders:
        p_col1, p_col2 = st.columns([2, 1])
        with p_col1:
            chosen = st.selectbox(
                "Select Rendered Short to Play",
                options=valid_renders,
                format_func=lambda p: f"{p.name} ({p.stat().st_size / (1024 * 1024):.1f} MB)",
            )
            if chosen:
                st.video(str(chosen))
        with p_col2:
            st.markdown("#### Clip Details")
            if chosen:
                st.write(f"📄 **File**: `{chosen.name}`")
                st.write(f"💾 **Size**: `{chosen.stat().st_size / (1024 * 1024):.1f} MB`")
                st.write("📐 **Resolution**: `1080x1920 (9:16 Vertical)`")
                with open(chosen, "rb") as vf:
                    st.download_button(
                        label="⬇️ Download Short (.mp4)",
                        data=vf,
                        file_name=chosen.name,
                        mime="video/mp4",
                        use_container_width=True,
                    )
    else:
        st.info(
            "No rendered clips yet. Click '🚀 Render' on any card above to encode a vertical short!"
        )


if __name__ == "__main__":
    run_app()
