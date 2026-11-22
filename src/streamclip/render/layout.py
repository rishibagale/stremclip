"""FFmpeg filtergraphs for vertical video layouts (stack, gameplay_only, facecam_only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from streamclip.models import FacecamBox, RenderSpec


def escape_ffmpeg_filter_path(path: Path | str) -> str:
    """Escape file path for inclusion in ffmpeg filter option strings (e.g. ass=filename='...')."""
    p = Path(path).as_posix()
    # In ffmpeg filter syntax, colons and single quotes must be escaped
    return p.replace(":", r"\:").replace("'", r"\'")


def suggest_facecam(vod: Any) -> FacecamBox:
    """Suggest high-probability facecam region for the VOD using auto detection or presets.

    Attempts automatic facecam detection from video frames, falling back to
    bottom-left 22% width and 26% height scaled to VOD dimensions
    (typical layout for streamers like CarryMinati and IShowSpeed).
    """
    try:
        from streamclip.render.face_detection import detect_facecam_region

        if hasattr(vod, "duration_s"):
            det = detect_facecam_region(vod)
            return det.box
    except Exception:
        pass

    if hasattr(vod, "width") and hasattr(vod, "height"):
        w = int(vod.width)
        h = int(vod.height)
    else:
        try:
            from streamclip.ingest.video import probe

            meta = probe(vod)
            w = meta.width
            h = meta.height
        except Exception:
            w = 1920
            h = 1080

    cam_w = round(w * 0.22)
    cam_h = round(h * 0.26)
    cam_x = round(w * 0.01)
    cam_y = round(h * 0.44)
    # Ensure even dimensions for ffmpeg scaling
    if cam_w % 2 != 0:
        cam_w += 1
    if cam_h % 2 != 0:
        cam_h += 1
    return FacecamBox(x=cam_x, y=cam_y, w=cam_w, h=cam_h)


def resolve_facecam(
    spec: RenderSpec,
    config: Any | None = None,
    default: FacecamBox | None = None,
) -> FacecamBox:
    """Resolve facecam bounding box from RenderSpec, Config, or fallback default."""
    if spec.facecam is not None:
        return spec.facecam
    if config is not None:
        fc = getattr(config, "facecam", None)
        if isinstance(fc, FacecamBox):
            return fc
        if isinstance(fc, dict):
            return FacecamBox(**fc)
    if default is not None:
        return default
    # Default bottom-left facecam region
    return FacecamBox(x=0, y=0, w=480, h=360)


def build_filtergraph(
    spec: RenderSpec,
    config: Any | None = None,
    ass_path: Path | str | None = None,
    top_split_ratio: float = 0.5,
) -> str:
    """Construct an ffmpeg -filter_complex video filtergraph string for vertical video rendering.

    Layouts:
      - 'gameplay_only': Center-crop video to 9:16 aspect ratio scaled to out_w x out_h (1080x1920).
      - 'facecam_only': Crop facecam box from config/spec, scaled to out_w x out_h (1080x1920).
      - 'stack': Vertical stack with gameplay on top (out_w x top_h)
                 and facecam below (out_w x bottom_h).

    Args:
        spec: RenderSpec specifying layout, facecam, captions, out_w, and out_h.
        config: Optional Config instance from which to resolve facecam box.
        ass_path: Optional path to .ass subtitle file for burn-in overlay.
        top_split_ratio: Height proportion allocated to top gameplay region in 'stack' layout.

    Returns:
        String representing the video portion of ffmpeg filter_complex, with output label [v].
    """
    out_w = spec.out_w
    out_h = spec.out_h

    # Subtitle burn-in filter if ass_path provided
    sub_filter = ""
    if ass_path:
        escaped_ass = escape_ffmpeg_filter_path(ass_path)
        sub_filter = f",ass=filename='{escaped_ass}'"

    if spec.layout == "gameplay_only":
        return (
            f"[0:v]scale=w='if(gt(a,{out_w}/{out_h}),-2,{out_w})':"
            f"h='if(gt(a,{out_w}/{out_h}),{out_h},-2)':flags=lanczos,"
            f"crop={out_w}:{out_h},setsar=1{sub_filter}[v]"
        )

    if spec.layout == "facecam_only":
        fc = resolve_facecam(spec, config)
        return (
            f"[0:v]crop={fc.w}:{fc.h}:{fc.x}:{fc.y},"
            f"scale=w='if(gt(a,{out_w}/{out_h}),-2,{out_w})':"
            f"h='if(gt(a,{out_w}/{out_h}),{out_h},-2)':flags=lanczos,"
            f"crop={out_w}:{out_h},setsar=1{sub_filter}[v]"
        )

    if spec.layout == "stack":
        fc = resolve_facecam(spec, config)
        top_h = round(out_h * top_split_ratio)
        # Ensure even height for ffmpeg scaling
        if top_h % 2 != 0:
            top_h += 1
        bottom_h = out_h - top_h

        # Gameplay filter on top, facecam on bottom
        gameplay_filter = (
            f"[v_game]scale=w='if(gt(a,{out_w}/{top_h}),-2,{out_w})':"
            f"h='if(gt(a,{out_w}/{top_h}),{top_h},-2)':flags=lanczos,"
            f"crop={out_w}:{top_h},setsar=1[top]"
        )
        facecam_filter = (
            f"[v_cam]crop={fc.w}:{fc.h}:{fc.x}:{fc.y},"
            f"scale=w='if(gt(a,{out_w}/{bottom_h}),-2,{out_w})':"
            f"h='if(gt(a,{out_w}/{bottom_h}),{bottom_h},-2)':flags=lanczos,"
            f"crop={out_w}:{bottom_h},setsar=1[bottom]"
        )

        if sub_filter:
            return (
                f"[0:v]split=2[v_game][v_cam];"
                f"{gameplay_filter};"
                f"{facecam_filter};"
                f"[top][bottom]vstack=inputs=2[v_stacked];"
                f"[v_stacked]null{sub_filter}[v]"
            )
        return (
            f"[0:v]split=2[v_game][v_cam];"
            f"{gameplay_filter};"
            f"{facecam_filter};"
            f"[top][bottom]vstack=inputs=2[v]"
        )

    raise ValueError(f"Unknown render layout: {spec.layout}")
