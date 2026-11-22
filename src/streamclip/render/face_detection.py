"""Facecam and speaker region detection inspired by SupoClip.

Provides automatic detection and temporal stabilization of streamer webcam boxes
from video frames using deep-learning/Haar face detectors or zero-dependency
chroma/edge spatial clustering.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from streamclip.models import FacecamBox, VodMeta
from streamclip.signals.frames import VideoFrame, sample_frames

# Dynamic import for optional OpenCV
try:
    cv2: Any = importlib.import_module("cv2")
    HAS_CV2 = True
except (ImportError, ModuleNotFoundError):
    cv2 = None
    HAS_CV2 = False


@dataclass
class DetectedFacecam:
    """Result of automatic facecam detection."""

    box: FacecamBox
    confidence: float
    preset_name: str
    is_detected: bool = True

    @property
    def x(self) -> int:
        return self.box.x

    @property
    def y(self) -> int:
        return self.box.y

    @property
    def w(self) -> int:
        return self.box.w

    @property
    def h(self) -> int:
        return self.box.h


# Standard quadrant candidate regions normalized to (x_ratio, y_ratio, w_ratio, h_ratio)
CANDIDATE_ZONES = {
    "bottom-left": (0.01, 0.44, 0.22, 0.26),
    "bottom-right": (0.77, 0.44, 0.22, 0.26),
    "top-left": (0.01, 0.02, 0.22, 0.26),
    "top-right": (0.77, 0.02, 0.22, 0.26),
}


def _detect_skin_mask(
    rgb_frame: np.ndarray[Any, np.dtype[np.uint8]],
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Detect skin-tone pixels in RGB space without requiring external libraries."""
    r = rgb_frame[:, :, 0].astype(np.int32)
    g = rgb_frame[:, :, 1].astype(np.int32)
    b = rgb_frame[:, :, 2].astype(np.int32)

    # Standard RGB skin-tone rules:
    # R > 85, G > 40, B > 20
    # max(R,G,B) - min(R,G,B) > 15
    # |R - G| > 15, R > G, R > B
    skin_rule = (
        (r > 85)
        & (g > 40)
        & (b > 20)
        & ((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 15)
        & (np.abs(r - g) > 15)
        & (r > g)
        & (r > b)
    )
    result: np.ndarray[Any, np.dtype[np.bool_]] = np.asarray(skin_rule, dtype=np.bool_)
    return result


def _detect_face_cv2(
    rgb_frame: np.ndarray[Any, np.dtype[np.uint8]],
) -> list[tuple[int, int, int, int]]:
    """Detect faces using cv2 CascadeClassifier if available."""
    if not HAS_CV2 or cv2 is None:
        return []
    try:
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    except Exception:
        return []


def detect_facecam_in_frame(
    frame: VideoFrame | np.ndarray[Any, np.dtype[np.uint8]],
    width: int | None = None,
    height: int | None = None,
) -> DetectedFacecam:
    """Analyze a single video frame to detect the most likely facecam region.

    Evaluates corner zones and checks for human face presence via cv2 or skin-chroma clustering.
    """
    arr = frame.array if isinstance(frame, VideoFrame) else frame
    w_target = width if width is not None else (arr.shape[1] if arr.ndim == 3 else 1920)
    h_target = height if height is not None else (arr.shape[0] if arr.ndim == 3 else 1080)
    if arr.ndim != 3 or arr.shape[0] < 50 or arr.shape[1] < 50:
        # Fallback to default bottom-left
        bx = round(w_target * 0.01)
        by = round(h_target * 0.44)
        bw = round(w_target * 0.22)
        bh = round(h_target * 0.26)
        return DetectedFacecam(
            box=FacecamBox(
                x=bx,
                y=by,
                w=bw if bw % 2 == 0 else bw + 1,
                h=bh if bh % 2 == 0 else bh + 1,
            ),
            confidence=0.0,
            preset_name="bottom-left",
            is_detected=False,
        )

    h_img, w_img, _ = arr.shape

    # 1. Try OpenCV Face Detector if available
    cv_faces = _detect_face_cv2(arr)
    if cv_faces:
        # Find which corner the detected face belongs to or expand around face
        for fx, fy, fw, fh in cv_faces:
            # Check if this face is in a streamer corner (not dead center of full-screen cutscene)
            cx = (fx + fw / 2) / w_img
            cy = (fy + fh / 2) / h_img
            # Corner distance check
            in_corner = (cx < 0.35 or cx > 0.65) or (cy < 0.35 or cy > 0.65)
            if in_corner:
                # Expand face bounding box to typical webcam framing (head + shoulders + room)
                pad_w = int(fw * 1.5)
                pad_h = int(fh * 1.6)
                cam_x = max(0, fx - pad_w // 2)
                cam_y = max(0, fy - pad_h // 3)
                cam_w = min(w_img - cam_x, fw + pad_w)
                cam_h = min(h_img - cam_y, fh + pad_h)
                # Scale to target video width/height if frame was resized
                scale_x = w_target / w_img
                scale_y = h_target / h_img
                final_x = round(cam_x * scale_x)
                final_y = round(cam_y * scale_y)
                final_w = round(cam_w * scale_x)
                final_h = round(cam_h * scale_y)
                if final_w % 2 != 0:
                    final_w += 1
                if final_h % 2 != 0:
                    final_h += 1
                return DetectedFacecam(
                    box=FacecamBox(x=final_x, y=final_y, w=final_w, h=final_h),
                    confidence=0.92,
                    preset_name="detected-cv2",
                    is_detected=True,
                )

    # 2. Chroma / Skin-tone & Edge Spatial Clustering Fallback
    skin_mask = _detect_skin_mask(arr)
    best_zone = "bottom-left"
    best_score = 0.0

    for zone_name, (zx, zy, zw, zh) in CANDIDATE_ZONES.items():
        x1 = int(zx * w_img)
        y1 = int(zy * h_img)
        x2 = int((zx + zw) * w_img)
        y2 = int((zy + zh) * h_img)

        zone_mask = skin_mask[y1:y2, x1:x2]
        if zone_mask.size == 0:
            continue

        skin_ratio = float(np.mean(zone_mask))
        # High skin density in a corner region strongly indicates a webcam face
        # Typically faces occupy 8% to 40% of the webcam box area
        if 0.05 <= skin_ratio <= 0.65:
            score = skin_ratio
            if score > best_score:
                best_score = score
                best_zone = zone_name

    zx, zy, zw, zh = CANDIDATE_ZONES[best_zone]
    cam_x = round(w_target * zx)
    cam_y = round(h_target * zy)
    cam_w = round(w_target * zw)
    cam_h = round(h_target * zh)
    if cam_w % 2 != 0:
        cam_w += 1
    if cam_h % 2 != 0:
        cam_h += 1

    confidence = min(1.0, best_score * 3.0) if best_score > 0.05 else 0.4
    return DetectedFacecam(
        box=FacecamBox(x=cam_x, y=cam_y, w=cam_w, h=cam_h),
        confidence=round(confidence, 2),
        preset_name=best_zone,
        is_detected=best_score > 0.05,
    )


def detect_facecam_region(
    vod: VodMeta,
    sample_timestamps: list[float] | None = None,
    num_samples: int = 5,
) -> DetectedFacecam:
    """Sample multiple keyframes from VOD and compute stabilized facecam bounding box.

    Applies temporal median aggregation across sampled frames.
    """
    dur = max(1.0, vod.duration_s)
    interval = max(1.0, dur / max(1, num_samples))
    try:
        frames = sample_frames(
            vod,
            start_s=0.0,
            end_s=dur,
            interval_s=interval,
            width=640,
            height=360,
        )
    except Exception:
        frames = []

    if not frames:
        # Fallback to standard bottom-left
        w, h = vod.width, vod.height
        cam_w = round(w * 0.22)
        cam_h = round(h * 0.26)
        cam_x = round(w * 0.01)
        cam_y = round(h * 0.44)
        if cam_w % 2 != 0:
            cam_w += 1
        if cam_h % 2 != 0:
            cam_h += 1
        return DetectedFacecam(
            box=FacecamBox(x=cam_x, y=cam_y, w=cam_w, h=cam_h),
            confidence=0.5,
            preset_name="bottom-left",
            is_detected=False,
        )

    detections = [detect_facecam_in_frame(f, vod.width, vod.height) for f in frames]

    # Median smoothing across detected box coordinates
    xs = [d.box.x for d in detections]
    ys = [d.box.y for d in detections]
    ws = [d.box.w for d in detections]
    hs = [d.box.h for d in detections]
    confs = [d.confidence for d in detections]

    med_x = int(np.median(xs))
    med_y = int(np.median(ys))
    med_w = int(np.median(ws))
    med_h = int(np.median(hs))
    if med_w % 2 != 0:
        med_w += 1
    if med_h % 2 != 0:
        med_h += 1

    # Most common preset name
    presets = [d.preset_name for d in detections]
    best_preset = max(set(presets), key=presets.count)
    med_conf = float(np.median(confs))

    return DetectedFacecam(
        box=FacecamBox(x=med_x, y=med_y, w=med_w, h=med_h),
        confidence=round(med_conf, 2),
        preset_name=best_preset,
        is_detected=med_conf >= 0.4,
    )
