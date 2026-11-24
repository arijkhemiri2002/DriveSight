"""
Camera and gaze frame extraction with timestamp-based alignment.

For each target timestamp t (derived from SAMPLING_FPS), the nearest frame
is selected by seeking to round(t * fps), reading, and recording the
actual frame time via cv2.CAP_PROP_POS_MSEC. Temporal residuals are
recorded so downstream stages can weight/filter by alignment quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from preprocess_config import PreprocessConfig


@dataclass
class ExtractedFrame:
    modality: str          # "camera" or "gaze"
    video_id: str
    split: str
    target_t: float        # requested timestamp (s)
    frame_index: int       # frame index in source video
    actual_t: float        # actual frame time (s) from container
    residual_sec: float    # |actual_t - target_t|
    width: int
    height: int
    path: Path             # output JPEG path
    ok: bool


def target_timestamps(duration_sec: float, sampling_fps: float) -> list[float]:
    """Generate target timestamps: 0, 1/sampling_fps, 2/... <= duration."""
    if duration_sec is None or duration_sec <= 0:
        return []
    step = 1.0 / sampling_fps
    n = int(np.floor(duration_sec * sampling_fps)) + 1
    ts = [i * step for i in range(n)]
    # keep only those strictly within duration (allow small epsilon)
    ts = [t for t in ts if t <= duration_sec + 1e-6]
    return ts


def extract_frame(video_path: Path, modality: str, target_t: float,
                  fps: float, out_path: Path) -> ExtractedFrame:
    """Extract the frame nearest to target_t from a video.

    Seeks to round(target_t * fps), reads the frame, and reports the
    actual container timestamp via CAP_PROP_POS_MSEC.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return _failed(modality, target_t, out_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_index = int(round(target_t * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    actual_t = float("nan")
    residual_sec = float("nan")
    ok = False

    if ret and frame is not None:
        actual_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        actual_t = actual_msec / 1000.0 if actual_msec > 0 else frame_index / fps
        residual_sec = abs(actual_t - target_t)
        ok = True

    cap.release()
    return ExtractedFrame(
        modality=modality, video_id="", split="", target_t=target_t,
        frame_index=frame_index, actual_t=actual_t, residual_sec=residual_sec,
        width=width, height=height, path=out_path, ok=ok,
    )


def extract_frame_from_capture(cap: cv2.VideoCapture, modality: str,
                               target_t: float, fps: float,
                               out_path: Path) -> tuple[Optional[np.ndarray], ExtractedFrame]:
    """Seek and read one frame from an already-open capture.

    Returns (frame_or_None, ExtractedFrame). The caller keeps the capture open.
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = int(round(target_t * fps))
    # clamp to valid range [0, frame_count-1] to avoid out-of-range seeks
    if frame_count > 0:
        frame_index = max(0, min(frame_index, frame_count - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    actual_t = float("nan")
    residual_sec = float("nan")
    ok = False

    if ret and frame is not None:
        actual_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        actual_t = actual_msec / 1000.0 if actual_msec > 0 else frame_index / fps
        residual_sec = abs(actual_t - target_t)
        ok = True

    ef = ExtractedFrame(
        modality=modality, video_id="", split="", target_t=target_t,
        frame_index=frame_index, actual_t=actual_t, residual_sec=residual_sec,
        width=width, height=height, path=out_path, ok=ok,
    )
    return (frame if ok else None), ef


def _failed(modality: str, target_t: float, out_path: Path) -> ExtractedFrame:
    return ExtractedFrame(
        modality=modality, video_id="", split="", target_t=target_t,
        frame_index=-1, actual_t=float("nan"), residual_sec=float("nan"),
        width=0, height=0, path=out_path, ok=False,
    )


def save_frame(frame: np.ndarray, out_path: Path, jpeg_quality: int = 95) -> bool:
    """Save a BGR frame as JPEG. Returns True on success."""
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_path), frame,
                         [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        return bool(ok)
    except Exception:
        return False


def frame_alignment_ok(residual_sec: Optional[float],
                       max_err: float) -> bool:
    """Check whether a frame's temporal residual is within tolerance."""
    if residual_sec is None or np.isnan(residual_sec):
        return False
    return bool(residual_sec <= max_err)
