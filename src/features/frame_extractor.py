"""
Timestamp-based frame selection with *sequential* decoding.

Why sequential decoding instead of seeking (CAP_PROP_POS_FRAMES)?
  * Seeking in H.264 is decoder-dependent: OpenCV jumps to a keyframe and decodes
    forward, and some streams land on a neighbouring frame without any error.
  * The old pilot measured the "residual" with CAP_PROP_POS_MSEC, which for
    constant-frame-rate files is just index/fps, so it could not detect a wrong seek.
  * BDDA clips are short (~10 s, max 32 s). Decoding every frame once and keeping
    only the sampled ones is exact by construction (the frame index is a counter)
    and avoids re-decoding a GOP for every seek.

Sampling plan
  Target times t_k = k / sampling_fps, kept only while t_k is inside BOTH videos:
      t_k <= min((n_cam - 1) / fps_cam, (n_gaze - 1) / fps_gaze)
  This fixes the pilot bug where the last target (t = duration) mapped past the last
  camera frame and was silently clamped to frame n-1 (residual 0.033 s at 30 fps).
  Each target maps to the nearest frame: idx = floor(t * fps + 0.5).

Only two decoded frames (one camera, one gaze) are held in memory at a time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class SamplePoint:
    sample_idx: int
    t_sec: float          # target time
    t_ms: int             # integer key, unique within a sequence
    cam_idx: int          # nearest camera frame index
    gaze_idx: int         # nearest gaze frame index


def nearest_index(t: float, fps: float) -> int:
    """Nearest frame index for time t (round half up, not banker's rounding)."""
    return int(math.floor(t * fps + 0.5))


def last_frame_time(n_frames: int, fps: float) -> float:
    return (n_frames - 1) / fps if n_frames > 0 and fps > 0 else -1.0


def sample_plan(cam_frames: int, cam_fps: float, gaze_frames: int, gaze_fps: float,
                sampling_fps: float) -> list[SamplePoint]:
    """Target times inside both videos and the frame each one maps to."""
    t_end = min(last_frame_time(cam_frames, cam_fps),
                last_frame_time(gaze_frames, gaze_fps))
    if t_end < 0:
        return []
    plan = []
    k = 0
    while True:
        t = k / sampling_fps
        if t > t_end + 1e-9:
            break
        ci = nearest_index(t, cam_fps)
        gi = nearest_index(t, gaze_fps)
        # t <= (n-1)/fps guarantees idx <= n-1; assert to catch manifest errors early
        assert ci <= cam_frames - 1 and gi <= gaze_frames - 1, (t, ci, gi)
        plan.append(SamplePoint(k, t, int(round(t * 1000)), ci, gi))
        k += 1
    return plan


class SequentialReader:
    """Forward-only frame reader: grab() every frame, retrieve() only requested ones.

    `read_at(idx)` must be called with non-decreasing indices.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        self.pos = -1                      # index of the last grabbed frame
        self._cache_idx = -1
        self._cache: Optional[np.ndarray] = None
        self.last_container_t: float = float("nan")

    @property
    def opened(self) -> bool:
        return bool(self.cap.isOpened())

    @property
    def fps(self) -> float:
        return float(self.cap.get(cv2.CAP_PROP_FPS))

    @property
    def size(self) -> tuple[int, int]:
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read_at(self, idx: int) -> Optional[np.ndarray]:
        if idx < self.pos:
            raise ValueError(f"SequentialReader is forward-only (asked {idx}, at {self.pos})")
        if idx == self._cache_idx:
            return self._cache
        while self.pos < idx:
            if not self.cap.grab():
                return None                       # stream ended before idx
            self.pos += 1
        ok, frame = self.cap.retrieve()
        if not ok or frame is None:
            return None
        # container timestamp of the frame just decoded (for drift / VFR detection)
        self.last_container_t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self._cache_idx, self._cache = idx, frame
        return frame

    def count_total_frames(self) -> int:
        """Grab to the end and return the number of decodable frames."""
        self._cache, self._cache_idx = None, -1
        while self.cap.grab():
            self.pos += 1
        return self.pos + 1

    def release(self) -> None:
        self._cache = None
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def seek_read(cap: cv2.VideoCapture, idx: int) -> Optional[np.ndarray]:
    """Legacy seek-based read used by the pilot (kept only for the accuracy check)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None
