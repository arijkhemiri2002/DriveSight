"""Synthetic BDDA-like sequences for tests (camera/gaze mp4 + GPS json).

Camera frames encode their own index as 10 black/white squares, so a test can tell
exactly which frame was decoded. Gaze frames contain one Gaussian blob whose x
position is a known function of the gaze frame index.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

N_BITS = 10
SQ = 64                    # square size (px) used to encode the frame index


def encode_cam_frame(idx: int, w=1280, h=720) -> np.ndarray:
    img = np.full((h, w, 3), 90, np.uint8)
    for b in range(N_BITS):
        v = 255 if (idx >> b) & 1 else 0
        x0 = 40 + b * (SQ + 40)
        img[100:100 + SQ, x0:x0 + SQ] = v
    return img


def decode_cam_index(img: np.ndarray) -> int:
    idx = 0
    for b in range(N_BITS):
        x0 = 40 + b * (SQ + 40)
        if img[100 + 16:100 + SQ - 16, x0 + 16:x0 + SQ - 16].mean() > 128:
            idx |= 1 << b
    return idx


def gaze_blob_x(idx: int, n: int) -> float:
    """Normalised x of the blob centre in the cropped frame."""
    return 0.15 + 0.7 * idx / max(n - 1, 1)


def make_gaze_frame(idx: int, n: int, w=1024, h=768, top=96, bot=672) -> np.ndarray:
    img = np.zeros((h, w), np.uint8)
    cx = int(gaze_blob_x(idx, n) * w)
    cy = top + int(0.4 * (bot - top))
    yy, xx = np.mgrid[0:h, 0:w]
    g = 230 * np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2 * 35.0 ** 2))
    img[top:bot] = g[top:bot].astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def write_video(path: Path, frames_fn, n: int, fps: float, size) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert vw.isOpened(), "VideoWriter failed (mp4v codec unavailable?)"
    for i in range(n):
        vw.write(frames_fn(i))
    vw.release()


def write_gps(path: Path, duration: float, speeds: list[float], start=1_500_000_000_000,
              courses=None) -> None:
    locs = []
    for k, v in enumerate(speeds):
        locs.append({"timestamp": start + int((0.5 + k) * 1000), "speed": v,
                     "course": (courses[k] if courses else 90.0), "latitude": 37.8,
                     "longitude": -122.27, "accuracy": 5.0})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"startTime": start, "endTime": start + int(duration * 1000),
                                "fileName": path.stem, "locations": locs}))


def make_sequence(root: Path, split: str, vid: str, cam_fps=30.0, cam_n=300,
                  gaze_fps=29.0, gaze_n=292, speeds=None) -> dict:
    write_video(root / split / "camera_videos" / f"{vid}.mp4", encode_cam_frame, cam_n, cam_fps, (1280, 720))
    write_video(root / split / "gazemap_videos" / f"{vid}_pure_hm.mp4",
                lambda i: make_gaze_frame(i, gaze_n), gaze_n, gaze_fps, (1024, 768))
    dur = cam_n / cam_fps
    if speeds is None:
        speeds = [5.0 + k for k in range(int(dur))]
    write_gps(root / split / "gps_jsons" / f"{vid}.json", dur, speeds)
    return {"split": split, "video_id": vid, "cam_w": 1280, "cam_h": 720, "cam_fps": cam_fps,
            "cam_frames": cam_n, "cam_duration": cam_n / cam_fps, "gaze_w": 1024, "gaze_h": 768,
            "gaze_fps": gaze_fps, "gaze_frames": gaze_n, "gaze_duration": gaze_n / gaze_fps}
