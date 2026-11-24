"""
Gaze map cropping, resizing, and camera/gaze overlay generation.

The raw gaze frame is 1024x768 with black bars at top/bottom.
The central region rows [96:672] (576 px tall) corresponds to the camera
image. This module verifies the crop visually, then applies it globally.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess_config import PreprocessConfig


@dataclass
class CropVerification:
    video_id: str
    split: str
    crop_top: int
    crop_bottom: int
    crop_height: int
    gaze_w: int
    gaze_h: int
    cam_w: int
    cam_h: int
    crop_preserves_center: bool
    notes: str


def crop_gaze(gaze_frame: np.ndarray, crop_top: int,
              crop_bottom: int) -> np.ndarray:
    """Remove black bars: keep rows [crop_top:crop_bottom].

    Works on grayscale (2D) or color (3D) arrays.
    """
    if gaze_frame.ndim == 2:
        return gaze_frame[crop_top:crop_bottom, :]
    return gaze_frame[crop_top:crop_bottom, :, :]


def resize_gaze_to_camera(gaze_crop: np.ndarray,
                          target_w: int, target_h: int) -> np.ndarray:
    """Resize cropped gaze to camera dimensions."""
    return cv2.resize(gaze_crop, (target_w, target_h),
                      interpolation=cv2.INTER_LINEAR)


def gaze_to_heatmap(gaze_resized_gray: np.ndarray) -> np.ndarray:
    """Convert a grayscale gaze map to a JET colormap heatmap (RGB)."""
    heatmap = cv2.applyColorMap(gaze_resized_gray, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def overlay_gaze_on_camera(camera_rgb: np.ndarray,
                           gaze_resized_gray: np.ndarray,
                           alpha_cam: float = 0.55,
                           alpha_gaze: float = 0.45,
                           intensity_threshold: int = 20) -> np.ndarray:
    """Blend gaze heatmap onto camera frame, only where gaze is intense."""
    heatmap = gaze_to_heatmap(gaze_resized_gray)
    mask = gaze_resized_gray > intensity_threshold
    blended = cv2.addWeighted(camera_rgb, alpha_cam, heatmap, alpha_gaze, 0)
    overlay = camera_rgb.copy()
    overlay[mask] = blended[mask]
    return overlay


def save_qc_triptych(camera_rgb: np.ndarray, gaze_raw_gray: np.ndarray,
                     gaze_crop_resized: np.ndarray, overlay: np.ndarray,
                     out_path: Path, title: str = "") -> None:
    """Save a 1x3 QC figure: camera | gaze crop | overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(camera_rgb)
    axes[0].set_title("Camera")
    axes[0].axis("off")

    axes[1].imshow(gaze_crop_resized, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Gaze (crop [96:672] -> 1280x720)")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def verify_crop_on_sequence(camera_path: Path, gaze_path: Path,
                            config: PreprocessConfig,
                            timestamps: Optional[list[float]] = None,
                            out_dir: Optional[Path] = None) -> CropVerification:
    """Visually verify the gaze crop on one sequence at a few timestamps.

    Saves triptych figures if out_dir is given. Returns a CropVerification
    with a heuristic check that the crop retains the central gaze region.
    """
    if timestamps is None:
        timestamps = [0.0, 2.0, 5.0, 9.0]

    cap_c = cv2.VideoCapture(str(camera_path))
    cap_g = cv2.VideoCapture(str(gaze_path))
    if not cap_c.isOpened() or not cap_g.isOpened():
        cap_c.release()
        cap_g.release()
        return CropVerification(
            video_id=camera_path.stem, split="", crop_top=config.gaze_crop_top,
            crop_bottom=config.gaze_crop_bottom,
            crop_height=config.gaze_crop_bottom - config.gaze_crop_top,
            gaze_w=0, gaze_h=0, cam_w=0, cam_h=0,
            crop_preserves_center=False,
            notes="could not open videos",
        )

    cam_w = int(cap_c.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap_c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    gaze_w = int(cap_g.get(cv2.CAP_PROP_FRAME_WIDTH))
    gaze_h = int(cap_g.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap_c.get(cv2.CAP_PROP_FPS)
    gaze_fps = cap_g.get(cv2.CAP_PROP_FPS)

    crop_ok = True
    notes = []

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for t in timestamps:
        cap_c.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * cam_fps)))
        cap_g.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * gaze_fps)))
        ret_c, cam_f = cap_c.read()
        ret_g, gaze_f = cap_g.read()
        if not ret_c or not ret_g:
            notes.append(f"frame unavailable at {t}s")
            continue

        cam_rgb = cv2.cvtColor(cam_f, cv2.COLOR_BGR2RGB)
        gaze_gray = cv2.cvtColor(gaze_f, cv2.COLOR_BGR2GRAY)
        gaze_crop = crop_gaze(gaze_gray, config.gaze_crop_top,
                              config.gaze_crop_bottom)
        gaze_resized = resize_gaze_to_camera(
            gaze_crop, config.expected_camera_w, config.expected_camera_h
        )
        overlay = overlay_gaze_on_camera(cam_rgb, gaze_resized)

        # heuristic: non-zero intensity should exist in the cropped region
        if gaze_crop.max() <= 10:
            crop_ok = False
            notes.append(f"gaze crop near-black at {t}s")

        if out_dir:
            vid = camera_path.stem
            save_qc_triptych(cam_rgb, gaze_gray, gaze_resized, overlay,
                             out_dir / f"{vid}_t{int(t)}.png",
                             title=f"{vid} @ {t}s")

    cap_c.release()
    cap_g.release()

    return CropVerification(
        video_id=camera_path.stem, split="",
        crop_top=config.gaze_crop_top,
        crop_bottom=config.gaze_crop_bottom,
        crop_height=config.gaze_crop_bottom - config.gaze_crop_top,
        gaze_w=gaze_w, gaze_h=gaze_h, cam_w=cam_w, cam_h=cam_h,
        crop_preserves_center=crop_ok,
        notes="; ".join(notes) if notes else "crop verified",
    )


def process_gaze_frame(gaze_frame_bgr: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Crop + resize a gaze frame to camera dimensions (grayscale)."""
    gaze_gray = cv2.cvtColor(gaze_frame_bgr, cv2.COLOR_BGR2GRAY)
    gaze_crop = crop_gaze(gaze_gray, config.gaze_crop_top,
                          config.gaze_crop_bottom)
    gaze_resized = resize_gaze_to_camera(
        gaze_crop, config.expected_camera_w, config.expected_camera_h
    )
    return gaze_resized
