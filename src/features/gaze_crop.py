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

from src.features.preprocess_config import PreprocessConfig


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
    """Save a 1x3 QC figure: camera | gaze crop | overlay (matplotlib, pilot-era)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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


# ----------------------------------------------------------------------------------
# Compact gaze descriptors (Phase 2 output) and fast QC triptych
# ----------------------------------------------------------------------------------
# The BDD-A gaze map is an *aggregate of several in-lab observers* watching the clip
# (Xia et al., ACCV 2018), smoothed into a heat map. These descriptors summarise that
# map; they are QC signals and candidate features, NOT a measurement of the recording
# driver's gaze. Coordinates are normalised to [0, 1] in the cropped 16:9 frame, which
# maps uniformly (x1.25) onto the 1280x720 camera frame.

DESCRIPTOR_DOWNSCALE = 4   # descriptors computed on a 256x144 copy (precision ~0.4 % of width)

GAZE_DESCRIPTOR_COLUMNS = [
    "gaze_max", "gaze_mean", "gaze_active_frac", "gaze_peak_x", "gaze_peak_y",
    "gaze_centroid_x", "gaze_centroid_y", "gaze_spread", "gaze_n_regions",
    "gaze_empty", "gaze_bar_max", "gaze_bar_mass_frac",
]


def gaze_descriptors(gaze_frame_bgr: np.ndarray, config: PreprocessConfig) -> dict:
    gray = cv2.cvtColor(gaze_frame_bgr, cv2.COLOR_BGR2GRAY) if gaze_frame_bgr.ndim == 3 else gaze_frame_bgr
    top, bot = config.gaze_crop_top, config.gaze_crop_bottom
    crop = gray[top:bot, :]
    bars = np.concatenate([gray[:top, :].ravel(), gray[bot:, :].ravel()])
    gmax = int(crop.max()) if crop.size else 0
    crop_mass = float(crop.sum(dtype=np.float64))
    bar_mass = float(bars.sum(dtype=np.float64))
    out = {
        "gaze_max": gmax,
        # Signal in the letterbox bars is attention that the crop discards (e.g. heat-map
        # blur spilling over the top edge of the image). Its share of the total map mass
        # says how much is lost; the max alone says nothing about the amount.
        "gaze_bar_max": int(bars.max()) if bars.size else 0,
        "gaze_bar_mass_frac": round(bar_mass / (bar_mass + crop_mass), 5) if (bar_mass + crop_mass) > 0 else 0.0,
        "gaze_empty": bool(gmax <= config.gaze_empty_max),
    }
    h, w = crop.shape
    small = cv2.resize(crop, (w // DESCRIPTOR_DOWNSCALE, h // DESCRIPTOR_DOWNSCALE),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    sh, sw = small.shape
    out["gaze_mean"] = round(float(small.mean()), 3)
    thr = float(config.gaze_intensity_threshold)
    mask = small > thr
    out["gaze_active_frac"] = round(float(mask.mean()), 5)
    if out["gaze_empty"] or not mask.any():
        out.update(gaze_peak_x=np.nan, gaze_peak_y=np.nan, gaze_centroid_x=np.nan,
                   gaze_centroid_y=np.nan, gaze_spread=np.nan, gaze_n_regions=0)
        return out
    py, px = np.unravel_index(int(np.argmax(small)), small.shape)
    out["gaze_peak_x"] = round((px + 0.5) / sw, 4)
    out["gaze_peak_y"] = round((py + 0.5) / sh, 4)
    wgt = np.clip(small - thr, 0, None)
    ys, xs = np.mgrid[0:sh, 0:sw]
    xs = (xs + 0.5) / sw
    ys = (ys + 0.5) / sh
    tot = wgt.sum()
    cx, cy = float((wgt * xs).sum() / tot), float((wgt * ys).sum() / tot)
    var = float((wgt * ((xs - cx) ** 2 + (ys - cy) ** 2)).sum() / tot)
    out["gaze_centroid_x"] = round(cx, 4)
    out["gaze_centroid_y"] = round(cy, 4)
    out["gaze_spread"] = round(var ** 0.5, 4)          # RMS distance to centroid (normalised units)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    min_area = max(1, int(0.001 * sh * sw))            # ignore specks < 0.1 % of the frame
    out["gaze_n_regions"] = int((stats[1:, cv2.CC_STAT_AREA] >= min_area).sum())
    return out


def make_qc_triptych(camera_bgr: np.ndarray, gaze_frame_bgr: np.ndarray,
                     config: PreprocessConfig, label: str = "") -> np.ndarray:
    """camera | gaze (cropped, resized) | overlay, as one BGR image (no matplotlib)."""
    gaze_resized = process_gaze_frame(gaze_frame_bgr, config)
    cam_rgb = cv2.cvtColor(camera_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.cvtColor(overlay_gaze_on_camera(cam_rgb, gaze_resized,
                                                  intensity_threshold=config.gaze_intensity_threshold),
                           cv2.COLOR_RGB2BGR)
    pw = config.qc_panel_width
    ph = int(round(pw * config.expected_camera_h / config.expected_camera_w))
    panels = [cv2.resize(camera_bgr, (pw, ph), interpolation=cv2.INTER_AREA),
              cv2.resize(cv2.cvtColor(gaze_resized, cv2.COLOR_GRAY2BGR), (pw, ph), interpolation=cv2.INTER_AREA),
              cv2.resize(overlay, (pw, ph), interpolation=cv2.INTER_AREA)]
    img = cv2.hconcat(panels)
    if label:
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(img, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img
