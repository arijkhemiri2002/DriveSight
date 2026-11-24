"""
Preprocessing configuration for DriveSight BDDA video preprocessing.

All parameters are centralized here so the pipeline is reproducible.
Paths are relative to PROJECT_ROOT (= <repo>/DriveSight).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---- Source dataset (read-only) ----
BDDA_ROOT = PROJECT_ROOT / "data" / "BDDA"
MANIFEST_CSV = PROJECT_ROOT / "reports" / "video_dataset" / "dataset_manifest.csv"

# ---- Output (does not touch data/BDDA) ----
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

# ---- Sampling / alignment ----
SAMPLING_FPS = 5.0                 # target frame sampling rate (Hz)
MAX_FRAME_ALIGN_ERR_SEC = 0.15     # max |actual_frame_time - target_t| allowed (s)
GPS_ASSOC_MAX_GAP_SEC = 1.0        # max |t_gps - target_t| for association (s)
GPS_INTERPOLATE = False            # interpolation disabled by default

# ---- Gaze crop (verified before global apply) ----
GAZE_CROP_TOP = 96                 # rows [96:672]
GAZE_CROP_BOTTOM = 672
EXPECTED_CAMERA_W = 1280
EXPECTED_CAMERA_H = 720
EXPECTED_GAZE_W = 1024
EXPECTED_GAZE_H = 768

# ---- Image encoding ----
JPEG_QUALITY = 95                  # camera/gaze frame JPEG quality

# ---- Pilot ----
PILOT_MODE = True
PILOT_PER_SPLIT = 3                # 3 sequences per split for the pilot

# ---- Splits ----
SPLITS = ["training", "validation", "test"]


@dataclass(frozen=True)
class PreprocessConfig:
    bdda_root: Path = BDDA_ROOT
    manifest_csv: Path = MANIFEST_CSV
    processed_root: Path = PROCESSED_ROOT
    sampling_fps: float = SAMPLING_FPS
    max_frame_align_err_sec: float = MAX_FRAME_ALIGN_ERR_SEC
    gps_assoc_max_gap_sec: float = GPS_ASSOC_MAX_GAP_SEC
    gps_interpolate: bool = GPS_INTERPOLATE
    gaze_crop_top: int = GAZE_CROP_TOP
    gaze_crop_bottom: int = GAZE_CROP_BOTTOM
    expected_camera_w: int = EXPECTED_CAMERA_W
    expected_camera_h: int = EXPECTED_CAMERA_H
    expected_gaze_w: int = EXPECTED_GAZE_W
    expected_gaze_h: int = EXPECTED_GAZE_H
    jpeg_quality: int = JPEG_QUALITY
    pilot_mode: bool = PILOT_MODE
    pilot_per_split: int = PILOT_PER_SPLIT
    splits: tuple[str, ...] = field(default_factory=lambda: tuple(SPLITS))

    def split_dir(self, split: str) -> Path:
        return self.processed_root / split

    def frames_dir(self, split: str, modality: str) -> Path:
        return self.split_dir(split) / "frames" / modality

    def metadata_dir(self, split: str) -> Path:
        return self.split_dir(split) / "metadata"

    def qc_dir(self, split: str) -> Path:
        return self.split_dir(split) / "metadata" / "qc"

    def aligned_records_path(self, split: str) -> Path:
        return self.metadata_dir(split) / "aligned_records.csv"
