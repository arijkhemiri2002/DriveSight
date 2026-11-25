"""
Central configuration for DriveSight Phase 2 (on-the-fly multimodal preprocessing).

Every parameter that changes the *content* of the output is part of
`PreprocessConfig.output_hash()`. A sequence is only skipped on resume when its
checkpoint was produced with the same hash, so changing a threshold can never
silently mix old and new results.

Paths are relative to PROJECT_ROOT (the Git repository root).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Bump when the CSV schema or the meaning of a column changes.
SCHEMA_VERSION = "2.1"   # 2.1: gaze_bar_mass_frac; bar warning based on mass, not max


@dataclass(frozen=True)
class PreprocessConfig:
    # ---- Locations -------------------------------------------------------------
    bdda_root: Path = PROJECT_ROOT / "data" / "BDDA"                       # READ-ONLY
    manifest_csv: Path = PROJECT_ROOT / "reports" / "video_dataset" / "dataset_manifest.csv"
    processed_root: Path = PROJECT_ROOT / "data" / "processed"
    reports_dir: Path = PROJECT_ROOT / "reports" / "video_dataset"

    # ---- Temporal sampling -----------------------------------------------------
    sampling_fps: float = 5.0          # target sample rate (Hz) -> one row every 200 ms

    # ---- GPS -------------------------------------------------------------------
    # Nearest GPS record is attached if |t_gps - t| <= this gap. BDDA GPS is 1 Hz with
    # records at 0.5, 1.5, ... s, so the observed maximum is 0.5 s; 1.0 s tolerates a
    # single slightly late record without accepting a missing one.
    gps_assoc_max_gap_sec: float = 1.0
    # Linear interpolation of speed is only done between two *valid* records that are
    # at most this far apart (one missing record at 1 Hz is tolerated).
    gps_interp_max_span_sec: float = 2.0
    # Acceleration is only computed between consecutive valid records at most this far apart.
    gps_accel_max_dt_sec: float = 2.0
    # Speeds above this are physically implausible for these urban/highway clips and are
    # treated as invalid (60 m/s = 216 km/h). Negative speeds (iOS sentinel -1) are invalid.
    gps_max_speed_m_s: float = 60.0

    # ---- Gaze crop (verified visually in the pilot QC) ------------------------
    gaze_crop_top: int = 96            # rows [96:672] of the 1024x768 gaze map
    gaze_crop_bottom: int = 672        # -> 1024x576 (16:9) == camera aspect ratio
    expected_camera_w: int = 1280
    expected_camera_h: int = 720
    expected_gaze_w: int = 1024
    expected_gaze_h: int = 768
    # Pixels <= this value are treated as background when computing gaze descriptors.
    # Same value as the overlay threshold used in the pilot QC figures.
    gaze_intensity_threshold: int = 20
    # A gaze frame whose maximum is <= this value is flagged gaze_empty.
    gaze_empty_max: int = 10
    # Warn when more than this share of a frame's gaze mass lies in the cropped-out bars.
    gaze_bar_mass_warn: float = 0.05

    # ---- QC --------------------------------------------------------------------
    qc_fraction: float = 0.05          # share of sequences that get one QC overlay
    qc_panel_width: int = 640          # width of each panel in the QC triptych
    qc_jpeg_quality: int = 85

    splits: tuple[str, ...] = field(default_factory=lambda: ("training", "validation", "test"))

    # ------------------------------------------------------------------------------
    # Paths of the output layout:
    #   data/processed/{split}/metadata/aligned_records.csv
    #   data/processed/{split}/metadata/sequence_stats.csv
    #   data/processed/{split}/metadata/sequence_records/{video_id}.csv
    #   data/processed/{split}/metadata/sequence_checkpoints/{video_id}.json
    #   data/processed/{split}/qc/overlays/{video_id}_t{ms}.jpg
    #   data/processed/{split}/logs/
    # ------------------------------------------------------------------------------
    def split_dir(self, split: str) -> Path:
        return self.processed_root / split

    def metadata_dir(self, split: str) -> Path:
        return self.split_dir(split) / "metadata"

    def sequence_records_dir(self, split: str) -> Path:
        return self.metadata_dir(split) / "sequence_records"

    def checkpoints_dir(self, split: str) -> Path:
        return self.metadata_dir(split) / "sequence_checkpoints"

    def aligned_records_path(self, split: str) -> Path:
        return self.metadata_dir(split) / "aligned_records.csv"

    def sequence_stats_path(self, split: str) -> Path:
        return self.metadata_dir(split) / "sequence_stats.csv"

    def qc_overlay_dir(self, split: str) -> Path:
        return self.split_dir(split) / "qc" / "overlays"

    def logs_dir(self, split: str) -> Path:
        return self.split_dir(split) / "logs"

    # ------------------------------------------------------------------------------
    def output_hash(self) -> str:
        """Hash of every setting that changes the content of a sequence CSV."""
        d = asdict(self)
        for k in ("bdda_root", "manifest_csv", "processed_root", "reports_dir",
                  "qc_fraction", "qc_panel_width", "qc_jpeg_quality", "splits"):
            d.pop(k, None)
        d["schema_version"] = SCHEMA_VERSION
        blob = json.dumps(d, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def assert_safe_output(self, path: Path) -> None:
        """Refuse to write anything inside the raw BDDA dataset."""
        p = Path(path).resolve()
        raw = self.bdda_root.resolve()
        if p == raw or raw in p.parents:
            raise PermissionError(f"Refusing to write inside the raw dataset: {p}")
