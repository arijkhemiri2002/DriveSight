"""
Phase 5 configuration: fusion of tracks (Phase 4), gaze and GPS (Phase 2) into one row per sampled
frame, braking labels, and the risk model.

Task: for every 200 ms sample, predict whether the driver brakes hard within the next `horizon_sec`
seconds ("imminent braking"). The label comes from GPS deceleration, so no manual annotation is needed.

Inputs (read-only, verified upstream):
  data/processed/{split}/metadata/aligned_records.csv   Phase 2, schema 2.1 (gaze descriptors, GPS)
  data/processed/{split}/tracks/tracks.parquet          Phase 4, schema 4.0 (tracked boxes)

Outputs:
  data/processed/{split}/fusion/frame_features.parquet  one row per Phase 2 row (features + label)
  data/processed/{split}/fusion/object_features.parquet one row per tracked box (per-object features)
  data/processed/{split}/fusion/braking_events.csv      one row per sequence
  data/processed/{split}/fusion/_schema.json            config hash, input hashes, coverage
  data/risk/                                            trained models
  reports/video_dataset/phase5_*.md / .csv              reports
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FUSION_SCHEMA_VERSION = "5.0"
SPLITS = ("training", "validation", "test")

VRU_CLASSES = ("pedestrian", "rider", "motorcycle", "bicycle")
VEHICLE_CLASSES = ("car", "truck", "bus")
ROAD_USER_CLASSES = VRU_CLASSES + VEHICLE_CLASSES


@dataclass(frozen=True)
class FusionConfig:
    processed_root: str = str(PROJECT_ROOT / "data" / "processed")
    reports_dir: str = str(PROJECT_ROOT / "reports" / "video_dataset")
    models_dir: str = str(PROJECT_ROOT / "data" / "risk")
    sampling_fps: float = 5.0

    # ---- braking label (GPS, ~1 Hz) --------------------------------------------------------
    hard_brake_decel: float = 2.0       # m/s^2; 1 Hz GPS smooths peaks, 2.0 m/s^2 ~ 0.2 g
    min_speed_before_brake: float = 1.5  # m/s; ignore "braking" from (almost) standstill
    max_gps_interval: float = 2.5       # s; no acceleration across larger GPS gaps
    horizon_sec: float = 2.0            # positive: braking onset within the next horizon_sec
    # clips without a hard braking (GPS deceleration stayed < hard_brake_decel, i.e. mild braking):
    # "negative" keeps them as negatives over their whole length. Almost all onsets sit at the same clip
    # time (5.5 s), so these clips are what lets a model be compared with others AT THE SAME CLIP TIME.
    no_event_policy: str = "negative"

    # ---- objects ----------------------------------------------------------------------------
    conf_vru: float = 0.15              # a tracked box counts at time t if its conf >= this (VRU)
    conf_other: float = 0.25            # ... (other classes)
    # ego-path corridor in normalised image coordinates (no lane detection): a box is "in path" if
    # bottom_y >= path_y_min and |cx - 0.5| <= half_w_top + growth * (bottom_y - y_min) / (1 - y_min)
    path_y_min: float = 0.45
    path_half_w_top: float = 0.06
    path_half_w_growth: float = 0.22
    near_path_margin: float = 0.10      # "near path": corridor widened by this on each side
    motion_lag_sec: float = 1.0         # past box used for growth / lateral speed: t - lag ...
    motion_lag_tol_sec: float = 0.4     # ... within +/- tol (strictly in the past)
    ttc_cap_sec: float = 10.0           # time-to-contact proxy is capped (no expansion -> cap)
    min_box_h_n: float = 0.02           # boxes smaller than this are too noisy for growth

    # ---- gaze ---------------------------------------------------------------------------------
    gaze_min_sigma: float = 0.05        # floor of the gaze spread used in the attention kernel
    gaze_shift_short_sec: float = 0.2
    gaze_shift_long_sec: float = 1.0

    # ---- model --------------------------------------------------------------------------------
    seed: int = 0
    n_bootstrap: int = 1000

    @property
    def aspect(self) -> float:
        return 1280.0 / 720.0           # x distances are converted to image-height units

    def fusion_params(self) -> dict:
        """Settings that change the fusion output (hashed)."""
        d = asdict(self)
        for k in ("processed_root", "reports_dir", "models_dir", "seed", "n_bootstrap"):
            d.pop(k)
        return d

    def fusion_hash(self, upstream: dict) -> str:
        payload = {"schema": FUSION_SCHEMA_VERSION, "params": self.fusion_params(), "upstream": upstream}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    # ---- paths ----------------------------------------------------------------------------------
    def split_dir(self, split: str) -> Path:
        return Path(self.processed_root) / split

    def records_path(self, split: str) -> Path:
        return self.split_dir(split) / "metadata" / "aligned_records.csv"

    def records_schema_path(self, split: str) -> Path:
        return self.split_dir(split) / "metadata" / "_schema.json"

    def tracks_path(self, split: str) -> Path:
        return self.split_dir(split) / "tracks" / "tracks.parquet"

    def tracks_schema_path(self, split: str) -> Path:
        return self.split_dir(split) / "tracks" / "_schema.json"

    def fusion_dir(self, split: str) -> Path:
        return self.split_dir(split) / "fusion"

    def frame_features_path(self, split: str) -> Path:
        return self.fusion_dir(split) / "frame_features.parquet"

    def object_features_path(self, split: str) -> Path:
        return self.fusion_dir(split) / "object_features.parquet"

    def events_path(self, split: str) -> Path:
        return self.fusion_dir(split) / "braking_events.csv"

    def fusion_schema_path(self, split: str) -> Path:
        return self.fusion_dir(split) / "_schema.json"


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def assert_safe_output(path: Path) -> None:
    """Never write under data/BDDA (raw data) nor into the Phase 2/3/4 folders."""
    p = Path(path).resolve()
    parts = [x.lower() for x in p.parts]
    if "bdda" in parts:
        raise RuntimeError(f"refusing to write under the raw BDDA folder: {p}")
    for protected in ("metadata", "detections", "tracks"):
        if protected in parts and "processed" in parts:
            raise RuntimeError(f"refusing to write into an upstream phase folder: {p}")
