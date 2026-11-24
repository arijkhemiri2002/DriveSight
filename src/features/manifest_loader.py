"""
Load and validate the clean dataset manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from preprocess_config import PreprocessConfig


def load_manifest(config: PreprocessConfig) -> pd.DataFrame:
    """Load dataset_manifest.csv and return a DataFrame."""
    if not config.manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {config.manifest_csv}")
    df = pd.read_csv(config.manifest_csv)
    return df


def select_sequences(df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    """Select sequences for processing (all, or pilot subset per split)."""
    required = ["split", "video_id", "cam_duration", "gaze_duration",
                "cam_fps", "gaze_fps", "cam_w", "cam_h", "gaze_w", "gaze_h"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")

    # The manifest is pre-filtered to sync_ok=True sequences only, so every
    # row is already a valid candidate. Keep them all (or the pilot subset).
    valid = df.copy()

    if config.pilot_mode:
        chosen = []
        for split in config.splits:
            sub = valid[valid["split"] == split].sort_values("video_id")
            chosen.append(sub.head(config.pilot_per_split))
        selected = pd.concat(chosen, ignore_index=True)
    else:
        selected = valid.sort_values(["split", "video_id"]).reset_index(drop=True)

    return selected.reset_index(drop=True)


def sequence_paths(split: str, video_id: str, config: PreprocessConfig) -> dict[str, Path]:
    """Return absolute paths to the three modality files for a sequence."""
    return {
        "camera": config.bdda_root / split / "camera_videos" / f"{video_id}.mp4",
        "gaze": config.bdda_root / split / "gazemap_videos" / f"{video_id}_pure_hm.mp4",
        "gps": config.bdda_root / split / "gps_jsons" / f"{video_id}.json",
    }
