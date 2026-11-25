"""
Load the per-frame records written by src/features/video_preprocessing.py and resolve
frame paths. Paths stored in aligned_records.csv are absolute on the machine that ran the
preprocessing; if a stored path does not exist, it is rebuilt from PROCESSED_ROOT
(<split>/frames/<modality>/<video_id>/<ms:06d>.jpg), so the folder can be moved.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import risk_config as C


def frame_path(split: str, modality: str, video_id: str, t: float) -> Path:
    return C.PROCESSED_ROOT / split / "frames" / modality / str(video_id) / f"{int(round(t * 1000)):06d}.jpg"


def _resolve(stored, split, modality, vid, t) -> str:
    if isinstance(stored, str) and stored and Path(stored).exists():
        return stored
    p = frame_path(split, modality, vid, t)
    return str(p) if p.exists() else ""


def load_records(splits=C.SPLITS) -> pd.DataFrame:
    parts = []
    for split in splits:
        p = C.PROCESSED_ROOT / split / "metadata" / "aligned_records.csv"
        if p.exists():
            parts.append(pd.read_csv(p, dtype={"video_id": str}))
    if not parts:
        raise FileNotFoundError(f"No aligned_records.csv under {C.PROCESSED_ROOT}. "
                                "Run src/features/video_preprocessing.py (pilot) or src/risk/run_full_preprocessing.py.")
    df = pd.concat(parts, ignore_index=True)
    df["video_id"] = df["video_id"].astype(str)
    df["time_seconds"] = df["time_seconds"].astype(float).round(3)
    df["camera_path"] = [_resolve(s, sp, "camera", v, t) for s, sp, v, t in
                         zip(df.get("camera_frame_path"), df["split"], df["video_id"], df["time_seconds"])]
    df["gaze_path"] = [_resolve(s, sp, "gaze", v, t) for s, sp, v, t in
                       zip(df.get("gaze_frame_path"), df["split"], df["video_id"], df["time_seconds"])]
    keep = ["split", "video_id", "time_seconds", "camera_path", "gaze_path"]
    return df[keep].sort_values(["split", "video_id", "time_seconds"]).reset_index(drop=True)
