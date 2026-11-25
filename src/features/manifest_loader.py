"""
Load the clean manifest (sync_ok sequences only) and select sequences to process.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.preprocess_config import PreprocessConfig

REQUIRED_COLUMNS = ["split", "video_id", "cam_duration", "gaze_duration",
                    "cam_fps", "gaze_fps", "cam_frames", "gaze_frames",
                    "cam_w", "cam_h", "gaze_w", "gaze_h"]


def load_manifest(config: PreprocessConfig) -> pd.DataFrame:
    """Load dataset_manifest.csv and check the columns the pipeline relies on."""
    if not config.manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {config.manifest_csv}")
    df = pd.read_csv(config.manifest_csv)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    df["video_id"] = df["video_id"].astype(str)
    if df.duplicated(["video_id"]).any():
        dups = df.loc[df.duplicated(["video_id"], keep=False), ["split", "video_id"]]
        raise ValueError(f"video_id appears more than once in the manifest:\n{dups}")
    unknown = set(df["split"]) - set(config.splits)
    if unknown:
        raise ValueError(f"Unknown splits in manifest: {unknown}")
    return df


def _sort_key(s: pd.Series) -> pd.Series:
    # numeric order for numeric ids ("2" < "10"), text order otherwise
    return pd.to_numeric(s, errors="coerce").fillna(float("inf"))


def sort_sequences(df: pd.DataFrame) -> pd.DataFrame:
    return (df.assign(_k=_sort_key(df["video_id"]))
              .sort_values(["split", "_k", "video_id"])
              .drop(columns="_k").reset_index(drop=True))


def select_pilot(df: pd.DataFrame, per_split: int = 3) -> pd.DataFrame:
    """Extended pilot: first `per_split` ids of each split (the original 9-sequence
    pilot), plus the edge cases most likely to break alignment:
      * validation/1744 (known invalid GPS speeds), already in the original pilot
      * one ~60 fps camera sequence
      * the longest sequence
    """
    parts = []
    for split in df["split"].unique():
        parts.append(sort_sequences(df[df["split"] == split]).head(per_split))
    base = pd.concat(parts)
    extra = []
    hi_fps = sort_sequences(df[df["cam_fps"] > 50])
    if len(hi_fps):
        extra.append(hi_fps.head(1))
    longest = df.loc[[df["cam_duration"].idxmax()]]
    extra.append(longest)
    out = pd.concat([base] + extra).drop_duplicates("video_id")
    return sort_sequences(out)


def select_sequences(df: pd.DataFrame, *, split: str | None = None,
                     video_ids: list[str] | None = None, pilot: bool = False,
                     limit: int | None = None) -> pd.DataFrame:
    sel = df
    if split:
        sel = sel[sel["split"] == split]
    if video_ids:
        wanted = {str(v) for v in video_ids}
        unknown = wanted - set(sel["video_id"])
        if unknown:
            raise ValueError(f"video ids not in the (filtered) manifest: {sorted(unknown)}")
        sel = sel[sel["video_id"].isin(wanted)]
    if pilot:
        sel = select_pilot(sel)
    sel = sort_sequences(sel)
    if limit:
        sel = sel.head(limit)
    return sel.reset_index(drop=True)


def sequence_paths(split: str, video_id: str, config: PreprocessConfig) -> dict[str, Path]:
    """Absolute paths to the three modality files of a sequence."""
    return {
        "camera": config.bdda_root / split / "camera_videos" / f"{video_id}.mp4",
        "gaze": config.bdda_root / split / "gazemap_videos" / f"{video_id}_pure_hm.mp4",
        "gps": config.bdda_root / split / "gps_jsons" / f"{video_id}.json",
    }
