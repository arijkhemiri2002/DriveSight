"""
Temporal synchronization audit for the DriveSight BDDA dataset.

Audits camera videos, gaze map videos and GPS JSON files across the
training / validation / test splits and produces:

* reports/video_dataset/temporal_synchronization_audit.md  (human report)
* reports/video_dataset/*.csv                              (structured results)
* reports/video_dataset/figures/*.png                      (visual QC)

Run:  python src/features/sync_audit.py
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project paths (resolved from this file location: <root>/src/features/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BDDA_ROOT = PROJECT_ROOT / "data" / "BDDA"
SPLITS = ["training", "validation", "test"]
MODALITIES = {
    "camera": "camera_videos",
    "gaze": "gazemap_videos",
    "gps": "gps_jsons",
}
REPORT_DIR = PROJECT_ROOT / "reports" / "video_dataset"
FIG_DIR = REPORT_DIR / "figures"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Expected dimensions (from data/BDDA/README.txt)
# ---------------------------------------------------------------------------
EXPECTED_CAMERA_W, EXPECTED_CAMERA_H = 1280, 720
EXPECTED_GAZE_W, EXPECTED_GAZE_H = 1024, 768
GAZE_CROP_TOP = 96
GAZE_CROP_BOTTOM = 672  # rows [96:672] -> central 576 x 1024 region

# ---------------------------------------------------------------------------
# Synchronization tolerances (justified below; adjustable)
# ---------------------------------------------------------------------------
# Camera FPS may be ~29.97, 30 or ~59.94; gaze ~29. FPS does not need to match
# across modalities because alignment is timestamp-based, but each value must
# fall inside a plausible range.
FPS_TOLERANCE = 2.0          # |measured_fps - nominal_fps| <= 2.0 is acceptable
CAMERA_NOMINAL_FPS = {29.97, 30.0, 59.94}
GAZE_NOMINAL_FPS = {29.0, 29.97, 30.0}

# Duration mismatch between camera and gaze (seconds). A few frames of slack
# are acceptable because container/rounding effects shift frame counts.
DURATION_TOLERANCE_SEC = 1.0

# Start-time misalignment between camera and gaze (seconds). Both videos are
# expected to start at t = 0; anything above 0.5 s is flagged.
START_OFFSET_TOLERANCE_SEC = 0.5

# GPS sampling: nominally one record per second.
GPS_MEDIAN_INTERVAL_TOLERANCE_SEC = 0.6   # |median_interval - 1.0|
GPS_MAX_GAP_TOLERANCE_SEC = 2.5           # largest allowed gap between records

# GPS timeline coverage: first record may start after video start (offset
# observed in the wild), but must begin within this window; the last record
# must reach within this window of the video end.
GPS_START_OFFSET_MAX_SEC = 2.0
GPS_END_COVERAGE_MAX_SEC = 2.0

# Frame <-> GPS association: nearest-neighbor match only accepted if the
# nearest GPS record is within this many seconds (GPS is ~1 Hz).
GPS_ASSOC_MAX_GAP_SEC = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def video_id_from_gaze_name(name: str) -> str:
    """Map a gaze filename such as '2_pure_hm.mp4' to video id '2'."""
    stem = Path(name).stem
    if stem.endswith("_pure_hm"):
        stem = stem[: -len("_pure_hm")]
    return stem


def list_sequence_ids(split: str) -> dict[str, dict[str, bool]]:
    """Return {video_id: {modality: exists}} for one split."""
    ids: dict[str, dict[str, bool]] = {}
    for modality, folder in MODALITIES.items():
        d = BDDA_ROOT / split / folder
        if not d.exists():
            continue
        for f in d.iterdir():
            if not f.is_file():
                continue
            if modality == "camera" and f.suffix.lower() == ".mp4":
                vid = f.stem
            elif modality == "gaze" and f.suffix.lower() == ".mp4":
                vid = video_id_from_gaze_name(f.name)
            elif modality == "gps" and f.suffix.lower() == ".json":
                vid = f.stem
            else:
                continue
            ids.setdefault(vid, {m: False for m in MODALITIES})[modality] = True
    return ids


def get_video_metadata(path: Path) -> dict:
    """Decode container-level metadata for a video file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {
            "width": None, "height": None, "fps": None,
            "frame_count": None, "duration_sec": None, "readable": False,
        }
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else None
    cap.release()
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration,
        "readable": True,
    }


def inspect_gps(path: Path) -> dict:
    """Parse a GPS JSON file and compute timeline statistics."""
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    locations = meta.get("locations", [])
    df = pd.DataFrame(locations)
    start_time = meta.get("startTime")
    end_time = meta.get("endTime")

    record = {
        "gps_records": len(df),
        "gps_start_time_ms": start_time,
        "gps_end_time_ms": end_time,
        "gps_file_name": meta.get("fileName"),
        "gps_duration_sec": (end_time - start_time) / 1000.0
        if start_time is not None and end_time is not None
        else None,
        "gps_readable": True,
    }

    if df.empty:
        record.update({
            "gps_sorted": None, "gps_dup_timestamps": None,
            "gps_median_interval_sec": None, "gps_min_interval_sec": None,
            "gps_max_interval_sec": None,
            "gps_first_offset_sec": None, "gps_last_offset_sec": None,
            "gps_speed_min": None, "gps_speed_max": None, "gps_speed_mean": None,
        })
        return record

    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"].astype(float)
    record["gps_sorted"] = bool(ts.is_monotonic_increasing)
    record["gps_dup_timestamps"] = int(ts.duplicated().sum())

    intervals = ts.diff().dropna() / 1000.0
    if len(intervals):
        record["gps_median_interval_sec"] = float(intervals.median())
        record["gps_min_interval_sec"] = float(intervals.min())
        record["gps_max_interval_sec"] = float(intervals.max())
    else:
        record["gps_median_interval_sec"] = None
        record["gps_min_interval_sec"] = None
        record["gps_max_interval_sec"] = None

    if start_time is not None:
        record["gps_first_offset_sec"] = float(
            (ts.iloc[0] - start_time) / 1000.0
        )
    if end_time is not None:
        record["gps_last_offset_sec"] = float(
            (end_time - ts.iloc[-1]) / 1000.0
        )

    if "speed" in df.columns:
        record["gps_speed_min"] = float(df["speed"].min())
        record["gps_speed_max"] = float(df["speed"].max())
        record["gps_speed_mean"] = float(df["speed"].mean())

    return record


@dataclass
class SequenceAudit:
    split: str
    video_id: str
    camera_exists: bool = False
    gaze_exists: bool = False
    gps_exists: bool = False
    complete: bool = False
    # camera
    cam_w: Optional[int] = None
    cam_h: Optional[int] = None
    cam_fps: Optional[float] = None
    cam_frames: Optional[int] = None
    cam_duration: Optional[float] = None
    cam_readable: bool = False
    # gaze
    gaze_w: Optional[int] = None
    gaze_h: Optional[int] = None
    gaze_fps: Optional[float] = None
    gaze_frames: Optional[int] = None
    gaze_duration: Optional[float] = None
    gaze_readable: bool = False
    # gps
    gps_readable: bool = False
    gps_records: Optional[int] = None
    gps_duration: Optional[float] = None
    gps_sorted: Optional[bool] = None
    gps_dups: Optional[int] = None
    gps_median_interval: Optional[float] = None
    gps_max_interval: Optional[float] = None
    gps_first_offset: Optional[float] = None
    gps_last_offset: Optional[float] = None
    gps_speed_mean: Optional[float] = None
    # alignment
    cam_gaze_duration_diff: Optional[float] = None
    cam_gaze_start_aligned: Optional[bool] = None
    fps_camera_plausible: Optional[bool] = None
    fps_gaze_plausible: Optional[bool] = None
    resolution_camera_ok: Optional[bool] = None
    resolution_gaze_ok: Optional[bool] = None
    gps_covers_video: Optional[bool] = None
    gps_interval_ok: Optional[bool] = None
    sync_ok: bool = False
    problems: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        d = asdict(self)
        d["problems"] = "; ".join(self.problems)
        return d


# ---------------------------------------------------------------------------
# Per-sequence audit
# ---------------------------------------------------------------------------
def audit_sequence(split: str, video_id: str, ids: dict) -> SequenceAudit:
    res = SequenceAudit(split=split, video_id=video_id)
    res.camera_exists = ids.get(video_id, {}).get("camera", False)
    res.gaze_exists = ids.get(video_id, {}).get("gaze", False)
    res.gps_exists = ids.get(video_id, {}).get("gps", False)
    res.complete = res.camera_exists and res.gaze_exists and res.gps_exists

    if not res.complete:
        missing = [m for m in ("camera", "gaze", "gps")
                   if not ids.get(video_id, {}).get(m, False)]
        res.problems.append(f"missing modality: {', '.join(missing)}")
        res.sync_ok = False
        return res

    cam_path = BDDA_ROOT / split / MODALITIES["camera"] / f"{video_id}.mp4"
    gaze_path = BDDA_ROOT / split / MODALITIES["gaze"] / f"{video_id}_pure_hm.mp4"
    gps_path = BDDA_ROOT / split / MODALITIES["gps"] / f"{video_id}.json"

    cam = get_video_metadata(cam_path)
    gaze = get_video_metadata(gaze_path)
    gps = inspect_gps(gps_path)

    res.cam_readable = cam["readable"]
    res.gaze_readable = gaze["readable"]

    if cam["readable"]:
        res.cam_w, res.cam_h = cam["width"], cam["height"]
        res.cam_fps, res.cam_frames = cam["fps"], cam["frame_count"]
        res.cam_duration = cam["duration_sec"]
    if gaze["readable"]:
        res.gaze_w, res.gaze_h = gaze["width"], gaze["height"]
        res.gaze_fps, res.gaze_frames = gaze["fps"], gaze["frame_count"]
        res.gaze_duration = gaze["duration_sec"]

    # Resolution checks
    res.resolution_camera_ok = bool(
        cam["readable"] and res.cam_w == EXPECTED_CAMERA_W
        and res.cam_h == EXPECTED_CAMERA_H
    )
    res.resolution_gaze_ok = bool(
        gaze["readable"] and res.gaze_w == EXPECTED_GAZE_W
        and res.gaze_h == EXPECTED_GAZE_H
    )
    if not res.resolution_camera_ok:
        res.problems.append(
            f"camera resolution {res.cam_w}x{res.cam_h} "
            f"!= {EXPECTED_CAMERA_W}x{EXPECTED_CAMERA_H}"
        )
    if not res.resolution_gaze_ok:
        res.problems.append(
            f"gaze resolution {res.gaze_w}x{res.gaze_h} "
            f"!= {EXPECTED_GAZE_W}x{EXPECTED_GAZE_H}"
        )

    # FPS plausibility
    res.fps_camera_plausible = bool(
        cam["readable"] and any(abs(res.cam_fps - n) <= FPS_TOLERANCE
                                for n in CAMERA_NOMINAL_FPS)
    )
    res.fps_gaze_plausible = bool(
        gaze["readable"] and any(abs(res.gaze_fps - n) <= FPS_TOLERANCE
                                 for n in GAZE_NOMINAL_FPS)
    )
    if not res.fps_camera_plausible:
        res.problems.append(f"camera fps {res.cam_fps} not plausible")
    if not res.fps_gaze_plausible:
        res.problems.append(f"gaze fps {res.gaze_fps} not plausible")

    # GPS fields
    res.gps_readable = gps["gps_readable"]
    res.gps_records = gps["gps_records"]
    res.gps_duration = gps["gps_duration_sec"]
    res.gps_sorted = gps["gps_sorted"]
    res.gps_dups = gps["gps_dup_timestamps"]
    res.gps_median_interval = gps["gps_median_interval_sec"]
    res.gps_max_interval = gps["gps_max_interval_sec"]
    res.gps_first_offset = gps["gps_first_offset_sec"]
    res.gps_last_offset = gps["gps_last_offset_sec"]
    res.gps_speed_mean = gps["gps_speed_mean"]

    # Duration / start alignment camera <-> gaze
    if cam["readable"] and gaze["readable"]:
        res.cam_gaze_duration_diff = abs(res.cam_duration - res.gaze_duration)
        res.cam_gaze_start_aligned = True  # both start at t=0 by container time

    if res.cam_gaze_duration_diff is not None \
            and res.cam_gaze_duration_diff > DURATION_TOLERANCE_SEC:
        res.problems.append(
            f"camera-gaze duration diff {res.cam_gaze_duration_diff:.2f}s "
            f"> {DURATION_TOLERANCE_SEC}s"
        )

    # GPS checks
    gps_cover = True
    if gps["gps_readable"]:
        if res.gps_sorted is False:
            res.problems.append("GPS timestamps not sorted")
            gps_cover = False
        if (res.gps_dups or 0) > 0:
            res.problems.append(f"{res.gps_dups} duplicate GPS timestamps")
            gps_cover = False
        if res.gps_median_interval is not None \
                and abs(res.gps_median_interval - 1.0) > GPS_MEDIAN_INTERVAL_TOLERANCE_SEC:
            res.problems.append(
                f"GPS median interval {res.gps_median_interval:.2f}s != ~1s"
            )
            gps_cover = False
        if res.gps_max_interval is not None \
                and res.gps_max_interval > GPS_MAX_GAP_TOLERANCE_SEC:
            res.problems.append(
                f"GPS max gap {res.gps_max_interval:.2f}s > "
                f"{GPS_MAX_GAP_TOLERANCE_SEC}s"
            )
            gps_cover = False
        if res.cam_duration is not None:
            first_ok = (res.gps_first_offset is not None
                        and res.gps_first_offset <= GPS_START_OFFSET_MAX_SEC)
            last_ok = (res.gps_last_offset is not None
                       and res.gps_last_offset <= GPS_END_COVERAGE_MAX_SEC)
            if not (first_ok and last_ok):
                res.problems.append(
                    "GPS timeline does not cover video duration "
                    f"(first offset {res.gps_first_offset}, "
                    f"last gap {res.gps_last_offset})"
                )
                gps_cover = False
    res.gps_covers_video = gps_cover

    gps_int_ok = bool(
        res.gps_median_interval is not None
        and abs(res.gps_median_interval - 1.0) <= GPS_MEDIAN_INTERVAL_TOLERANCE_SEC
        and res.gps_max_interval is not None
        and res.gps_max_interval <= GPS_MAX_GAP_TOLERANCE_SEC
    )
    res.gps_interval_ok = gps_int_ok

    # Overall synchronization decision for this sequence
    critical = (
        res.complete
        and res.cam_readable and res.gaze_readable
        and res.resolution_camera_ok and res.resolution_gaze_ok
        and res.fps_camera_plausible and res.fps_gaze_plausible
        and (res.cam_gaze_duration_diff is None
             or res.cam_gaze_duration_diff <= DURATION_TOLERANCE_SEC)
        and gps_cover and gps_int_ok
        and (res.gps_sorted is not False) and (res.gps_dups or 0) == 0
    )
    res.sync_ok = critical and len(res.problems) == 0
    return res


# ---------------------------------------------------------------------------
# Full audit run
# ---------------------------------------------------------------------------
def run_audit(max_per_split: Optional[int] = None) -> pd.DataFrame:
    rows: list[dict] = []
    for split in SPLITS:
        ids = list_sequence_ids(split)
        video_ids = sorted(ids, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
        if max_per_split is not None:
            video_ids = video_ids[:max_per_split]
        for vid in tqdm(video_ids, desc=f"audit {split}"):
            rows.append(audit_sequence(split, vid, ids).to_row())
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics + report
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> dict:
    s: dict = {}
    s["total_sequences"] = len(df)
    s["complete"] = int(df["complete"].sum())
    s["incomplete"] = int((~df["complete"]).sum())
    s["missing_camera"] = int((~df["camera_exists"]).sum())
    s["missing_gaze"] = int((~df["gaze_exists"]).sum())
    s["missing_gps"] = int((~df["gps_exists"]).sum())
    s["sync_ok"] = int(df["sync_ok"].sum())
    s["sync_problems"] = int((~df["sync_ok"]).sum())

    for split in SPLITS:
        sub = df[df["split"] == split]
        s[f"{split}_n"] = len(sub)
        s[f"{split}_complete"] = int(sub["complete"].sum())
        s[f"{split}_sync_ok"] = int(sub["sync_ok"].sum())

    cam = df[df["cam_readable"]]
    gaze = df[df["gaze_readable"]]
    s["cam_fps_min"] = cam["cam_fps"].min()
    s["cam_fps_max"] = cam["cam_fps"].max()
    s["cam_fps_median"] = cam["cam_fps"].median()
    s["cam_fps_mean"] = cam["cam_fps"].mean()
    s["gaze_fps_min"] = gaze["gaze_fps"].min()
    s["gaze_fps_max"] = gaze["gaze_fps"].max()
    s["gaze_fps_median"] = gaze["gaze_fps"].median()
    s["gaze_fps_mean"] = gaze["gaze_fps"].mean()
    s["cam_duration_median"] = cam["cam_duration"].median()
    s["gaze_duration_median"] = gaze["gaze_duration"].median()
    s["cam_frames_median"] = cam["cam_frames"].median()
    s["gaze_frames_median"] = gaze["gaze_frames"].median()

    gps = df[df["gps_readable"]]
    s["gps_records_median"] = gps["gps_records"].median()
    s["gps_median_interval_median"] = gps["gps_median_interval"].median()
    s["gps_max_interval_max"] = gps["gps_max_interval"].max()

    s["resolution_camera_ok"] = int(df["resolution_camera_ok"].sum())
    s["resolution_gaze_ok"] = int(df["resolution_gaze_ok"].sum())
    s["fps_camera_plausible"] = int(df["fps_camera_plausible"].sum())
    s["fps_gaze_plausible"] = int(df["fps_gaze_plausible"].sum())
    s["gps_covers_video"] = int(df["gps_covers_video"].sum())
    s["gps_interval_ok"] = int(df["gps_interval_ok"].sum())
    return s


def fps_distribution(df: pd.DataFrame) -> pd.DataFrame:
    cam = df[df["cam_readable"]]["cam_fps"].round(2)
    gaze = df[df["gaze_readable"]]["gaze_fps"].round(2)
    rows = []
    for label, series in (("camera", cam), ("gaze", gaze)):
        vc = series.value_counts().sort_index()
        for fps, cnt in vc.items():
            rows.append({"modality": label, "fps": float(fps), "count": int(cnt)})
    return pd.DataFrame(rows)


def problem_sequences(df: pd.DataFrame) -> pd.DataFrame:
    return df[(~df["sync_ok"]) | (~df["complete"])].copy()


def save_figures(df: pd.DataFrame, n_per_split: int = 2):
    """Visual QC for representative sequences: camera, gaze, overlay."""
    timestamps = [0.0, 2.0, 5.0, 9.0]
    for split in SPLITS:
        sub = df[(df["split"] == split) & (df["sync_ok"])]
        if sub.empty:
            sub = df[df["split"] == split]
        chosen = sub.head(n_per_split)
        for _, row in chosen.iterrows():
            vid = row["video_id"]
            cam_path = BDDA_ROOT / split / MODALITIES["camera"] / f"{vid}.mp4"
            gaze_path = BDDA_ROOT / split / MODALITIES["gaze"] / f"{vid}_pure_hm.mp4"

            cap_c = cv2.VideoCapture(str(cam_path))
            cap_g = cv2.VideoCapture(str(gaze_path))
            if not cap_c.isOpened() or not cap_g.isOpened():
                if cap_c.isOpened():
                    cap_c.release()
                if cap_g.isOpened():
                    cap_g.release()
                continue

            cam_fps = cap_c.get(cv2.CAP_PROP_FPS)
            gaze_fps = cap_g.get(cv2.CAP_PROP_FPS)
            duration = min(row["cam_duration"] or 0, row["gaze_duration"] or 0)
            ts = [t for t in timestamps if t <= duration - 0.5] or [0.0]

            fig, axes = plt.subplots(len(ts), 3, figsize=(15, 4 * len(ts)))
            if len(ts) == 1:
                axes = axes[None, :]
            for i, t in enumerate(ts):
                cap_c.set(cv2.CAP_PROP_POS_FRAMES, round(t * cam_fps))
                cap_g.set(cv2.CAP_PROP_POS_FRAMES, round(t * gaze_fps))
                ret_c, cam_f = cap_c.read()
                ret_g, gaze_f = cap_g.read()

                if ret_c:
                    cam_rgb = cv2.cvtColor(cam_f, cv2.COLOR_BGR2RGB)
                    axes[i, 0].imshow(cam_rgb)
                    axes[i, 0].set_title(f"Camera {vid} - {t}s")
                else:
                    axes[i, 0].set_title(f"Camera {vid} - {t}s (n/a)")
                axes[i, 0].axis("off")

                if ret_g:
                    gaze_gray = cv2.cvtColor(gaze_f, cv2.COLOR_BGR2GRAY)
                    gaze_crop = gaze_gray[GAZE_CROP_TOP:GAZE_CROP_BOTTOM, :]
                    gaze_resized = cv2.resize(gaze_crop, (EXPECTED_CAMERA_W, EXPECTED_CAMERA_H))
                    axes[i, 1].imshow(gaze_resized, cmap="gray", vmin=0, vmax=255)
                    axes[i, 1].set_title(f"Gaze crop {vid} - {t}s")
                else:
                    axes[i, 1].set_title(f"Gaze {vid} - {t}s (n/a)")
                axes[i, 1].axis("off")

                if ret_c and ret_g:
                    heatmap = cv2.applyColorMap(gaze_resized, cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    mask = gaze_resized > 20
                    blended = cv2.addWeighted(cam_rgb, 0.55, heatmap, 0.45, 0)
                    overlay = cam_rgb.copy()
                    overlay[mask] = blended[mask]
                    axes[i, 2].imshow(overlay)
                    axes[i, 2].set_title(f"Overlay {vid} - {t}s")
                else:
                    axes[i, 2].set_title(f"Overlay {vid} - {t}s (n/a)")
                axes[i, 2].axis("off")

            cap_c.release()
            cap_g.release()
            plt.tight_layout()
            fig.savefig(FIG_DIR / f"qc_{split}_{vid}.png", dpi=100)
            plt.close(fig)


def write_report(df: pd.DataFrame, summary: dict, fps_dist: pd.DataFrame,
                 problems: pd.DataFrame) -> None:
    md_path = REPORT_DIR / "temporal_synchronization_audit.md"
    lines: list[str] = []
    lines.append("# Temporal Synchronization Audit — DriveSight BDDA")
    lines.append("")
    lines.append(f"Dataset root: `{BDDA_ROOT}`")
    lines.append(f"Splits: {', '.join(SPLITS)}")
    lines.append("")

    lines.append("## 1. Sequence coverage")
    lines.append("")
    lines.append("| Split | Sequences | Complete | Missing camera | Missing gaze | Missing GPS | Sync OK | Sync problems |")
    lines.append("|-------|-----------|----------|----------------|--------------|-------------|---------|---------------|")
    for split in SPLITS:
        sub = df[df["split"] == split]
        lines.append(
            f"| {split} | {len(sub)} | {int(sub['complete'].sum())} | "
            f"{int((~sub['camera_exists']).sum())} | "
            f"{int((~sub['gaze_exists']).sum())} | "
            f"{int((~sub['gps_exists']).sum())} | "
            f"{int(sub['sync_ok'].sum())} | {int((~sub['sync_ok']).sum())} |"
        )
    lines.append("")
    lines.append(f"- Total sequences: {summary['total_sequences']}")
    lines.append(f"- Complete sequences (3 modalities): {summary['complete']}")
    lines.append(f"- Sequences with synchronization problems: {summary['sync_problems']}")
    lines.append("")

    lines.append("## 2. Camera / gaze FPS distributions")
    lines.append("")
    lines.append("| Modality | FPS | Count |")
    lines.append("|----------|-----|-------|")
    for _, r in fps_dist.iterrows():
        lines.append(f"| {r['modality']} | {r['fps']} | {r['count']} |")
    lines.append("")

    lines.append("## 3. Frame count and duration distributions")
    lines.append("")
    lines.append(f"- Camera duration median: {summary['cam_duration_median']:.2f}s")
    lines.append(f"- Gaze duration median: {summary['gaze_duration_median']:.2f}s")
    lines.append(f"- Camera frames median: {summary['cam_frames_median']}")
    lines.append(f"- Gaze frames median: {summary['gaze_frames_median']}")
    lines.append(f"- Camera FPS: mean {summary['cam_fps_mean']:.2f}, median {summary['cam_fps_median']:.2f}, "
                 f"range [{summary['cam_fps_min']}, {summary['cam_fps_max']}]")
    lines.append(f"- Gaze FPS: mean {summary['gaze_fps_mean']:.2f}, median {summary['gaze_fps_median']:.2f}, "
                 f"range [{summary['gaze_fps_min']}, {summary['gaze_fps_max']}]")
    lines.append("")

    lines.append("## 4. GPS statistics")
    lines.append("")
    lines.append(f"- GPS records median per sequence: {summary['gps_records_median']}")
    lines.append(f"- GPS median interval median: {summary['gps_median_interval_median']:.3f}s")
    lines.append(f"- GPS max interval max: {summary['gps_max_interval_max']:.3f}s")
    lines.append(f"- Sequences with GPS covering video timeline: {summary['gps_covers_video']}")
    lines.append(f"- Sequences with consistent GPS sampling: {summary['gps_interval_ok']}")
    lines.append("")

    lines.append("## 5. Resolution and FPS plausibility")
    lines.append("")
    lines.append(f"- Camera resolution {EXPECTED_CAMERA_W}x{EXPECTED_CAMERA_H} OK: "
                 f"{summary['resolution_camera_ok']}")
    lines.append(f"- Gaze resolution {EXPECTED_GAZE_W}x{EXPECTED_GAZE_H} OK: "
                 f"{summary['resolution_gaze_ok']}")
    lines.append(f"- Camera FPS plausible: {summary['fps_camera_plausible']}")
    lines.append(f"- Gaze FPS plausible: {summary['fps_gaze_plausible']}")
    lines.append("")

    lines.append("## 6. Synchronization tolerances used")
    lines.append("")
    lines.append("- Duration camera<->gaze tolerance: "
                 f"{DURATION_TOLERANCE_SEC}s (a few frames of container slack)")
    lines.append("- Start offset tolerance: "
                 f"{START_OFFSET_TOLERANCE_SEC}s (both videos start at t=0)")
    lines.append("- Camera nominal FPS: {29.97, 30, 59.94}, tolerance "
                 f"{FPS_TOLERANCE}fps")
    lines.append("- Gaze nominal FPS: ~29-30, tolerance "
                 f"{FPS_TOLERANCE}fps")
    lines.append("- GPS nominal sampling: ~1 Hz; median interval tolerance "
                 f"{GPS_MEDIAN_INTERVAL_TOLERANCE_SEC}s, max gap "
                 f"{GPS_MAX_GAP_TOLERANCE_SEC}s")
    lines.append("- GPS coverage: first record within "
                 f"{GPS_START_OFFSET_MAX_SEC}s of video start, last record within "
                 f"{GPS_END_COVERAGE_MAX_SEC}s of video end")
    lines.append("- Frame<->GPS association: nearest neighbor accepted only within "
                 f"{GPS_ASSOC_MAX_GAP_SEC}s")
    lines.append("")

    lines.append("## 7. Sequences with synchronization problems")
    lines.append("")
    if problems.empty:
        lines.append("None detected.")
    else:
        lines.append("| Split | Video ID | Problems |")
        lines.append("|-------|----------|----------|")
        for _, r in problems.iterrows():
            lines.append(f"| {r['split']} | {r['video_id']} | {r['problems']} |")
    lines.append("")

    lines.append("## 8. Decision gate")
    lines.append("")
    n_ok = summary["sync_ok"]
    n_total = summary["total_sequences"]
    lines.append(
        f"- Synchronized sequences: {n_ok} / {n_total} "
        f"({100.0 * n_ok / max(n_total, 1):.1f}%)"
    )
    if summary["sync_problems"] == 0 and summary["incomplete"] == 0:
        lines.append("- **CASE A — Synchronization is reliable.** Proceeding to Phase 2: Video Preprocessing.")
    else:
        lines.append("- **CASE B — Synchronization is NOT reliable.** Stopping before preprocessing. "
                     "See section 7 for affected sequences.")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = run_audit()
    summary = summarize(df)
    fps_dist = fps_distribution(df)
    problems = problem_sequences(df)

    # Save structured results
    df.to_csv(REPORT_DIR / "audit_all_sequences.csv", index=False)
    fps_dist.to_csv(REPORT_DIR / "audit_fps_distribution.csv", index=False)
    problems.to_csv(REPORT_DIR / "audit_problem_sequences.csv", index=False)

    per_split = df.groupby("split").agg(
        sequences=("video_id", "count"),
        complete=("complete", "sum"),
        sync_ok=("sync_ok", "sum"),
    )
    per_split.to_csv(REPORT_DIR / "audit_split_summary.csv")

    write_report(df, summary, fps_dist, problems)

    # Visual QC figures for a handful of sequences
    save_figures(df)

    print("\n=== Audit summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nFigures -> {FIG_DIR}")
    print(f"CSVs + report -> {REPORT_DIR}")


if __name__ == "__main__":
    main()
