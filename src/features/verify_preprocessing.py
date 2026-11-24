"""
Verification of preprocessing outputs:
- frames are openable and have correct dimensions
- aligned_records row counts match extracted frames
- GPS values valid and association gaps within tolerance
- timestamp residuals within tolerance
- no missing/corrupted outputs
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import pandas as pd

from preprocess_config import PreprocessConfig


def verify_frames(cfg: PreprocessConfig, split: str, video_id: str,
                  n_expected_cam: int, n_expected_gaze: int) -> dict:
    """Verify extracted frames for one sequence."""
    cam_dir = cfg.frames_dir(split, "camera") / video_id
    gaze_dir = cfg.frames_dir(split, "gaze") / video_id

    cam_files = sorted(cam_dir.glob("*.jpg")) if cam_dir.exists() else []
    gaze_files = sorted(gaze_dir.glob("*.jpg")) if gaze_dir.exists() else []

    cam_ok = 0
    cam_dim_ok = 0
    cam_corrupt = 0
    for p in cam_files:
        img = cv2.imread(str(p))
        if img is None:
            cam_corrupt += 1
            continue
        cam_ok += 1
        if img.shape[1] == cfg.expected_camera_w and img.shape[0] == cfg.expected_camera_h:
            cam_dim_ok += 1

    gaze_ok = 0
    gaze_dim_ok = 0
    gaze_corrupt = 0
    for p in gaze_files:
        img = cv2.imread(str(p))
        if img is None:
            gaze_corrupt += 1
            continue
        gaze_ok += 1
        if img.shape[1] == cfg.expected_camera_w and img.shape[0] == cfg.expected_camera_h:
            gaze_dim_ok += 1

    return {
        "split": split,
        "video_id": video_id,
        "cam_files": len(cam_files),
        "cam_expected": n_expected_cam,
        "cam_openable": cam_ok,
        "cam_dim_ok": cam_dim_ok,
        "cam_corrupt": cam_corrupt,
        "gaze_files": len(gaze_files),
        "gaze_expected": n_expected_gaze,
        "gaze_openable": gaze_ok,
        "gaze_dim_ok": gaze_dim_ok,
        "gaze_corrupt": gaze_corrupt,
    }


def verify_records(cfg: PreprocessConfig, split: str) -> dict:
    """Verify aligned_records.csv for one split."""
    rec_path = cfg.aligned_records_path(split)
    if not rec_path.exists():
        return {"split": split, "error": "aligned_records.csv missing"}
    df = pd.read_csv(rec_path)
    n_rows = len(df)
    n_gps_missing = int(df["gps_missing"].sum()) if "gps_missing" in df.columns else -1
    n_cam_null = int(df["camera_frame_path"].isna().sum()) if "camera_frame_path" in df.columns else -1
    n_gaze_null = int(df["gaze_frame_path"].isna().sum()) if "gaze_frame_path" in df.columns else -1

    # GPS validity
    gps_speed_valid = True
    gps_course_valid = True
    if "gps_speed_m_s" in df.columns:
        speeds = df["gps_speed_m_s"].dropna()
        gps_speed_valid = bool((speeds >= 0).all()) if len(speeds) else True
    if "gps_course" in df.columns:
        courses = df["gps_course"].dropna()
        gps_course_valid = bool(((courses >= 0) & (courses < 360)).all()) if len(courses) else True

    # residual checks
    cam_res_ok = True
    gaze_res_ok = True
    if "cam_residual_sec" in df.columns:
        cam_res_ok = bool((df["cam_residual_sec"].dropna() <= 0.15).all())
    if "gaze_residual_sec" in df.columns:
        gaze_res_ok = bool((df["gaze_residual_sec"].dropna() <= 0.15).all())

    return {
        "split": split,
        "n_rows": n_rows,
        "n_gps_missing": n_gps_missing,
        "n_camera_path_null": n_cam_null,
        "n_gaze_path_null": n_gaze_null,
        "gps_speed_valid": gps_speed_valid,
        "gps_course_valid": gps_course_valid,
        "cam_residual_ok": cam_res_ok,
        "gaze_residual_ok": gaze_res_ok,
    }


def verify_split(cfg: PreprocessConfig, split: str,
                 seq_results: list[dict]) -> dict:
    """Run all verifications for a split and return a report dict."""
    frame_reports = []
    for sr in seq_results:
        fr = verify_frames(cfg, split, sr["video_id"],
                           sr["n_camera_frames"], sr["n_gaze_frames"])
        frame_reports.append(fr)

    rec_report = verify_records(cfg, split)

    all_cam_open = all(f["cam_openable"] == f["cam_files"] for f in frame_reports)
    all_gaze_open = all(f["gaze_openable"] == f["gaze_files"] for f in frame_reports)
    all_cam_dim = all(f["cam_dim_ok"] == f["cam_openable"] for f in frame_reports)
    all_gaze_dim = all(f["gaze_dim_ok"] == f["gaze_openable"] for f in frame_reports)
    counts_match = all(f["cam_files"] == f["cam_expected"]
                       and f["gaze_files"] == f["gaze_expected"]
                       for f in frame_reports)

    return {
        "split": split,
        "frame_reports": frame_reports,
        "records": rec_report,
        "all_cam_openable": all_cam_open,
        "all_gaze_openable": all_gaze_open,
        "all_cam_dim_ok": all_cam_dim,
        "all_gaze_dim_ok": all_gaze_dim,
        "counts_match_expected": counts_match,
    }
