"""
Phase 5 features for an arbitrary dash-cam video (demo app): the same chain as on BDDA, without gaze
and GPS, which a plain video does not have.

    detections per sampled frame (5 Hz, any resolution)
      -> boxes rescaled to 1280x720, Phase 3 columns
      -> ByteTrack with the frozen Phase 4 setting (src/tracking)
      -> object and frame features of src/fusion/fusion_core.py (gaze and GPS columns empty)

Only the vision features (objects + motion) are meaningful here; the model to use is the one trained
on VISION_FEATURES (data/risk/risk_model_vision_v5.joblib).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.fusion.fusion_config import VRU_CLASSES, FusionConfig
from src.fusion.fusion_core import frame_features, object_features

CLASS_NAMES = ["pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle", "traffic light", "traffic sign"]
W, H = 1280.0, 720.0


def boxes_table(dets: dict, width: int, height: int, fps: float = 5.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """dets: {t_sec: DataFrame(cls, conf, x1, y1, x2, y2) in source pixels} -> Phase 3 boxes and frames."""
    sx, sy = W / width, H / height
    out, frames = [], []
    for t in sorted(dets):
        k, t_ms = int(round(t * fps)), int(round(t * 1000))
        frames.append({"sample_idx": k, "t_ms": t_ms})
        d = dets[t]
        if d is None or not len(d):
            continue
        d = d.sort_values("conf", ascending=False).reset_index(drop=True)
        x1, x2 = d["x1"].to_numpy(float) * sx, d["x2"].to_numpy(float) * sx
        y1, y2 = d["y1"].to_numpy(float) * sy, d["y2"].to_numpy(float) * sy
        cls = d["cls"].astype(str).to_numpy()
        out.append(pd.DataFrame({
            "split": "demo", "video_id": "upload", "sample_idx": k, "t_ms": t_ms, "det_idx": np.arange(len(d)),
            "cls_id": [CLASS_NAMES.index(c) if c in CLASS_NAMES else -1 for c in cls], "cls_name": cls,
            "is_vru": np.isin(cls, VRU_CLASSES), "conf": d["conf"].to_numpy(float),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "cx_n": (x1 + x2) / 2 / W, "cy_n": (y1 + y2) / 2 / H,
            "w_n": (x2 - x1) / W, "h_n": (y2 - y1) / H, "area_frac": (x2 - x1) * (y2 - y1) / (W * H),
            "bottom_y_n": y2 / H, "x_offset_n": (x1 + x2) / 2 / W - 0.5,
            "touches_border": (x1 <= 1) | (y1 <= 1) | (x2 >= W - 1) | (y2 >= H - 1)}))
    boxes = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["split", "video_id", "sample_idx", "t_ms", "det_idx", "cls_id", "cls_name", "is_vru", "conf",
                 "x1", "y1", "x2", "y2", "cx_n", "cy_n", "w_n", "h_n", "area_frac", "bottom_y_n", "x_offset_n",
                 "touches_border"])
    return boxes, pd.DataFrame(frames)


def empty_rows(frames: pd.DataFrame) -> pd.DataFrame:
    """Phase 2-like rows without gaze and GPS."""
    r = frames.copy()
    r["split"], r["video_id"] = "demo", "upload"
    r["t_sec"] = r["t_ms"] / 1000.0
    for c in ("gaze_peak_x", "gaze_peak_y", "gaze_centroid_x", "gaze_centroid_y", "gaze_spread", "gaze_active_frac",
              "gaze_n_regions", "gps_rec_t_sec", "gps_speed_m_s"):
        r[c] = np.nan
    r["gaze_ok"], r["gaze_empty"] = False, True
    r["gps_matched"], r["gps_speed_valid"] = False, False
    return r


def video_features(dets: dict, width: int, height: int, cfg: FusionConfig | None = None):
    """Returns (frame features, object features) for one uploaded video."""
    from src.tracking.track_config import TrackConfig
    from src.tracking.tracker_core import track_sequence

    cfg = cfg or FusionConfig()
    boxes, frames = boxes_table(dets, width, height, cfg.sampling_fps)
    if len(boxes):
        tracks, _ = track_sequence(boxes, frames, TrackConfig())
        tracks["split"], tracks["video_id"] = "demo", "upload"
    else:
        tracks = boxes.assign(track_id=pd.Series(dtype=int))
    rows = empty_rows(frames)
    obj = object_features(tracks, rows, cfg)
    fr = frame_features(rows, obj, cfg)
    return fr, obj


def reasons(row: pd.Series) -> list[str]:
    """Short cues behind a frame's score, for the UI only."""
    r = []
    if row.get("n_vru_near_path", 0) > 0:
        r.append("vulnerable road user near the path")
    if row.get("lead_present", 0) and row.get("lead_h_n", 0) > 0.25:
        r.append("road user close ahead")
    ttc = row.get("min_ttc_near", np.nan)
    if np.isfinite(ttc) and ttc < 2.5:
        r.append("closing in fast")
    if row.get("n_new_near_1s", 0) > 0 and row.get("vru_near_path", 0):
        r.append("new road user near the path")
    return r
