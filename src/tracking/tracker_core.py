"""
Core of Phase 4: run Ultralytics' BYTETracker over the stored boxes of ONE sequence, and summarise tracks.

Pure functions (no file I/O), used by run_tracking.py, tune_tracking.py and the tests.

How BYTETracker is driven (read from ultralytics/trackers/byte_tracker.py, version 8.4.189):
  - update(results) expects a NumPy object exposing `conf`, `cls` and `xywh` (box CENTRE x, y, width, height),
    supporting len() and boolean indexing. `Detections` below is that object.
  - it must be called for EVERY sampled frame in time order, including frames with no box, so that lost
    tracks age correctly (their lifetime is counted in frames).
  - each STrack keeps `idx`, the position of its matched box in the array passed for that frame. Boxes are
    passed in det_idx order, so idx maps back to the Phase 3 det_idx. This link is what lets Phase 5 join a
    track to its gaze and GPS rows.
  - update() returns only CONFIRMED tracks. A new track is confirmed when it is matched again in the next
    frame (or immediately in the very first frame). To keep the first box of each object, we read the
    tracker's state after every update (`tracked_stracks` whose last frame is the current one), which also
    contains not-yet-confirmed tracks, and keep the birth box of every track that is confirmed later.
A new tracker is created for every sequence, so track ids restart at 1 per sequence and never cross a clip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.tracking.track_config import TrackConfig

BOX_KEEP = ["cls_id", "cls_name", "is_vru", "conf", "x1", "y1", "x2", "y2", "cx_n", "cy_n", "w_n", "h_n",
            "area_frac", "bottom_y_n", "x_offset_n", "touches_border"]
TRACK_COLUMNS = ["split", "video_id", "sample_idx", "t_ms", "det_idx", "track_id", "track_frame",
                 "is_birth"] + BOX_KEEP
SUMMARY_COLUMNS = ["split", "video_id", "track_id", "cls_name", "cls_majority_frac", "n_classes", "is_vru",
                   "first_t_ms", "last_t_ms", "duration_sec", "n_boxes", "span_frames", "n_gaps", "max_gap_frames",
                   "mean_conf", "max_conf", "first_cx_n", "last_cx_n", "d_cx_n", "first_cy_n", "last_cy_n", "d_cy_n",
                   "first_bottom_y_n", "last_bottom_y_n", "d_bottom_y_n", "first_h_n", "last_h_n", "log_area_ratio",
                   "max_center_jump", "min_consecutive_iou", "touches_border_frac"]


class Detections:
    """Minimal NumPy 'Results-like' object for BYTETracker.update: conf, cls, xywh, len(), boolean indexing."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    @classmethod
    def from_boxes(cls, b: pd.DataFrame) -> "Detections":
        x1, y1, x2, y2 = (b[k].to_numpy(np.float32) for k in ("x1", "y1", "x2", "y2"))
        xywh = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1) if len(b) else np.zeros((0, 4))
        return cls(xywh, b["conf"].to_numpy(np.float32), b["cls_id"].to_numpy(np.float32))

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask) -> "Detections":
        return Detections(self.xywh[mask], self.conf[mask], self.cls[mask])


def new_tracker(cfg: TrackConfig):
    from ultralytics.trackers.byte_tracker import BYTETracker
    return BYTETracker(cfg.tracker_args())


def track_sequence(boxes: pd.DataFrame, frames: pd.DataFrame, cfg: TrackConfig) -> tuple[pd.DataFrame, dict]:
    """Track one sequence.

    boxes:  Phase 3 box table of the sequence (split, video_id, sample_idx, t_ms, det_idx, cls_id, conf, x1..y2, ...)
    frames: one row per sampled frame (sample_idx, t_ms), including frames without boxes
    Returns the tracked boxes (TRACK_COLUMNS) and statistics.
    """
    from ultralytics.trackers.basetrack import TrackState

    tracker = new_tracker(cfg)
    frames = frames.sort_values("t_ms").reset_index(drop=True)
    by_t = {t: g.sort_values("det_idx") for t, g in boxes.groupby("t_ms")} if len(boxes) else {}
    empty = boxes.iloc[0:0]
    recs = []                                          # (t_ms, sample_idx, det_idx, track_id, confirmed)
    for fr in frames.itertuples(index=False):
        b = by_t.get(int(fr.t_ms), empty)
        det_idx = b["det_idx"].to_numpy(int) if len(b) else np.zeros(0, int)
        out = tracker.update(Detections.from_boxes(b))
        seen = {}
        for s in tracker.tracked_stracks:
            if s.state == TrackState.Tracked and s.end_frame == tracker.frame_id:
                i = int(s.idx)
                seen[int(s.track_id)] = (int(det_idx[i]), bool(s.is_activated))
        # consistency with what update() returns: every confirmed output row is in `seen`, same box
        for row in np.atleast_2d(out) if len(out) else []:
            tid, i = int(row[4]), int(row[7])
            assert tid in seen and seen[tid][0] == int(det_idx[i]), "tracker output and state disagree"
        for tid, (d, conf_ok) in seen.items():
            recs.append((int(fr.t_ms), int(fr.sample_idx), d, tid, conf_ok))

    r = pd.DataFrame(recs, columns=["t_ms", "sample_idx", "det_idx", "track_id", "confirmed"])
    confirmed_ids = set(r.loc[r["confirmed"], "track_id"])
    n_unconfirmed = r.loc[~r["track_id"].isin(confirmed_ids), "track_id"].nunique()
    keep = r["track_id"].isin(confirmed_ids) & (r["confirmed"] | cfg.keep_birth_box)
    r = r[keep].copy()
    assert not r.duplicated(["track_id", "t_ms"]).any(), "a track has two boxes in one frame"
    assert not r.duplicated(["t_ms", "det_idx"]).any(), "a box is assigned to two tracks"

    t = r.merge(boxes, on=["t_ms", "sample_idx", "det_idx"], how="left", validate="one_to_one")
    assert t["conf"].notna().all(), "tracked box not found in the Phase 3 table"
    t = t.sort_values(["track_id", "t_ms"]).reset_index(drop=True)
    first = t.groupby("track_id")["t_ms"].transform("min")
    t["is_birth"] = t["t_ms"] == first
    t["track_frame"] = t.groupby("track_id").cumcount()
    t = t.reindex(columns=TRACK_COLUMNS).sort_values(["t_ms", "det_idx"]).reset_index(drop=True)

    conf_ok = boxes["conf"] >= 0.5 if len(boxes) else pd.Series(dtype=bool)
    tracked_keys = set(zip(t["t_ms"], t["det_idx"]))
    hi_keys = list(zip(boxes.loc[conf_ok, "t_ms"], boxes.loc[conf_ok, "det_idx"])) if len(boxes) else []
    stats = {
        "n_frames": int(len(frames)), "n_boxes_in": int(len(boxes)),
        "n_boxes_usable": int((boxes["conf"] > cfg.track_low_thresh).sum()) if len(boxes) else 0,
        "n_boxes_tracked": int(len(t)), "n_tracks": int(t["track_id"].nunique()),
        "n_tracks_vru": int(t.loc[t["is_vru"].astype(bool), "track_id"].nunique()) if len(t) else 0,
        "n_unconfirmed_dropped": int(n_unconfirmed),
        "n_boxes_conf50": len(hi_keys),
        "n_boxes_conf50_tracked": int(sum(k in tracked_keys for k in hi_keys)),
    }
    return t, stats


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = ix * iy
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / np.maximum(area(a) + area(b) - inter, 1e-9)


def summarize_tracks(t: pd.DataFrame) -> pd.DataFrame:
    """One row per track (SUMMARY_COLUMNS). Displacements are last minus first box, in normalised image units.
    Vectorised (groupby), so it can run inside the tuning grid."""
    if len(t) == 0:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    k = ["split", "video_id", "track_id"]
    o = t.sort_values(k + ["t_ms"]).reset_index(drop=True)
    g = o.groupby(k, sort=True)
    # majority class: most boxes, ties -> higher mean confidence
    c = o.groupby(k + ["cls_name"]).agg(n=("conf", "size"), m=("conf", "mean")).reset_index()
    c = c.sort_values(k + ["n", "m"], ascending=[True, True, True, False, False])
    maj = c.drop_duplicates(k).set_index(k)
    ncls = c.groupby(k).size()
    first, last = g.first(), g.last()
    n = g.size()
    # consecutive pairs inside a track
    nxt = g.shift(-1)
    pair = nxt["t_ms"].notna()
    gap = (nxt["sample_idx"] - o["sample_idx"] - 1).where(pair)
    jump = np.hypot(nxt["cx_n"] - o["cx_n"], nxt["cy_n"] - o["cy_n"]).where(pair)
    ix = (np.minimum(o["x2"], nxt["x2"]) - np.maximum(o["x1"], nxt["x1"])).clip(lower=0)
    iy = (np.minimum(o["y2"], nxt["y2"]) - np.maximum(o["y1"], nxt["y1"])).clip(lower=0)
    inter = ix * iy
    union = (o["x2"] - o["x1"]) * (o["y2"] - o["y1"]) + (nxt["x2"] - nxt["x1"]) * (nxt["y2"] - nxt["y1"]) - inter
    iou = (inter / union.clip(lower=1e-9)).where(pair)
    pk = o[k]
    agg_pair = pd.DataFrame({"gap": gap, "has_gap": (gap > 0).where(pair), "jump": jump, "iou": iou}).join(pk) \
        .groupby(k).agg(n_gaps=("has_gap", "sum"), max_gap=("gap", "max"), max_jump=("jump", "max"), min_iou=("iou", "min"))
    out = pd.DataFrame({
        "cls_name": maj["cls_name"], "cls_majority_frac": maj["n"] / n, "n_classes": ncls,
        "first_t_ms": first["t_ms"], "last_t_ms": last["t_ms"], "duration_sec": (last["t_ms"] - first["t_ms"]) / 1000,
        "n_boxes": n, "span_frames": last["sample_idx"] - first["sample_idx"] + 1,
        "n_gaps": agg_pair["n_gaps"].fillna(0), "max_gap_frames": agg_pair["max_gap"].fillna(0),
        "mean_conf": g["conf"].mean(), "max_conf": g["conf"].max(),
        "first_cx_n": first["cx_n"], "last_cx_n": last["cx_n"], "d_cx_n": last["cx_n"] - first["cx_n"],
        "first_cy_n": first["cy_n"], "last_cy_n": last["cy_n"], "d_cy_n": last["cy_n"] - first["cy_n"],
        "first_bottom_y_n": first["bottom_y_n"], "last_bottom_y_n": last["bottom_y_n"],
        "d_bottom_y_n": last["bottom_y_n"] - first["bottom_y_n"], "first_h_n": first["h_n"], "last_h_n": last["h_n"],
        "log_area_ratio": np.log(last["area_frac"].clip(lower=1e-9) / first["area_frac"].clip(lower=1e-9)),
        "max_center_jump": agg_pair["max_jump"].fillna(0.0), "min_consecutive_iou": agg_pair["min_iou"],
        "touches_border_frac": g["touches_border"].mean(),
    })
    out["is_vru"] = out["cls_name"].isin(["pedestrian", "rider", "motorcycle", "bicycle"])
    out = out.reset_index()
    for col in ("track_id", "first_t_ms", "last_t_ms", "n_boxes", "span_frames", "n_gaps", "max_gap_frames", "n_classes"):
        out[col] = out[col].astype(int)
    for col in ("cls_majority_frac", "mean_conf", "max_conf", "touches_border_frac"):
        out[col] = out[col].astype(float)
    return out.reindex(columns=SUMMARY_COLUMNS)
