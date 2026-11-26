"""
Fusion of one sequence: Phase 2 rows (gaze descriptors, GPS) + Phase 4 tracked boxes -> per-object
features (one row per tracked box) and per-frame features (one row per Phase 2 row), plus the label.

Every feature at time t uses only data at time <= t:
  - object motion compares the box at t with the SAME track's box about `motion_lag_sec` earlier,
  - gaze shifts compare the gaze centroid at t with earlier rows,
  - GPS kinematics use the rebuilt 1 Hz records with t_rec <= t (gps_labels.causal_kinematics).
Clip time is never a feature. Features that grow with clip time are capped (track age <= 1 s) and the
first `motion_lag_sec + tol` of each clip is flagged `warmup` (motion and gaze shifts undefined there);
warm-up rows are out of the training/evaluation scope.

Gaze: Phase 2 keeps the gaze-map centroid, peak and spread (normalised, same frame as the camera). The
attention on a box is a Gaussian kernel of the distance from the gaze centroid to the box (0 inside),
with sigma = max(spread, gaze_min_sigma), in image-height units. It approximates "share of attention on
the object" without re-decoding the maps. The map is an in-lab attention reference (average of
observers), not the recording driver's gaze.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.fusion.fusion_config import ROAD_USER_CLASSES, VEHICLE_CLASSES, VRU_CLASSES, FusionConfig
from src.fusion.gps_labels import causal_kinematics, gps_records, label_rows, sequence_event

KEY = ["split", "video_id", "sample_idx", "t_ms"]

OBJECT_COLUMNS = KEY + ["det_idx", "track_id", "cls_name", "is_vru", "is_vehicle", "conf", "counted",
                        "cx_n", "bottom_y_n", "h_n", "area_frac", "x_offset_n", "touches_border",
                        "in_path", "near_path", "age_sec", "scale_rate", "ttc_sec", "lateral_closing",
                        "bottom_rate", "gaze_dist", "attention", "gaze_peak_in_box", "threat"]

# ---- feature groups (used for ablations) -----------------------------------------------------------
OBJECT_FEATURES = ["n_road_users", "n_vru", "n_vehicle", "n_inpath", "n_vru_near_path", "n_traffic_light",
                   "lead_present", "lead_is_vru", "lead_h_n", "lead_bottom_y", "lead_age_sec",
                   "vru_present", "vru_h_n", "vru_bottom_y", "vru_abs_offset", "vru_near_path", "vru_max_conf",
                   "max_threat"]
MOTION_FEATURES = ["lead_scale_rate", "lead_ttc_sec", "lead_bottom_rate", "vru_scale_rate", "vru_ttc_sec",
                   "vru_lateral_closing", "min_ttc_inpath", "min_ttc_near", "max_scale_rate_near",
                   "n_new_near_1s"]
GAZE_FEATURES = ["gaze_spread", "gaze_peak_x", "gaze_peak_y", "gaze_centroid_dist", "gaze_active_frac",
                 "gaze_n_regions", "gaze_shift_s", "gaze_shift_l", "lead_attention", "vru_attention",
                 "gaze_on_road_max", "unattended_threat", "attended_threat_share"]
KINEMATIC_FEATURES = ["speed_kmh", "accel_past", "speed_trend", "gps_valid"]
# gps_age_sec (time since the last 1 Hz record) is stored but is NOT a model input: GPS records fall at
# x.5 s of clip time and braking onsets too, so the age gives away the clip time within each second
# (alone it reached ROC-AUC 0.61 on validation).
STORED_ONLY = ["gps_age_sec"]
ALL_FEATURES = OBJECT_FEATURES + MOTION_FEATURES + GAZE_FEATURES + KINEMATIC_FEATURES
VISION_FEATURES = OBJECT_FEATURES + MOTION_FEATURES          # what an arbitrary dash-cam video gives

FRAME_COLUMNS = KEY + ["t_sec", "warmup"] + ALL_FEATURES + STORED_ONLY + ["has_event", "onset_sec", "lead_sec", "y", "in_scope"]


# ------------------------------------------------------------------------------------------- geometry
def corridor_half_width(bottom_y: np.ndarray, cfg: FusionConfig) -> np.ndarray:
    f = np.clip((bottom_y - cfg.path_y_min) / (1.0 - cfg.path_y_min), 0.0, 1.0)
    return cfg.path_half_w_top + cfg.path_half_w_growth * f


def in_path(cx: np.ndarray, bottom_y: np.ndarray, cfg: FusionConfig, margin: float = 0.0) -> np.ndarray:
    return (bottom_y >= cfg.path_y_min) & (np.abs(cx - 0.5) <= corridor_half_width(bottom_y, cfg) + margin)


def point_box_distance(px, py, x1, y1, x2, y2, aspect: float) -> np.ndarray:
    """Distance from a point to a box (0 inside), normalised coords, x scaled to image-height units."""
    dx = np.maximum.reduce([x1 - px, np.zeros_like(px), px - x2]) * aspect
    dy = np.maximum.reduce([y1 - py, np.zeros_like(py), py - y2])
    return np.hypot(dx, dy)


def threat_score(o: pd.DataFrame) -> np.ndarray:
    """A-priori threat of one object (not fitted): where it is x how big/close x how fast it grows.

        path weight   1.0 in the ego corridor, 0.5 near it, 0.15 elsewhere
        closeness     box height / 0.25 of the image (vehicles) or / 0.15 (VRU), clipped to [0, 1]
        approach      0.5 + 0.5 * clip(scale_rate / 0.5 per s, 0, 1)   (unknown motion -> 0.5)
        VRU factor    1.3
    """
    w = np.where(o["in_path"], 1.0, np.where(o["near_path"], 0.5, 0.15))
    close = np.clip(o["h_n"].to_numpy(float) / np.where(o["is_vru"], 0.15, 0.25), 0, 1)
    appr = 0.5 + 0.5 * np.clip(np.nan_to_num(o["scale_rate"].to_numpy(float), nan=0.0) / 0.5, 0, 1)
    return w * close * appr * np.where(o["is_vru"], 1.3, 1.0)


# ------------------------------------------------------------------------------------------ objects
def object_features(tracks: pd.DataFrame, rows: pd.DataFrame, cfg: FusionConfig) -> pd.DataFrame:
    if tracks.empty:
        return pd.DataFrame(columns=OBJECT_COLUMNS)
    o = tracks.sort_values(["track_id", "t_ms"]).reset_index(drop=True).copy()
    o["video_id"] = o["video_id"].astype(str)
    o["is_vru"] = o["cls_name"].isin(VRU_CLASSES).to_numpy()
    o["is_vehicle"] = o["cls_name"].isin(VEHICLE_CLASSES).to_numpy()
    o["counted"] = o["conf"].to_numpy(float) >= np.where(o["is_vru"], cfg.conf_vru, cfg.conf_other)
    cx, by = o["cx_n"].to_numpy(float), o["bottom_y_n"].to_numpy(float)
    o["in_path"] = in_path(cx, by, cfg)
    o["near_path"] = in_path(cx, by, cfg, cfg.near_path_margin)
    o["age_sec"] = (o["t_ms"] - o.groupby("track_id")["t_ms"].transform("min")) / 1000.0

    # same track, about motion_lag_sec earlier (strictly in the past)
    lag, tol = int(round(cfg.motion_lag_sec * 1000)), int(round(cfg.motion_lag_tol_sec * 1000))
    tid = o["track_id"].astype(np.int64)
    left = pd.DataFrame({"track_id": tid, "_target": o["t_ms"].astype(np.int64) - lag,
                         "_row": np.arange(len(o))}).sort_values("_target")
    right = pd.DataFrame({"track_id": tid, "_target": o["t_ms"].astype(np.int64),
                          "_t_past": o["t_ms"].astype(np.int64), "_h_past": o["h_n"].astype(float),
                          "_x_past": o["x_offset_n"].astype(float), "_b_past": o["bottom_y_n"].astype(float)}
                         ).sort_values("_target")
    m = pd.merge_asof(left, right, on="_target", by="track_id", direction="nearest", tolerance=tol)
    m = m.sort_values("_row").reset_index(drop=True)
    past_ok = (m["_t_past"] < o["t_ms"]).to_numpy() & m["_t_past"].notna().to_numpy()
    dt = (o["t_ms"].to_numpy(float) - m["_t_past"].to_numpy(float)) / 1000.0
    h, hp = o["h_n"].to_numpy(float), m["_h_past"].to_numpy(float)
    big = (h >= cfg.min_box_h_n) & (hp >= cfg.min_box_h_n)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(past_ok & big, (np.log(h) - np.log(hp)) / dt, np.nan)
        o["scale_rate"] = sr
        o["ttc_sec"] = np.where(np.isnan(sr), np.nan,
                                np.where(sr > 1.0 / cfg.ttc_cap_sec, 1.0 / sr, cfg.ttc_cap_sec))
        o["lateral_closing"] = np.where(past_ok, (np.abs(m["_x_past"].to_numpy(float)) -
                                                  np.abs(o["x_offset_n"].to_numpy(float))) / dt, np.nan)
        o["bottom_rate"] = np.where(past_ok, (by - m["_b_past"].to_numpy(float)) / dt, np.nan)

    # gaze at the same sample
    g = rows[["t_ms", "gaze_centroid_x", "gaze_centroid_y", "gaze_peak_x", "gaze_peak_y", "gaze_spread",
              "gaze_empty"]].drop_duplicates("t_ms")
    o = o.merge(g, on="t_ms", how="left")
    W, H = 1280.0, 720.0
    x1, y1, x2, y2 = (o[c].to_numpy(float) for c in ("x1", "y1", "x2", "y2"))
    x1, x2, y1, y2 = x1 / W, x2 / W, y1 / H, y2 / H
    gx, gy = o["gaze_centroid_x"].to_numpy(float), o["gaze_centroid_y"].to_numpy(float)
    gaze_ok = ~o["gaze_empty"].fillna(True).astype(bool).to_numpy() & np.isfinite(gx)
    d = point_box_distance(gx, gy, x1, y1, x2, y2, cfg.aspect)
    sigma = np.maximum(o["gaze_spread"].to_numpy(float), cfg.gaze_min_sigma)
    o["gaze_dist"] = np.where(gaze_ok, d, np.nan)
    o["attention"] = np.where(gaze_ok, np.exp(-0.5 * (d / sigma) ** 2), np.nan)
    px, py = o["gaze_peak_x"].to_numpy(float), o["gaze_peak_y"].to_numpy(float)
    o["gaze_peak_in_box"] = gaze_ok & (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)
    o["threat"] = threat_score(o)
    return o[OBJECT_COLUMNS].sort_values(["t_ms", "det_idx"]).reset_index(drop=True)


# ------------------------------------------------------------------------------------------- frames
def _pick(df: pd.DataFrame, mask: pd.Series, by: str, cols: dict, keys: pd.Index) -> pd.DataFrame:
    """For each t_ms, the row of df[mask] with the largest `by`; columns renamed by `cols`."""
    s = df[mask]
    if s.empty:
        return pd.DataFrame(index=keys, columns=list(cols.values()), dtype=float)
    s = s.sort_values(["t_ms", by, "det_idx"], ascending=[True, False, True]).drop_duplicates("t_ms")
    return s.set_index("t_ms")[list(cols)].rename(columns=cols).reindex(keys)


def frame_features(rows: pd.DataFrame, obj: pd.DataFrame, cfg: FusionConfig) -> pd.DataFrame:
    rows = rows.sort_values("t_ms").reset_index(drop=True)
    keys = pd.Index(rows["t_ms"].to_numpy(), name="t_ms")
    f = pd.DataFrame(index=keys)
    o = obj[obj["counted"]] if len(obj) else obj
    road = o[o["cls_name"].isin(ROAD_USER_CLASSES)] if len(o) else o

    def count(mask):
        return mask.groupby(o["t_ms"]).sum().reindex(keys, fill_value=0).astype(int) if len(o) else 0

    f["n_road_users"] = count(o["cls_name"].isin(ROAD_USER_CLASSES)) if len(o) else 0
    f["n_vru"] = count(o["is_vru"]) if len(o) else 0
    f["n_vehicle"] = count(o["is_vehicle"]) if len(o) else 0
    f["n_inpath"] = count(o["in_path"] & o["cls_name"].isin(ROAD_USER_CLASSES)) if len(o) else 0
    f["n_vru_near_path"] = count(o["is_vru"] & o["near_path"]) if len(o) else 0
    f["n_traffic_light"] = count(o["cls_name"] == "traffic light") if len(o) else 0
    age_cap = cfg.motion_lag_sec
    if len(road):
        road = road.assign(age_c=np.minimum(road["age_sec"], age_cap))
        lead = _pick(road, road["in_path"], "bottom_y_n",
                     {"is_vru": "lead_is_vru", "h_n": "lead_h_n", "bottom_y_n": "lead_bottom_y", "age_c": "lead_age_sec",
                      "scale_rate": "lead_scale_rate", "ttc_sec": "lead_ttc_sec", "bottom_rate": "lead_bottom_rate",
                      "attention": "lead_attention"}, keys)
        vru = _pick(road, road["is_vru"], "bottom_y_n",
                    {"h_n": "vru_h_n", "bottom_y_n": "vru_bottom_y", "x_offset_n": "vru_abs_offset",
                     "near_path": "vru_near_path", "scale_rate": "vru_scale_rate", "ttc_sec": "vru_ttc_sec",
                     "lateral_closing": "vru_lateral_closing", "attention": "vru_attention"}, keys)
        vru["vru_abs_offset"] = vru["vru_abs_offset"].astype(float).abs()
        grp = road.groupby("t_ms")
        near = road[road["near_path"]]
        ng = near.groupby("t_ms")
        ip = road[road["in_path"]].groupby("t_ms")
        agg = pd.DataFrame(index=keys)
        agg["vru_max_conf"] = road[road["is_vru"]].groupby("t_ms")["conf"].max().reindex(keys)
        agg["max_threat"] = grp["threat"].max().reindex(keys)
        agg["min_ttc_inpath"] = ip["ttc_sec"].min().reindex(keys)
        agg["min_ttc_near"] = ng["ttc_sec"].min().reindex(keys)
        agg["max_scale_rate_near"] = ng["scale_rate"].max().reindex(keys)
        agg["n_new_near_1s"] = (near["age_sec"] < age_cap).groupby(near["t_ms"]).sum().reindex(keys, fill_value=0)
        agg["gaze_on_road_max"] = grp["attention"].max().reindex(keys)
        un = road["threat"] * (1.0 - road["attention"])
        agg["unattended_threat"] = un.groupby(road["t_ms"]).max().reindex(keys)
        att = (road["threat"] * road["attention"]).groupby(road["t_ms"]).sum().reindex(keys)
        tot = grp["threat"].sum().reindex(keys)
        agg["attended_threat_share"] = att / tot.where(tot > 0)
        f = f.join(lead).join(vru).join(agg)
    for c in OBJECT_FEATURES + MOTION_FEATURES + GAZE_FEATURES:
        if c not in f.columns:
            f[c] = np.nan
    f["lead_present"] = f["lead_h_n"].notna().astype(int)
    f["vru_present"] = f["vru_h_n"].notna().astype(int)
    for c in ("lead_is_vru", "vru_near_path"):
        f[c] = f[c].astype(float)
    # absent objects: neutral values (no object = no threat, contact time at the cap)
    f["max_threat"] = f["max_threat"].fillna(0.0)
    f["unattended_threat"] = f["unattended_threat"].fillna(0.0)
    f["n_new_near_1s"] = f["n_new_near_1s"].fillna(0).astype(int)
    f = f.reset_index()

    # gaze descriptors of the row itself + causal shifts
    r = rows.copy()
    empty = r["gaze_empty"].astype(bool) | ~r["gaze_ok"].astype(bool)
    for c in ("gaze_spread", "gaze_peak_x", "gaze_peak_y", "gaze_active_frac", "gaze_n_regions"):
        f[c] = np.where(empty, np.nan, r[c].to_numpy(float))
    cx = np.where(empty, np.nan, r["gaze_centroid_x"].to_numpy(float))
    cy = np.where(empty, np.nan, r["gaze_centroid_y"].to_numpy(float))
    f["gaze_centroid_dist"] = np.hypot((cx - 0.5) * cfg.aspect, cy - 0.5)
    t = r["t_ms"].to_numpy()
    pos = {v: i for i, v in enumerate(t)}
    for name, sec in (("gaze_shift_s", cfg.gaze_shift_short_sec), ("gaze_shift_l", cfg.gaze_shift_long_sec)):
        lag = int(round(sec * 1000))
        j = np.array([pos.get(v - lag, -1) for v in t])
        ok = j >= 0
        out = np.full(len(t), np.nan)
        out[ok] = np.hypot((cx[ok] - cx[j[ok]]) * cfg.aspect, cy[ok] - cy[j[ok]])
        f[name] = out

    # GPS (causal) and keys
    kin = causal_kinematics(gps_records(r), r["t_sec"].to_numpy(float), cfg)
    for c in KINEMATIC_FEATURES + STORED_ONLY:
        f[c] = kin[c].to_numpy()
    for c in KEY + ["t_sec"]:
        f[c] = r[c].to_numpy()
    f["video_id"] = f["video_id"].astype(str)
    f["warmup"] = r["t_sec"].to_numpy(float) < cfg.motion_lag_sec + cfg.motion_lag_tol_sec - 1e-9
    return f


def fuse_sequence(split: str, video_id: str, rows: pd.DataFrame, tracks: pd.DataFrame, cfg: FusionConfig):
    """Returns (frame_df, object_df, event_dict) for one sequence."""
    rows = rows.sort_values("t_ms").reset_index(drop=True)
    obj = object_features(tracks, rows, cfg)
    fr = frame_features(rows, obj, cfg)
    ev = sequence_event(split, video_id, rows, cfg)
    lab = label_rows(fr["t_sec"].to_numpy(float), ev, cfg)
    for c in lab.columns:
        fr[c] = lab[c].to_numpy()
    fr["in_scope"] = fr["in_scope"].astype(bool) & ~fr["warmup"].astype(bool)
    return fr[FRAME_COLUMNS], obj, ev
