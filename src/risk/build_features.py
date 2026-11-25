"""
Step 3 - Build one feature row per sampled frame, plus its label.

Inputs: aligned_records.csv (frames), data/risk/detections.csv, data/risk/braking_events.csv,
        gaze maps (data/processed/.../frames/gaze), GPS JSONs (data/BDDA/.../gps_jsons).

Every feature uses information available at time t or before (no look-ahead):
  objects     counts per class, nearest/largest VRU and vehicle, objects in the ego-path corridor
  motion      growth of the largest in-path object / VRU over the last 0.2 s and 1 s (approach proxy)
  attention   share of the gaze map on VRUs, vehicles, in-path objects and the path corridor;
              gaze spread (entropy), peak position, centroid shift
  kinematics  last GPS speed, past acceleration, 2 s speed trend (GPS records with t_gps <= t only;
              aligned_records' nearest-GPS columns are NOT used because the nearest record can be in the future)

Label (from braking_events.csv, onset = GPS braking onset):
  y = 1 if 0 <= onset - t <= HORIZON_SEC, else 0; frames after the onset are out of scope (in_scope = False).
  Clips without a detected event follow NO_EVENT_POLICY ("exclude" -> out of scope, "negative" -> y = 0).
time_seconds and lead_sec are kept for evaluation only and must never be used as model inputs.

Output: data/risk/frame_features.csv
Run from the repository root:
    python src/risk/build_features.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import risk_config as C
from gps_utils import causal_kinematics, load_gps
from records import load_records

GW, GH = C.IMG_W // C.GAZE_READ_SCALE, C.IMG_H // C.GAZE_READ_SCALE
SHORT_N = max(1, int(round(C.DELTA_SHORT_SEC * C.SAMPLING_FPS)))
LONG_N = max(1, int(round(C.DELTA_LONG_SEC * C.SAMPLING_FPS)))
CLASS_COL = {"pedestrian": "n_pedestrian", "rider": "n_rider", "car": "n_car", "truck": "n_truck", "bus": "n_bus",
             "motorcycle": "n_motorcycle", "bicycle": "n_bicycle", "traffic light": "n_traffic_light",
             "traffic sign": "n_traffic_sign"}

# feature groups (used by train_risk_model.py for ablations)
OBJECT_FEATURES = list(CLASS_COL.values()) + [
    "n_vru", "n_vehicle", "vru_max_area", "vru_max_ybottom", "vru_min_dx", "vru_inpath_n", "vru_inpath_max_area",
    "vru_max_conf", "veh_max_area", "veh_inpath_n", "veh_inpath_max_area", "veh_inpath_max_ybottom", "lead_area"]
MOTION_FEATURES = [f"{b}_{k}" for b in ("lead_area", "vru_max_area", "veh_inpath_max_area") for k in ("growth_s", "growth_l")] + [
    "n_vru_delta_l"]
GAZE_FEATURES = ["gaze_on_vru", "gaze_on_vehicle", "gaze_on_inpath", "gaze_on_path_zone", "gaze_entropy",
                 "gaze_peak_x", "gaze_peak_y", "gaze_spread", "gaze_centroid_dist", "gaze_shift_s", "gaze_shift_l"]
KINEMATIC_FEATURES = ["speed_kmh", "accel_past", "speed_trend", "gps_valid"]
ALL_FEATURES = OBJECT_FEATURES + MOTION_FEATURES + GAZE_FEATURES + KINEMATIC_FEATURES


def path_corridor_mask() -> np.ndarray:
    ys = (np.arange(GH) + 0.5) / GH
    xs = (np.arange(GW) + 0.5) / GW
    half = C.PATH_HALF_W_TOP + C.PATH_HALF_W_GROWTH * np.clip((ys - C.PATH_Y_MIN) / (1 - C.PATH_Y_MIN), 0, 1)
    m = (np.abs(xs[None, :] - 0.5) <= half[:, None]) & (ys[:, None] >= C.PATH_Y_MIN)
    return m


CORRIDOR = path_corridor_mask()


def in_path(cx: np.ndarray, yb: np.ndarray) -> np.ndarray:
    half = C.PATH_HALF_W_TOP + C.PATH_HALF_W_GROWTH * np.clip((yb - C.PATH_Y_MIN) / (1 - C.PATH_Y_MIN), 0, 1)
    return (yb >= C.PATH_Y_MIN) & (np.abs(cx - 0.5) <= half)


def box_mask(boxes: np.ndarray) -> np.ndarray:
    m = np.zeros((GH, GW), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        c1, c2 = int(np.floor(x1 / C.IMG_W * GW)), int(np.ceil(x2 / C.IMG_W * GW))
        r1, r2 = int(np.floor(y1 / C.IMG_H * GH)), int(np.ceil(y2 / C.IMG_H * GH))
        m[max(r1, 0):max(r2, 0), max(c1, 0):max(c2, 0)] = True
    return m


def read_gaze(path: str):
    import cv2
    if not path:
        return None
    g = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if g is None:
        return None
    if g.shape != (GH, GW):
        g = cv2.resize(g, (GW, GH), interpolation=cv2.INTER_AREA)
    return g.astype(np.float64)


def object_features(det: pd.DataFrame) -> tuple[dict, dict]:
    """det: boxes of one frame (already confidence-filtered). Returns features and box arrays for gaze."""
    f = {c: 0 for c in CLASS_COL.values()}
    for cls, n in det["cls"].value_counts().items():
        if cls in CLASS_COL:
            f[CLASS_COL[cls]] = int(n)
    x1, y1, x2, y2 = (det[k].to_numpy(float) for k in ("x1", "y1", "x2", "y2"))
    area = (x2 - x1) * (y2 - y1) / (C.IMG_W * C.IMG_H)
    cx, yb = (x1 + x2) / 2 / C.IMG_W, y2 / C.IMG_H
    ip = in_path(cx, yb)
    vru = det["cls"].isin(C.VRU_CLASSES).to_numpy()
    veh = det["cls"].isin(C.VEHICLE_CLASSES).to_numpy()
    road = vru | veh

    def mx(a, m, default=0.0):
        return float(a[m].max()) if m.any() else default

    f.update({
        "n_vru": int(vru.sum()), "n_vehicle": int(veh.sum()),
        "vru_max_area": mx(area, vru), "vru_max_ybottom": mx(yb, vru),
        "vru_min_dx": float(np.abs(cx[vru] - 0.5).min()) if vru.any() else 0.5,
        "vru_inpath_n": int((vru & ip).sum()), "vru_inpath_max_area": mx(area, vru & ip),
        "vru_max_conf": mx(det["conf"].to_numpy(float), vru),
        "veh_max_area": mx(area, veh), "veh_inpath_n": int((veh & ip).sum()),
        "veh_inpath_max_area": mx(area, veh & ip), "veh_inpath_max_ybottom": mx(yb, veh & ip),
        "lead_area": mx(area, road & ip),
    })
    xyxy = np.stack([x1, y1, x2, y2], axis=1) if len(det) else np.zeros((0, 4))
    return f, {"vru": xyxy[vru], "veh": xyxy[veh], "inpath": xyxy[road & ip]}


def gaze_features(g, boxes: dict) -> dict:
    nan = {k: np.nan for k in GAZE_FEATURES if not k.startswith("gaze_shift")}
    if g is None or g.sum() <= 0:
        return {**nan, "_cx": np.nan, "_cy": np.nan}
    p = g / g.sum()
    nz = p[p > 0]
    ys, xs = np.mgrid[0:GH, 0:GW]
    xs_n, ys_n = (xs + 0.5) / GW, (ys + 0.5) / GH
    cx, cy = float((p * xs_n).sum()), float((p * ys_n).sum())
    pk = np.unravel_index(int(np.argmax(p)), p.shape)
    return {
        "gaze_on_vru": float(p[box_mask(boxes["vru"])].sum()),
        "gaze_on_vehicle": float(p[box_mask(boxes["veh"])].sum()),
        "gaze_on_inpath": float(p[box_mask(boxes["inpath"])].sum()),
        "gaze_on_path_zone": float(p[CORRIDOR].sum()),
        "gaze_entropy": float(-(nz * np.log(nz)).sum() / np.log(p.size)),
        "gaze_peak_x": (pk[1] + 0.5) / GW, "gaze_peak_y": (pk[0] + 0.5) / GH,
        "gaze_spread": float(np.sqrt((p * ((xs_n - cx) ** 2 + (ys_n - cy) ** 2)).sum())),
        "gaze_centroid_dist": float(np.hypot(cx - 0.5, cy - 0.5)),
        "_cx": cx, "_cy": cy,
    }


def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["split", "video_id", "time_seconds"]).reset_index(drop=True)
    g = df.groupby(["split", "video_id"], sort=False)
    eps = 0.005   # 0.5% of the image: damps the jump when an object appears / disappears
    for b in ("lead_area", "vru_max_area", "veh_inpath_max_area"):
        for k, n, dt in (("growth_s", SHORT_N, C.DELTA_SHORT_SEC), ("growth_l", LONG_N, C.DELTA_LONG_SEC)):
            prev = g[b].shift(n)
            df[f"{b}_{k}"] = (np.log(df[b] + eps) - np.log(prev + eps)) / dt   # relative growth rate, 1/s
    df["n_vru_delta_l"] = df["n_vru"] - g["n_vru"].shift(LONG_N)
    for k, n in (("gaze_shift_s", SHORT_N), ("gaze_shift_l", LONG_N)):
        df[k] = np.hypot(df["_cx"] - g["_cx"].shift(n), df["_cy"] - g["_cy"].shift(n))
    return df.drop(columns=["_cx", "_cy"])


def add_labels(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    ev = events[["split", "video_id", "has_event", "onset_sec"]].copy()
    ev["video_id"] = ev["video_id"].astype(str)
    df = df.merge(ev, on=["split", "video_id"], how="left")
    df["has_event"] = df["has_event"].fillna(False).astype(bool)
    df["lead_sec"] = df["onset_sec"] - df["time_seconds"]
    df["y"] = ((df["lead_sec"] >= 0) & (df["lead_sec"] <= C.HORIZON_SEC)).astype(int)
    scope = df["has_event"] & (df["lead_sec"] >= -1e-6)
    if C.NO_EVENT_POLICY == "negative":
        scope |= ~df["has_event"]
    df["in_scope"] = scope
    return df


def main() -> None:
    if not C.DETECTIONS_CSV.exists() or not C.LABELS_SEQ_CSV.exists():
        raise SystemExit("Run braking_labels.py and detect_objects.py first.")
    recs = load_records()
    events = pd.read_csv(C.LABELS_SEQ_CSV, dtype={"video_id": str})
    det = pd.read_csv(C.DETECTIONS_CSV, dtype={"video_id": str})
    det["time_seconds"] = det["time_seconds"].astype(float).round(3)
    thr = np.where(det["cls"].isin(C.VRU_CLASSES), C.CONF_VRU, C.CONF_OTHER)
    det = det[det["conf"] >= thr]
    det_groups = {k: g for k, g in det.groupby(["split", "video_id", "time_seconds"])}
    empty = det.iloc[0:0]
    done = set()
    if C.DETECTIONS_DONE.exists():
        done = set(C.DETECTIONS_DONE.read_text().split())

    rows = []
    n_seq = recs.groupby(["split", "video_id"]).ngroups
    for i, ((split, vid), seq) in enumerate(recs.groupby(["split", "video_id"], sort=True), 1):
        if done and f"{split}/{vid}" not in done:
            continue                                   # detector has not processed this sequence
        gps_path = C.BDDA_ROOT / split / "gps_jsons" / f"{vid}.json"
        gps = load_gps(gps_path) if gps_path.exists() else pd.DataFrame(columns=["t", "speed", "course"])
        kin = causal_kinematics(gps, seq["time_seconds"].to_numpy(), C.MAX_GPS_INTERVAL)
        for (_, r), (_, k) in zip(seq.iterrows(), kin.iterrows()):
            d = det_groups.get((split, vid, r["time_seconds"]), empty)
            of, boxes = object_features(d)
            gf = gaze_features(read_gaze(r["gaze_path"]), boxes)
            rows.append({"split": split, "video_id": vid, "time_seconds": r["time_seconds"],
                         "has_gaze": bool(r["gaze_path"]), **of, **gf,
                         "speed_kmh": k["speed_mps"] * 3.6, "accel_past": k["accel_past"],
                         "speed_trend": k["speed_trend"], "gps_valid": int(bool(k["gps_valid"]))})
        if i % 50 == 0 or i == n_seq:
            print(f"  {i}/{n_seq} sequences")

    df = add_labels(add_temporal(pd.DataFrame(rows)), events)
    C.RISK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.FEATURES_CSV, index=False)
    s = df[df["in_scope"]]
    print(f"Features -> {C.FEATURES_CSV}: {len(df)} frames, {len(s)} in scope, "
          f"{int(s['y'].sum())} positive ({100 * s['y'].mean():.1f}%)")


if __name__ == "__main__":
    main()
