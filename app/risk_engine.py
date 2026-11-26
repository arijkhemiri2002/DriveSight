"""DriveSight demo app - video risk engine.

Phase 5 (preferred): if data/risk/risk_model_vision_v5.joblib exists (python -m src.fusion.train_risk),
the video goes through the same chain as BDDA - detection, ByteTrack with the frozen Phase 4 setting,
Phase 5 object + motion features (src/fusion/live.py) - and is scored by that model.
Otherwise (older pipeline, kept as a fallback):

Turns a driving video into a per-frame risk timeline, reusing the risk stage in src/risk:
  1. sample the video at 5 fps (same rate as the BDDA preprocessing),
  2. run the selected detector on each sampled frame,
  3. build the object + motion features of src/risk/build_features.py (boxes rescaled to 1280x720),
  4. score each frame:
       - learned vision model  data/risk/risk_model_vision.joblib, if train_risk_model.py has produced it
                               (objects + motion only: an arbitrary video has no attention map and no GPS),
       - otherwise the rule baseline of src/risk/rule_baseline.py, with the speed given by the user
         and no attention / GPS terms.
A research demo, not a safety system.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "risk"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import risk_config as RC  # noqa: E402
from build_features import add_temporal, object_features  # noqa: E402
from rule_baseline import rule_score  # noqa: E402

VISION_MODEL = RC.RISK_DATA_DIR / "risk_model_vision.joblib"
V5_MODEL = RC.RISK_DATA_DIR / "risk_model_vision_v5.joblib"     # Phase 5 (tracks + fusion features)
SAMPLE_FPS = RC.SAMPLING_FPS
RULE_DEFAULT_THRESHOLD = 0.35     # demo setting for the rule score (no labelled data to tune it on); the UI slider overrides it
_vision_cache: dict = {}


def _load(path: Path):
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    c = _vision_cache.setdefault(str(path), {})
    if c.get("mtime") != mtime:
        import joblib
        c.update(mtime=mtime, bundle=joblib.load(path))
    return c["bundle"]


def load_v5_model():
    """Phase 5 vision model bundle, or None."""
    return _load(V5_MODEL)


def load_vision_model():
    """Return the older learned vision model bundle, or None. Reloaded if the file changes."""
    return _load(VISION_MODEL)


def method_info() -> dict:
    v5 = load_v5_model()
    if v5 is not None:
        return {"method": "learned-v5", "label": f"Learned model ({v5.get('kind', 'model')}, tracked objects + motion)",
                "note": f"Phase 5 model trained on BDDA to predict hard braking within {v5.get('horizon_sec', 2.0)} s, "
                        f"from YOLOv8s detections tracked by ByteTrack. Validation same-time AUC "
                        f"{v5.get('val_same_time_AUC', float('nan')):.3f} (0.5 = chance). A plain video has no "
                        "attention map and no GPS, so only the vision part of the model is used.",
                "threshold": float(v5["threshold"]), "uses_speed": False}
    b = load_vision_model()
    if b is None:
        return {"method": "rule", "label": "Rule-based score",
                "note": "No learned model yet (run src/risk/train_risk_model.py). The driving speed comes from the field above; "
                        "a plain video has no attention map or GPS, so those terms are off.",
                "threshold": RULE_DEFAULT_THRESHOLD, "uses_speed": True}
    return {"method": "learned", "label": f"Learned model ({b.get('kind', 'model')}, objects + motion)",
            "note": f"Trained on BDDA to predict hard braking within {b.get('horizon_sec', RC.HORIZON_SEC)} s. "
                    f"Validation AP {b.get('val_AP', float('nan')):.3f}.",
            "threshold": float(b["threshold"]), "uses_speed": False}


def sample_times(duration: float) -> np.ndarray:
    n = int(np.floor(duration * SAMPLE_FPS)) + 1
    return np.round(np.arange(n) / SAMPLE_FPS, 3)


def read_frames(path: str, max_sec: float):
    """Yield (t, BGR frame) at SAMPLE_FPS, by timestamp. Returns metadata via the first yield."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("this file is not a readable video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or n <= 0:
        cap.release()
        raise ValueError("cannot read the video frame rate / length")
    duration = n / fps
    if duration > max_sec + 1e-6:
        cap.release()
        raise OverflowError(f"video too long ({duration:.0f} s, {max_sec:.0f} s max)")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    meta = {"fps": fps, "frames": n, "duration": duration, "width": w, "height": h}
    times = sample_times(duration)
    wanted = {min(int(round(t * fps)), n - 1): t for t in times}
    frames, idx, last = [], 0, max(wanted)
    while True:                                   # sequential read: robust for every codec (no seeking)
        ok = cap.grab()
        if not ok:
            break
        if idx in wanted:
            ok, frame = cap.retrieve()
            if ok:
                frames.append((wanted[idx], frame))
        idx += 1
        if idx > last:
            break
    cap.release()
    return meta, frames


def detect_frames(net, frames, device, imgsz, iou, min_conf, class_names, batch=8):
    """Run the detector; return {t: DataFrame(cls, conf, x1..y2 in source pixels)}."""
    out = {}
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        res = net.predict([f for _, f in chunk], imgsz=imgsz, conf=min_conf, iou=iou, device=device, verbose=False)
        for (t, _), r in zip(chunk, res):
            b = r.boxes
            if b is None or len(b) == 0:
                out[t] = pd.DataFrame(columns=["cls", "conf", "x1", "y1", "x2", "y2"])
                continue
            xyxy = b.xyxy.cpu().numpy()
            out[t] = pd.DataFrame({"cls": [class_names[int(c)] for c in b.cls.cpu().numpy()],
                                   "conf": b.conf.cpu().numpy().astype(float),
                                   "x1": xyxy[:, 0], "y1": xyxy[:, 1], "x2": xyxy[:, 2], "y2": xyxy[:, 3]})
    return out


def features_from_detections(dets: dict, width: int, height: int) -> pd.DataFrame:
    """Object + motion features per frame, using the same thresholds and code as src/risk."""
    sx, sy = RC.IMG_W / width, RC.IMG_H / height
    rows = []
    for t in sorted(dets):
        d = dets[t]
        thr = np.where(d["cls"].isin(RC.VRU_CLASSES), RC.CONF_VRU, RC.CONF_OTHER) if len(d) else []
        d = d[d["conf"] >= thr].copy() if len(d) else d
        if len(d):
            d[["x1", "x2"]] *= sx
            d[["y1", "y2"]] *= sy
        of, _ = object_features(d)
        rows.append({"split": "demo", "video_id": "upload", "time_seconds": t, **of, "_cx": np.nan, "_cy": np.nan})
    return add_temporal(pd.DataFrame(rows))


def score(df: pd.DataFrame, speed_kmh: float) -> np.ndarray:
    b = load_vision_model()
    if b is not None:
        return b["model"].predict_proba(df[b["features"]])[:, 1]
    x = df.copy()
    x["speed_kmh"] = speed_kmh
    for c in ("gaze_on_vru", "gaze_on_inpath", "accel_past"):
        x[c] = np.nan
    return rule_score(x)


def snapshot(dets: pd.DataFrame, width: int, height: int) -> dict:
    """Instant danger index for a single photo: the proximity part of the rule baseline
    (src/risk/rule_baseline.py), from the same object features. No motion and no speed, so it says how
    close road users are to the car's path, not whether braking is imminent (that needs a video)."""
    row = features_from_detections({0.0: dets}, width, height).iloc[0]
    prox_lead = float(np.clip(np.sqrt(row["lead_area"] / 0.10), 0, 1.5))
    centrality = 1 - float(np.clip(row["vru_min_dx"] / 0.5, 0, 1))
    prox_vru = float(np.clip(np.sqrt(row["vru_max_area"] / 0.02) * centrality, 0, 1.5))
    if row["vru_inpath_n"] > 0:                                   # a VRU inside the path corridor counts fully
        prox_vru = max(prox_vru, float(np.clip(np.sqrt(row["vru_inpath_max_area"] / 0.02), 0, 1.5)))
    score = float(np.clip(max(prox_lead, prox_vru) / 1.5, 0, 1))
    why = [r for r in reasons(row) if r != "closing in fast"]
    if not why and row["lead_area"] > 0.02:
        why.append("road user ahead in the lane")
    return {"score": round(score, 3), "level": "hi" if score >= 0.7 else "mid" if score >= 0.4 else "lo",
            "reasons": why, "n_vru": int(row["n_vru"]), "vru_in_path": int(row["vru_inpath_n"]),
            "lead_area_pct": round(100 * float(row["lead_area"]), 1)}


def reasons(row: pd.Series) -> list[str]:
    """Short human-readable cues behind a frame's score (for the UI; not used for scoring)."""
    r = []
    if row.get("vru_inpath_n", 0) > 0:
        r.append("vulnerable road user in the path")
    elif row.get("vru_max_area", 0) > 0.01 and row.get("vru_min_dx", 0.5) < 0.25:
        r.append("vulnerable road user close to the path")
    if row.get("lead_area", 0) > 0.06:
        r.append("road user close ahead")
    g = row.get("lead_area_growth_l", np.nan)
    if np.isfinite(g) and g > 0.35 and row.get("lead_area", 0) > 0.01:
        r.append("closing in fast")
    return r


def analyse_steps(net, video_path: str, *, device, imgsz, iou, min_conf, class_names, speed_kmh: float, max_sec: float,
                  batch: int = 8):
    """Generator: yields {"stage", "done", "total"} progress events, then {"result": ...} as the last item."""
    yield {"stage": "decode", "done": 0, "total": 1}
    meta, frames = read_frames(video_path, max_sec)
    if not frames:
        raise ValueError("no frame could be decoded from this video")
    n = len(frames)
    dets = {}
    yield {"stage": "detect", "done": 0, "total": n}
    for i in range(0, n, batch):
        dets.update(detect_frames(net, frames[i:i + batch], device, imgsz, iou, min_conf, class_names, batch=batch))
        yield {"stage": "detect", "done": min(i + batch, n), "total": n}
    yield {"stage": "score", "done": 0, "total": 1}
    v5 = load_v5_model()
    if v5 is not None:
        from src.fusion.live import reasons as v5_reasons, video_features
        feats, obj = video_features(dets, meta["width"], meta["height"])
        feats["time_seconds"] = feats["t_ms"] / 1000.0
        feats["risk"] = v5["model"].predict_proba(feats[v5["features"]])[:, 1]
        why = v5_reasons
        by_ms = {int(round(t * 1000)): t for t in dets}
        feats["time_seconds"] = [by_ms[int(m)] for m in feats["t_ms"]]
    else:
        feats = features_from_detections(dets, meta["width"], meta["height"])
        feats["risk"] = score(feats, speed_kmh)
        why = reasons
        obj = None
    # per tracked box: track id, in the ego path, contact-time proxy (Phase 5 only)
    track_info = {}
    if obj is not None and len(obj):
        for r in obj.itertuples(index=False):
            ttc = float(r.ttc_sec) if np.isfinite(r.ttc_sec) and r.ttc_sec < 9.99 else None
            track_info[(int(r.t_ms), int(r.det_idx))] = (int(r.track_id), bool(r.in_path), ttc)
    W, H = meta["width"], meta["height"]
    out_frames = []
    for _, row in feats.iterrows():
        t = row["time_seconds"]
        # same order as src/fusion/live.boxes_table (confidence, descending), so det_idx matches the tracks
        d = dets[t].sort_values("conf", ascending=False).reset_index(drop=True)
        t_ms = int(round(t * 1000))
        boxes = []
        for i, (c, s, x1, y1, x2, y2) in enumerate(zip(d["cls"], d["conf"], d["x1"], d["y1"], d["x2"], d["y2"])):
            if s < min(RC.CONF_VRU, RC.CONF_OTHER):                    # the page never draws anything below this
                continue
            b = {"name": c, "conf": round(float(s), 4),
                 "box": [round(x1 / W, 5), round(y1 / H, 5), round(x2 / W, 5), round(y2 / H, 5)]}
            ti = track_info.get((t_ms, i))
            if ti:
                b["id"], b["inpath"] = ti[0], ti[1]
                if ti[2] is not None:
                    b["ttc"] = round(ti[2], 2)
            boxes.append(b)
        fr = {"t": float(t), "risk": round(float(row["risk"]), 4), "dets": boxes,
              "n_vru": int(row["n_vru"]), "reasons": why(row)}
        if obj is not None:                                            # what the Phase 5 model sees
            ttc = row.get("min_ttc_near", np.nan)
            fr["sees"] = {"tracked": int(row.get("n_road_users", 0)), "in_path": int(row.get("n_inpath", 0)),
                          "vru_near": int(row.get("n_vru_near_path", 0)),
                          "min_ttc": round(float(ttc), 2) if np.isfinite(ttc) and ttc < 9.99 else None}
        out_frames.append(fr)
    tracking = None
    if obj is not None and len(obj):
        lens = obj.groupby("track_id")["t_ms"].count()
        tracking = {"tracks": int(lens.size), "median_len_sec": round(float(lens.median()) / SAMPLE_FPS, 1)}
    yield {"result": {**method_info(), "sample_fps": SAMPLE_FPS, "duration": meta["duration"], "width": W, "height": H,
                      "speed_kmh": speed_kmh, "frames": out_frames, "tracking": tracking}}


def analyse(net, video_path: str, **kw):
    """Run the whole analysis and return the result dict."""
    for ev in analyse_steps(net, video_path, **kw):
        if "result" in ev:
            return ev["result"]
