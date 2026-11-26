"""
DriveSight Phase 3: YOLO detection on the Phase 2 sampling grid (no frame storage).

For every sequence, the frames to analyse are read from the verified Phase 2 output
(metadata/sequence_records/{video_id}.csv: sample_idx, t_ms, cam_frame_idx). The camera video
is decoded sequentially with the same SequentialReader, so every detection joins exactly to a
Phase 2 row by (video_id, t_ms). The two phases share one definition of the timestamp-to-frame
mapping.

Outputs per split (next to the Phase 2 outputs):
  detections/sequence_detections/{video_id}.parquet   one row per box (conf >= 0.05)
  detections/sequence_detections/{video_id}_frames.csv one row per sampled frame (also 0 boxes)
  detections/sequence_checkpoints/{video_id}.json     status, config hash, file hashes, timings
  detections/detections.parquet, detections/frame_summary.csv   merged
  qc/detections/{video_id}_t{ms}.jpg                  boxes drawn on the middle frame (~5 %)

Run from the repository root:
    python -m src.detection.run_detection --pilot
    python -m src.detection.run_detection                       # all 1,161 sequences, resumable
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.features.preprocess_config import PreprocessConfig
from src.features.manifest_loader import load_manifest, select_sequences, sequence_paths, sort_sequences
from src.features.frame_extractor import SequentialReader
from src.features.run_preprocessing import (sha256_file, atomic_write_csv, atomic_write_json,
                                            qc_selected, checkpoint_is_valid as phase2_ok)
from src.detection.detect_config import DetectConfig, CLASS_NAMES, DETECTION_SCHEMA_VERSION
from src.detection.box_features import box_table, BOX_COLUMNS

log = logging.getLogger("drivesight.detect")

KEY_COLUMNS = ["split", "video_id", "sample_idx", "t_ms"]
DET_COLUMNS = KEY_COLUMNS + BOX_COLUMNS
FRAME_COLUMNS = KEY_COLUMNS + ["cam_frame_idx", "n_det", "n_vru", "max_conf", "max_conf_vru"]
DTYPES = {"split": "string", "video_id": "string", "sample_idx": "int32", "t_ms": "int32",
          "det_idx": "int16", "cls_id": "int8", "cls_name": "string", "is_vru": "bool",
          "conf": "float32", "x1": "float32", "y1": "float32", "x2": "float32", "y2": "float32",
          "cx_n": "float32", "cy_n": "float32", "w_n": "float32", "h_n": "float32",
          "area_frac": "float32", "aspect_wh": "float32", "bottom_y_n": "float32",
          "x_offset_n": "float32", "touches_border": "bool"}
COLORS = {"pedestrian": (0, 0, 255), "rider": (0, 80, 255), "motorcycle": (0, 160, 255),
          "bicycle": (0, 220, 255), "car": (255, 160, 0), "truck": (255, 80, 0), "bus": (200, 0, 120),
          "traffic light": (0, 200, 0), "traffic sign": (160, 160, 160)}


# ----------------------------------------------------------------------------------
def atomic_write_parquet(df: pd.DataFrame, path: Path, cfg: DetectConfig) -> None:
    cfg.pre.assert_safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def typed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=DET_COLUMNS)
    return df.astype({c: t for c, t in DTYPES.items() if c in df.columns})


def draw_qc(frame: np.ndarray, boxes: pd.DataFrame, label: str, min_conf: float = 0.25) -> np.ndarray:
    img = frame.copy()
    for b in boxes[boxes["conf"] >= min_conf].itertuples():
        c = COLORS.get(b.cls_name, (255, 255, 255))
        p1, p2 = (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2))
        cv2.rectangle(img, p1, p2, c, 2)
        cv2.putText(img, f"{b.cls_name} {b.conf:.2f}", (p1[0], max(p1[1] - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, label + f"  (boxes with conf >= {min_conf})", (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------------
def detect_sequence(row: pd.Series, cfg: DetectConfig, detector) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    split, vid = str(row["split"]), str(row["video_id"])
    t0 = time.perf_counter()
    infer0 = getattr(detector, "infer_seconds", 0.0)
    pre = cfg.pre
    grid = pd.read_csv(pre.sequence_records_dir(split) / f"{vid}.csv",
                       usecols=["sample_idx", "t_ms", "cam_frame_idx", "cam_ok"])
    W, H = pre.expected_camera_w, pre.expected_camera_h
    want_qc = qc_selected(vid, cfg.qc_fraction)
    qc_sample = int(grid["sample_idx"].iloc[len(grid) // 2]) if len(grid) else -1
    problems, boxes_parts, frames_rows = [], [], []

    def flush(batch):
        if not batch:
            return
        preds = detector.predict([f for _, f in batch])
        for (g, f), dets in zip(batch, preds):
            bt = box_table(dets, W, H, cfg.border_px)
            for k, v in (("split", split), ("video_id", vid), ("sample_idx", int(g.sample_idx)),
                         ("t_ms", int(g.t_ms))):
                bt.insert(list(KEY_COLUMNS).index(k), k, v)
            boxes_parts.append(bt)
            vru = bt[bt["is_vru"].astype(bool)] if len(bt) else bt
            frames_rows.append({"split": split, "video_id": vid, "sample_idx": int(g.sample_idx),
                                "t_ms": int(g.t_ms), "cam_frame_idx": int(g.cam_frame_idx),
                                "n_det": len(bt), "n_vru": len(vru),
                                "max_conf": float(bt["conf"].max()) if len(bt) else np.nan,
                                "max_conf_vru": float(vru["conf"].max()) if len(vru) else np.nan})
            if want_qc and int(g.sample_idx) == qc_sample:
                nonlocal_qc[0] = (f.copy(), bt, int(g.t_ms))
        batch.clear()

    nonlocal_qc = [None]
    batch = []
    with SequentialReader(sequence_paths(split, vid, pre)["camera"]) as reader:
        if not reader.opened:
            problems.append("could not open camera video")
        elif reader.size != (W, H):
            problems.append(f"camera size {reader.size}")
        else:
            for g in grid.itertuples():
                frame = reader.read_at(int(g.cam_frame_idx))
                if frame is None:
                    problems.append(f"decode failed at frame {g.cam_frame_idx}")
                    break
                batch.append((g, frame))
                if len(batch) >= cfg.batch:
                    flush(batch)
            flush(batch)
    qc = nonlocal_qc[0]

    boxes = typed(pd.concat(boxes_parts, ignore_index=True)) if boxes_parts else typed(pd.DataFrame())
    frames = pd.DataFrame(frames_rows, columns=FRAME_COLUMNS)
    if len(frames) != len(grid):
        problems.append(f"{len(frames)} frames analysed, {len(grid)} expected")
    n_by_cls = boxes["cls_name"].value_counts().to_dict() if len(boxes) else {}
    stats = {
        "n_frames": len(frames), "n_det": len(boxes),
        "n_det_by_class": {c: int(n_by_cls.get(c, 0)) for c in CLASS_NAMES},
        "n_det_conf25": int((boxes["conf"] >= 0.25).sum()) if len(boxes) else 0,
        "n_vru": int(boxes["is_vru"].sum()) if len(boxes) else 0,
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "infer_sec": round(getattr(detector, "infer_seconds", 0.0) - infer0, 3),
        "problems": problems,
    }
    stats["_qc"] = qc
    return boxes, frames, stats


def det_checkpoint_valid(cfg: DetectConfig, split: str, vid: str, h: str) -> bool:
    p = cfg.checkpoints_dir(split) / f"{vid}.json"
    b = cfg.seq_det_dir(split) / f"{vid}.parquet"
    f = cfg.seq_det_dir(split) / f"{vid}_frames.csv"
    if not (p.exists() and b.exists() and f.exists()):
        return False
    try:
        ck = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (ck.get("status") == "done" and ck.get("config_hash") == h
            and ck.get("boxes_sha256") == sha256_file(b) and ck.get("frames_sha256") == sha256_file(f))


def run_one(row, cfg: DetectConfig, detector, h: str) -> dict:
    split, vid = str(row["split"]), str(row["video_id"])
    started = datetime.now().isoformat(timespec="seconds")
    ck_path = cfg.checkpoints_dir(split) / f"{vid}.json"
    base = {"split": split, "video_id": vid, "config_hash": h, "started": started}
    p2_csv = cfg.pre.sequence_records_dir(split) / f"{vid}.csv"
    if not phase2_ok(cfg.pre, split, vid, cfg.pre.output_hash()):
        ck = {**base, "status": "problem", "problems": ["no valid Phase 2 checkpoint for this sequence"],
              "finished": datetime.now().isoformat(timespec="seconds")}
        atomic_write_json(ck, ck_path, cfg.pre)
        return ck
    try:
        boxes, frames, stats = detect_sequence(row, cfg, detector)
    except Exception as e:
        ck = {**base, "status": "failed", "error": f"{type(e).__name__}: {e}",
              "traceback": traceback.format_exc(), "finished": datetime.now().isoformat(timespec="seconds")}
        atomic_write_json(ck, ck_path, cfg.pre)
        return ck
    qc = stats.pop("_qc")
    qc_rel = None
    if qc is not None:
        frame, bt, t_ms = qc
        img = draw_qc(frame, bt, f"{split}/{vid} t={t_ms/1000:.1f}s {cfg.model_key}")
        qp = cfg.qc_dir(split) / f"{vid}_t{t_ms:05d}.jpg"
        cfg.pre.assert_safe_output(qp)
        qp.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            tmp = qp.with_name(qp.name + ".tmp")
            tmp.write_bytes(buf.tobytes())
            os.replace(tmp, qp)
            qc_rel = str(qp.relative_to(cfg.pre.processed_root))
    bpath = cfg.seq_det_dir(split) / f"{vid}.parquet"
    fpath = cfg.seq_det_dir(split) / f"{vid}_frames.csv"
    atomic_write_parquet(boxes, bpath, cfg)
    atomic_write_csv(frames, fpath, cfg.pre)
    ck = {**base, "status": "done" if not stats["problems"] else "problem",
          "schema_version": DETECTION_SCHEMA_VERSION,
          "phase2_csv_sha256": sha256_file(p2_csv),
          "boxes_sha256": sha256_file(bpath), "frames_sha256": sha256_file(fpath),
          "qc_image": qc_rel, "finished": datetime.now().isoformat(timespec="seconds"), **stats}
    atomic_write_json(ck, ck_path, cfg.pre)
    return ck


def merge_split(cfg: DetectConfig, manifest: pd.DataFrame, split: str, h: str, env: dict) -> dict:
    seqs = sort_sequences(manifest[manifest["split"] == split])["video_id"].tolist()
    ok = [v for v in seqs if det_checkpoint_valid(cfg, split, v, h)]
    if not ok:
        return {"split": split, "merged": 0, "manifest": len(seqs)}
    boxes = pd.concat([pd.read_parquet(cfg.seq_det_dir(split) / f"{v}.parquet") for v in ok], ignore_index=True)
    frames = pd.concat([pd.read_csv(cfg.seq_det_dir(split) / f"{v}_frames.csv", dtype={"video_id": str})
                        for v in ok], ignore_index=True)
    assert not frames.duplicated(["video_id", "t_ms"]).any(), "duplicate frame keys"
    assert not boxes.duplicated(["video_id", "t_ms", "det_idx"]).any(), "duplicate box keys"
    atomic_write_parquet(typed(boxes), cfg.detections_path(split), cfg)
    atomic_write_csv(frames, cfg.frame_summary_path(split), cfg.pre)
    atomic_write_json({"schema_version": DETECTION_SCHEMA_VERSION, "config_hash": h, "env": env,
                       "config": cfg.describe(), "merged_sequences": len(ok), "manifest_sequences": len(seqs),
                       "missing": [v for v in seqs if v not in ok], "frames": len(frames), "boxes": len(boxes),
                       "merged_at": datetime.now().isoformat(timespec="seconds")},
                      cfg.det_dir(split) / "_schema.json", cfg.pre)
    return {"split": split, "merged": len(ok), "manifest": len(seqs), "frames": len(frames), "boxes": len(boxes)}


def check_weights(cfg: DetectConfig) -> str:
    sha = sha256_file(cfg.weights_path)
    ref = cfg.weights_dir / "SHA256.txt"
    if ref.exists():
        refs = {l.split()[-1].lstrip("*"): l.split()[0].lower() for l in ref.read_text().splitlines() if l.strip()}
        exp = refs.get(cfg.weights_path.name)
        if exp and exp != sha:
            raise SystemExit(f"{cfg.weights_path.name}: SHA-256 does not match SHA256.txt. Refusing to run.")
    return sha


def environment(detector) -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "device": getattr(detector, "device_type", "?"), "gpu": getattr(detector, "gpu_name", "?"),
           "half": getattr(detector, "half", None)}
    try:
        import torch, ultralytics
        env.update(torch=torch.__version__, cuda=torch.version.cuda, ultralytics=ultralytics.__version__)
    except ImportError:
        pass
    return env


# ----------------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--split", choices=["training", "validation", "test"])
    ap.add_argument("--videos", nargs="+")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--model", default="yolov8s", choices=["yolov8s", "yolo11s", "yolo11n"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--half", default="auto", choices=["auto", "true", "false"])
    ap.add_argument("--qc-fraction", type=float)
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--processed-root", type=Path)
    ap.add_argument("--bdda-root", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--weights-dir", type=Path)
    return ap.parse_args(argv)


def main(argv=None, detector=None) -> int:
    a = parse_args(argv)
    pkw = {k: v for k, v in (("processed_root", a.processed_root), ("bdda_root", a.bdda_root),
                             ("manifest_csv", a.manifest)) if v}
    pre = PreprocessConfig(**pkw)
    dkw = {"model_key": a.model, "batch": a.batch, "device": a.device, "half": a.half, "pre": pre}
    if a.qc_fraction is not None:
        dkw["qc_fraction"] = a.qc_fraction
    if a.weights_dir:
        dkw["weights_dir"] = a.weights_dir
    cfg = DetectConfig(**dkw)
    manifest = load_manifest(pre)
    sel = select_sequences(manifest, split=a.split, video_ids=a.videos, pilot=a.pilot, limit=a.limit)

    if detector is None:
        weights_sha = check_weights(cfg)
        from src.detection.detector import YoloDetector
        detector = YoloDetector(cfg)
        import ultralytics
        ul_version = ultralytics.__version__
    else:                                              # tests inject a fake detector
        weights_sha = getattr(detector, "weights_sha", "fake")
        ul_version = getattr(detector, "version", "fake")
    h = cfg.output_hash(weights_sha256=weights_sha, device_type=getattr(detector, "device_type", "cpu"),
                        half=bool(getattr(detector, "half", False)), ultralytics_version=ul_version)
    env = environment(detector)

    log_dir = pre.processed_root / "logs"
    pre.assert_safe_output(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"detect_{datetime.now():%Y%m%d_%H%M%S}.log"
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for hd in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        hd.setFormatter(fmt)
        log.addHandler(hd)
    log.info("DriveSight Phase 3 | %s | imgsz %d conf>=%.2f iou %.2f | config %s | %d sequences %s | env %s",
             cfg.model_key, cfg.imgsz, cfg.conf_floor, cfg.nms_iou, h, len(sel),
             sel["split"].value_counts().to_dict(), env)

    if hasattr(detector, "warmup") and len(sel):
        r0 = sel.iloc[0]
        with SequentialReader(sequence_paths(r0["split"], r0["video_id"], pre)["camera"]) as rd:
            f = rd.read_at(0)
        if f is not None:
            detector.warmup(f)

    counts = {"done": 0, "problem": 0, "failed": 0, "skipped": 0}
    t_start = time.perf_counter()
    try:
        for i, r in enumerate(sel.itertuples(index=False), 1):
            r = pd.Series(r._asdict())
            if not a.force and det_checkpoint_valid(cfg, r["split"], r["video_id"], h):
                counts["skipped"] += 1
                continue
            ck = run_one(r, cfg, detector, h)
            counts[ck["status"]] += 1
            msg = f"[{i}/{len(sel)}] {ck['split']}/{ck['video_id']}: {ck['status']}"
            if ck["status"] == "done":
                fps = ck["n_frames"] / ck["elapsed_sec"] if ck["elapsed_sec"] else 0
                msg += (f" frames={ck['n_frames']} boxes={ck['n_det']} (conf>=.25: {ck['n_det_conf25']}, "
                        f"VRU {ck['n_vru']}) {ck['elapsed_sec']}s ({fps:.1f} frames/s, infer {ck['infer_sec']}s)")
            if ck.get("problems"):
                msg += " | PROBLEMS: " + "; ".join(ck["problems"])
            if ck.get("error"):
                msg += " | ERROR: " + ck["error"]
            log.info(msg)
            if i % 25 == 0:
                n = counts["done"] + counts["problem"] + counts["failed"]
                rate = (time.perf_counter() - t_start) / max(n, 1)
                log.info("  ~%.2f s/sequence, ETA %.1f min", rate, rate * (len(sel) - i) / 60)
    except KeyboardInterrupt:
        log.warning("Interrupted. Rerun the same command to resume.")
        return 130
    total = time.perf_counter() - t_start
    log.info("finished: %s in %.1f min", counts, total / 60)
    if not a.no_merge:
        for s in dict.fromkeys(sel["split"]):
            log.info("merge %s: %s", s, merge_split(cfg, manifest, s, h, env))
    atomic_write_json({"config_hash": h, "counts": counts, "env": env, "minutes": round(total / 60, 2),
                       "infer_seconds": round(getattr(detector, "infer_seconds", 0.0), 1),
                       "frames": getattr(detector, "frames_seen", None)},
                      log_path.with_name(log_path.stem + "_summary.json"), pre)
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
