"""
DriveSight Phase 4: ByteTrack on the stored Phase 3 detections (no video decoding, no YOLO, no GPU).

For every sequence, reads the verified Phase 3 output
    detections/sequence_detections/{video_id}.parquet      one row per box (conf >= 0.05)
    detections/sequence_detections/{video_id}_frames.csv   one row per sampled frame (also 0 boxes)
runs one fresh BYTETracker over the frames in time order (src/tracking/tracker_core.py) and writes, under
data/processed/{split}/tracks/ (Phase 2 and 3 outputs are never modified):
    sequence_tracks/{video_id}.parquet       one row per tracked box: video_id, t_ms, det_idx (link to the
                                             Phase 3 box), track_id, class, confidence, box geometry
    sequence_tracks/{video_id}_summary.csv   one row per track
    sequence_checkpoints/{video_id}.json     status, config hash (includes the Phase 3 hash), file hashes,
                                             lineage (SHA-256 of the Phase 3 files it read), statistics
    tracks.parquet, track_summary.csv, _schema.json      merged per split

Run from the repository root:
    python -m src.tracking.run_tracking --pilot
    python -m src.tracking.run_tracking --split validation          # tuning split
    python -m src.tracking.run_tracking                              # all 1,161 sequences, resumable
Threshold options (--track-high-thresh ...) override the defaults of track_config.py; choose them on the
validation split with src/tracking/tune_tracking.py, never on test.
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

import pandas as pd

from src.features.preprocess_config import PreprocessConfig
from src.features.manifest_loader import load_manifest, select_sequences, sort_sequences
from src.features.run_preprocessing import sha256_file, atomic_write_csv, atomic_write_json
from src.detection.detect_config import DetectConfig
from src.tracking.track_config import TrackConfig, TRACK_SCHEMA_VERSION
from src.tracking.tracker_core import track_sequence, summarize_tracks, TRACK_COLUMNS, SUMMARY_COLUMNS

log = logging.getLogger("drivesight.track")

DTYPES = {"split": "string", "video_id": "string", "sample_idx": "int32", "t_ms": "int32", "det_idx": "int16",
          "track_id": "int32", "track_frame": "int16", "is_birth": "bool", "cls_id": "int8", "cls_name": "string",
          "is_vru": "bool", "conf": "float32", "x1": "float32", "y1": "float32", "x2": "float32", "y2": "float32",
          "cx_n": "float32", "cy_n": "float32", "w_n": "float32", "h_n": "float32", "area_frac": "float32",
          "bottom_y_n": "float32", "x_offset_n": "float32", "touches_border": "bool"}


def typed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=TRACK_COLUMNS)
    return df.astype({c: t for c, t in DTYPES.items() if c in df.columns})


def atomic_write_parquet(df: pd.DataFrame, path: Path, cfg: TrackConfig) -> None:
    cfg.pre.assert_safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------- Phase 3 input
def phase3_files(cfg: TrackConfig, split: str, vid: str) -> dict:
    d = cfg.det
    return {"checkpoint": d.checkpoints_dir(split) / f"{vid}.json",
            "boxes": d.seq_det_dir(split) / f"{vid}.parquet",
            "frames": d.seq_det_dir(split) / f"{vid}_frames.csv"}


def phase3_status(cfg: TrackConfig, split: str, vid: str) -> tuple[bool, dict, str]:
    """A sequence is trackable only if its Phase 3 checkpoint is 'done' and its files match their hashes."""
    p = phase3_files(cfg, split, vid)
    if not all(x.exists() for x in p.values()):
        return False, {}, "Phase 3 files missing"
    try:
        ck = json.loads(p["checkpoint"].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return False, {}, f"unreadable Phase 3 checkpoint ({e})"
    if ck.get("status") != "done":
        return False, ck, f"Phase 3 status {ck.get('status')}"
    if sha256_file(p["boxes"]) != ck.get("boxes_sha256") or sha256_file(p["frames"]) != ck.get("frames_sha256"):
        return False, ck, "Phase 3 files do not match their checkpoint hashes"
    return True, ck, ""


def phase3_hash(cfg: TrackConfig, sel: pd.DataFrame) -> str:
    """The single Phase 3 config hash of the selected sequences (all must share it)."""
    hs = set()
    for r in sel.itertuples(index=False):
        p = phase3_files(cfg, r.split, str(r.video_id))["checkpoint"]
        if p.exists():
            try:
                hs.add(json.loads(p.read_text(encoding="utf-8")).get("config_hash"))
            except (json.JSONDecodeError, OSError):
                pass
    hs.discard(None)
    if len(hs) > 1:
        raise SystemExit(f"Phase 3 outputs come from several configurations {sorted(hs)}: re-run Phase 3 first.")
    if not hs:
        raise SystemExit("No Phase 3 checkpoint found for the selected sequences. Extract drivesight_processed.zip "
                         "into data/processed (or run Phase 3) first.")
    return hs.pop()


# ---------------------------------------------------------------------------------- one sequence
def track_checkpoint_valid(cfg: TrackConfig, split: str, vid: str, h: str) -> bool:
    p = cfg.checkpoints_dir(split) / f"{vid}.json"
    t = cfg.seq_tracks_dir(split) / f"{vid}.parquet"
    s = cfg.seq_tracks_dir(split) / f"{vid}_summary.csv"
    if not (p.exists() and t.exists() and s.exists()):
        return False
    try:
        ck = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not (ck.get("status") == "done" and ck.get("config_hash") == h
            and ck.get("tracks_sha256") == sha256_file(t) and ck.get("summary_sha256") == sha256_file(s)):
        return False
    p3 = phase3_files(cfg, split, vid)                 # input changed since -> redo
    return (p3["boxes"].exists() and sha256_file(p3["boxes"]) == ck.get("phase3_boxes_sha256")
            and sha256_file(p3["frames"]) == ck.get("phase3_frames_sha256"))


def run_one(row: pd.Series, cfg: TrackConfig, h: str) -> dict:
    split, vid = str(row["split"]), str(row["video_id"])
    base = {"split": split, "video_id": vid, "config_hash": h, "started": datetime.now().isoformat(timespec="seconds")}
    ck_path = cfg.checkpoints_dir(split) / f"{vid}.json"
    ok, p3ck, why = phase3_status(cfg, split, vid)
    if not ok:
        ck = {**base, "status": "problem", "problems": [why], "finished": datetime.now().isoformat(timespec="seconds")}
        atomic_write_json(ck, ck_path, cfg.pre)
        return ck
    p3 = phase3_files(cfg, split, vid)
    t0 = time.perf_counter()
    try:
        boxes = pd.read_parquet(p3["boxes"])
        boxes["video_id"] = boxes["video_id"].astype(str)
        frames = pd.read_csv(p3["frames"], dtype={"video_id": str}, usecols=["sample_idx", "t_ms"])
        tracks, stats = track_sequence(boxes, frames, cfg)
        tracks["split"], tracks["video_id"] = split, vid
        summary = summarize_tracks(tracks)
    except Exception as e:
        ck = {**base, "status": "failed", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
              "finished": datetime.now().isoformat(timespec="seconds")}
        atomic_write_json(ck, ck_path, cfg.pre)
        return ck
    problems = []
    if stats["n_frames"] != p3ck.get("n_frames", stats["n_frames"]):
        problems.append(f"{stats['n_frames']} frames read, Phase 3 checkpoint says {p3ck.get('n_frames')}")
    tpath = cfg.seq_tracks_dir(split) / f"{vid}.parquet"
    spath = cfg.seq_tracks_dir(split) / f"{vid}_summary.csv"
    atomic_write_parquet(typed(tracks), tpath, cfg)
    atomic_write_csv(summary.reindex(columns=SUMMARY_COLUMNS), spath, cfg.pre)
    stats["elapsed_sec"] = round(time.perf_counter() - t0, 3)
    ck = {**base, "status": "done" if not problems else "problem", "schema_version": TRACK_SCHEMA_VERSION,
          "phase3_config_hash": p3ck.get("config_hash"),
          "phase3_boxes_sha256": sha256_file(p3["boxes"]), "phase3_frames_sha256": sha256_file(p3["frames"]),
          "tracks_sha256": sha256_file(tpath), "summary_sha256": sha256_file(spath), "problems": problems,
          "finished": datetime.now().isoformat(timespec="seconds"), **stats}
    atomic_write_json(ck, ck_path, cfg.pre)
    return ck


def merge_split(cfg: TrackConfig, manifest: pd.DataFrame, split: str, h: str, env: dict) -> dict:
    seqs = sort_sequences(manifest[manifest["split"] == split])["video_id"].tolist()
    ok = [v for v in seqs if track_checkpoint_valid(cfg, split, v, h)]
    if not ok:
        return {"split": split, "merged": 0, "manifest": len(seqs)}
    tracks = pd.concat([pd.read_parquet(cfg.seq_tracks_dir(split) / f"{v}.parquet") for v in ok], ignore_index=True)
    summ = pd.concat([pd.read_csv(cfg.seq_tracks_dir(split) / f"{v}_summary.csv", dtype={"video_id": str})
                      for v in ok], ignore_index=True)
    assert not tracks.duplicated(["video_id", "track_id", "t_ms"]).any(), "duplicate (track, frame)"
    assert not tracks.duplicated(["video_id", "t_ms", "det_idx"]).any(), "box in two tracks"
    atomic_write_parquet(typed(tracks), cfg.tracks_path(split), cfg)
    atomic_write_csv(summ, cfg.summary_path(split), cfg.pre)
    atomic_write_json({"schema_version": TRACK_SCHEMA_VERSION, "config_hash": h, "env": env,
                       "params": cfg.params(), "merged_sequences": len(ok), "manifest_sequences": len(seqs),
                       "missing": [v for v in seqs if v not in ok], "tracked_boxes": len(tracks),
                       "tracks": len(summ), "merged_at": datetime.now().isoformat(timespec="seconds")},
                      cfg.tracks_dir(split) / "_schema.json", cfg.pre)
    return {"split": split, "merged": len(ok), "manifest": len(seqs), "tracked_boxes": len(tracks), "tracks": len(summ)}


def environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import ultralytics
        env["ultralytics"] = ultralytics.__version__
    except ImportError:
        pass
    import importlib.util
    env["assignment"] = "lap" if importlib.util.find_spec("lap") else "scipy (lap not installed)"
    return env


# ---------------------------------------------------------------------------------- CLI
def add_tracker_args(ap: argparse.ArgumentParser) -> None:
    d = TrackConfig()
    ap.add_argument("--track-high-thresh", type=float, default=d.track_high_thresh)
    ap.add_argument("--track-low-thresh", type=float, default=d.track_low_thresh)
    ap.add_argument("--new-track-thresh", type=float, default=d.new_track_thresh)
    ap.add_argument("--match-thresh", type=float, default=d.match_thresh)
    ap.add_argument("--track-buffer-sec", type=float, default=d.track_buffer_sec)
    ap.add_argument("--no-fuse-score", action="store_true")
    ap.add_argument("--drop-birth-box", action="store_true", help="keep only boxes returned as confirmed")
    ap.add_argument("--processed-root", type=Path)
    ap.add_argument("--bdda-root", type=Path)
    ap.add_argument("--manifest", type=Path)


def build_config(a) -> TrackConfig:
    pkw = {k: v for k, v in (("processed_root", a.processed_root), ("bdda_root", a.bdda_root),
                             ("manifest_csv", a.manifest)) if v}
    pre = PreprocessConfig(**pkw)
    return TrackConfig(track_high_thresh=a.track_high_thresh, track_low_thresh=a.track_low_thresh,
                       new_track_thresh=a.new_track_thresh, match_thresh=a.match_thresh,
                       track_buffer_sec=a.track_buffer_sec, fuse_score=not a.no_fuse_score,
                       keep_birth_box=not a.drop_birth_box, det=DetectConfig(pre=pre))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--split", choices=["training", "validation", "test"])
    ap.add_argument("--videos", nargs="+")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    add_tracker_args(ap)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    cfg = build_config(a)
    pre = cfg.pre
    manifest = load_manifest(pre)
    sel = select_sequences(manifest, split=a.split, video_ids=a.videos, pilot=a.pilot, limit=a.limit)
    env = environment()
    h = cfg.output_hash(phase3_hash=phase3_hash(cfg, sel), ultralytics_version=env.get("ultralytics", "?"))

    log_dir = pre.processed_root / "logs"
    pre.assert_safe_output(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"track_{datetime.now():%Y%m%d_%H%M%S}.log"
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for hd in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        hd.setFormatter(fmt)
        log.addHandler(hd)
    log.info("DriveSight Phase 4 | ByteTrack %s | config %s | %d sequences %s | env %s",
             cfg.params(), h, len(sel), sel["split"].value_counts().to_dict(), env)

    counts = {"done": 0, "problem": 0, "failed": 0, "skipped": 0}
    t_start = time.perf_counter()
    try:
        for i, r in enumerate(sel.itertuples(index=False), 1):
            r = pd.Series(r._asdict())
            if not a.force and track_checkpoint_valid(cfg, r["split"], str(r["video_id"]), h):
                counts["skipped"] += 1
                continue
            ck = run_one(r, cfg, h)
            counts[ck["status"]] += 1
            msg = f"[{i}/{len(sel)}] {ck['split']}/{ck['video_id']}: {ck['status']}"
            if "n_tracks" in ck:
                msg += (f" frames={ck['n_frames']} boxes in={ck['n_boxes_in']} tracked={ck['n_boxes_tracked']} "
                        f"tracks={ck['n_tracks']} (VRU {ck['n_tracks_vru']}) unconfirmed={ck['n_unconfirmed_dropped']} "
                        f"{ck['elapsed_sec']}s")
            if ck.get("problems"):
                msg += " | PROBLEMS: " + "; ".join(ck["problems"])
            if ck.get("error"):
                msg += " | ERROR: " + ck["error"]
            log.info(msg)
    except KeyboardInterrupt:
        log.warning("Interrupted. Rerun the same command to resume.")
        return 130
    total = time.perf_counter() - t_start
    log.info("finished: %s in %.1f min", counts, total / 60)
    if not a.no_merge:
        for s in dict.fromkeys(sel["split"]):
            log.info("merge %s: %s", s, merge_split(cfg, manifest, s, h, env))
    atomic_write_json({"config_hash": h, "params": cfg.params(), "counts": counts, "env": env,
                       "minutes": round(total / 60, 2)}, log_path.with_name(log_path.stem + "_summary.json"), pre)
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
