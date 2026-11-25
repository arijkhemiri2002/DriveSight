"""
DriveSight Phase 2 — on-the-fly multimodal preprocessing (no frame storage).

For every sequence of the clean manifest:
  1. decode camera and gaze videos sequentially (2 frames in memory at a time)
  2. sample at 5 Hz inside both videos, nearest frame per modality
  3. crop the gaze map and compute compact gaze descriptors
  4. load GPS, mark invalid values, attach nearest record + interpolated speed
  5. write  metadata/sequence_records/{video_id}.csv   (atomic)
     then   metadata/sequence_checkpoints/{video_id}.json (atomic, with CSV hash)
  6. optionally write one QC triptych (hash-sampled ~5 % + every problem sequence)
Then merge every valid per-sequence CSV of a split into metadata/aligned_records.csv.

Resume is automatic: a sequence is skipped when its checkpoint is "done", was made with
the same config hash, and its CSV still matches the recorded SHA-256. Use --force to
recompute. Raw BDDA files are opened read-only; writing under data/BDDA is refused.

Run from the repository root:
    python -m src.features.run_preprocessing --pilot          # extended pilot (11 seq)
    python -m src.features.run_preprocessing                  # all 1,161 sequences
    python -m src.features.run_preprocessing --split validation --videos 1744
    python -m src.features.run_preprocessing --dry-run        # plan + row estimate only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.features.preprocess_config import PreprocessConfig, SCHEMA_VERSION
from src.features.manifest_loader import load_manifest, select_sequences, sequence_paths, sort_sequences
from src.features.frame_extractor import SequentialReader, sample_plan
from src.features.gaze_crop import gaze_descriptors, make_qc_triptych, GAZE_DESCRIPTOR_COLUMNS
from src.features.gps_aligner import load_gps_track, align_gps, GPS_COLUMNS

log = logging.getLogger("drivesight.preprocess")

FRAME_COLUMNS = [
    "split", "video_id", "sample_idx", "t_sec", "t_ms",
    "cam_frame_idx", "cam_frame_t_sec", "cam_residual_sec", "cam_ok", "cam_mean_intensity",
    "gaze_frame_idx", "gaze_frame_t_sec", "gaze_residual_sec", "gaze_ok",
    "cam_gaze_dt_sec",
]
RECORD_COLUMNS = FRAME_COLUMNS + GAZE_DESCRIPTOR_COLUMNS + GPS_COLUMNS


# ----------------------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str, cfg: PreprocessConfig) -> None:
    cfg.assert_safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)          # atomic on Windows and POSIX


def atomic_write_csv(df: pd.DataFrame, path: Path, cfg: PreprocessConfig) -> None:
    atomic_write_text(path, df.to_csv(index=False, lineterminator="\n"), cfg)


def atomic_write_json(obj: dict, path: Path, cfg: PreprocessConfig) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=_json_default), cfg)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def qc_selected(video_id: str, fraction: float) -> bool:
    """Deterministic ~fraction sample, independent of run order."""
    h = int(hashlib.sha1(str(video_id).encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < fraction


def _r(x, nd=6):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def _early_exit(problems, warnings, t0):
    """Sequence that cannot be processed: empty records + full stats skeleton."""
    return pd.DataFrame(columns=RECORD_COLUMNS), {
        "n_rows": 0, "problems": problems, "warnings": warnings,
        "elapsed_sec": round(time.perf_counter() - t0, 3), "_qc_frames": None, "_qc_reason": None}


# ----------------------------------------------------------------------------------
# one sequence
# ----------------------------------------------------------------------------------
def process_sequence(row: pd.Series, cfg: PreprocessConfig, force_qc: bool = False):
    """Return (records DataFrame, stats dict). Never writes anything itself."""
    t0 = time.perf_counter()
    split, vid = str(row["split"]), str(row["video_id"])
    paths = sequence_paths(split, vid, cfg)
    problems: list[str] = []     # hard problems: data unusable / pipeline assumption broken
    warnings: list[str] = []     # soft issues: recorded, rows still usable with flags

    for k, p in paths.items():
        if not p.exists():
            problems.append(f"missing {k} file: {p.name}")
    if problems:
        return _early_exit(problems, warnings, t0)

    cam_fps, gaze_fps = float(row["cam_fps"]), float(row["gaze_fps"])
    cam_n, gaze_n = int(row["cam_frames"]), int(row["gaze_frames"])
    plan = sample_plan(cam_n, cam_fps, gaze_n, gaze_fps, cfg.sampling_fps)
    if not plan:
        problems.append("empty sampling plan (zero frames or fps in manifest)")

    want_qc = force_qc or qc_selected(vid, cfg.qc_fraction)
    qc_sample = plan[len(plan) // 2].sample_idx if plan else -1
    qc_frames = None

    rows = []
    cam_drift, gaze_drift = [], []
    with SequentialReader(paths["camera"]) as cr, SequentialReader(paths["gaze"]) as gr:
        if not cr.opened or not gr.opened:
            problems.append("could not open camera or gaze video")
            return _early_exit(problems, warnings, t0)
        if abs(cr.fps - cam_fps) > 0.01 or abs(gr.fps - gaze_fps) > 0.01:
            problems.append(f"fps differs from manifest (cam {cr.fps:.3f}/{cam_fps}, gaze {gr.fps:.3f}/{gaze_fps})")
        if cr.size != (cfg.expected_camera_w, cfg.expected_camera_h):
            problems.append(f"camera size {cr.size}")
        if gr.size != (cfg.expected_gaze_w, cfg.expected_gaze_h):
            problems.append(f"gaze size {gr.size}")

        for p in plan:
            cam = cr.read_at(p.cam_idx)
            cam_ct = cr.last_container_t
            gaze = gr.read_at(p.gaze_idx)
            gaze_ct = gr.last_container_t
            cam_ok, gaze_ok = cam is not None, gaze is not None
            cam_t, gaze_t = p.cam_idx / cam_fps, p.gaze_idx / gaze_fps
            rec = {
                "split": split, "video_id": vid, "sample_idx": p.sample_idx,
                "t_sec": round(p.t_sec, 6), "t_ms": p.t_ms,
                "cam_frame_idx": p.cam_idx, "cam_frame_t_sec": round(cam_t, 6),
                "cam_residual_sec": round(cam_t - p.t_sec, 6), "cam_ok": cam_ok,
                "cam_mean_intensity": round(float(cam.mean()), 2) if cam_ok else np.nan,
                "gaze_frame_idx": p.gaze_idx, "gaze_frame_t_sec": round(gaze_t, 6),
                "gaze_residual_sec": round(gaze_t - p.t_sec, 6), "gaze_ok": gaze_ok,
                "cam_gaze_dt_sec": round(cam_t - gaze_t, 6),
            }
            if gaze_ok:
                rec.update(gaze_descriptors(gaze, cfg))
            else:
                rec.update({c: np.nan for c in GAZE_DESCRIPTOR_COLUMNS})
            rows.append(rec)
            if cam_ok and p.cam_idx > 0 and np.isfinite(cam_ct):
                cam_drift.append(cam_ct - cam_t)
            if gaze_ok and p.gaze_idx > 0 and np.isfinite(gaze_ct):
                gaze_drift.append(gaze_ct - gaze_t)
            if want_qc and p.sample_idx == qc_sample and cam_ok and gaze_ok:
                qc_frames = (cam.copy(), gaze.copy(), p.t_ms)
            del cam, gaze

        cam_decoded = cr.count_total_frames()
        gaze_decoded = gr.count_total_frames()

    df = pd.DataFrame(rows)
    targets = df["t_sec"].to_numpy(dtype=float) if len(df) else np.array([])
    track = load_gps_track(paths["gps"], cfg)
    gps = align_gps(track, targets, cfg)
    df = pd.concat([df.reset_index(drop=True), gps.reset_index(drop=True)], axis=1)
    df = df.reindex(columns=RECORD_COLUMNS)

    n_cam_fail = int((~df["cam_ok"].astype(bool)).sum()) if len(df) else 0
    n_gaze_fail = int((~df["gaze_ok"].astype(bool)).sum()) if len(df) else 0
    if n_cam_fail or n_gaze_fail:
        problems.append(f"decode failures: camera {n_cam_fail}, gaze {n_gaze_fail}")
    if cam_decoded < cam_n or gaze_decoded < gaze_n:
        problems.append(f"fewer decodable frames than manifest (cam {cam_decoded}/{cam_n}, gaze {gaze_decoded}/{gaze_n})")
    if len(track) == 0:
        problems.append("no usable GPS records")

    def _maxabs(a):
        return float(np.max(np.abs(a))) if len(a) else np.nan

    cam_period, gaze_period = 1 / cam_fps, 1 / gaze_fps
    cdrift, gdrift = _maxabs(cam_drift), _maxabs(gaze_drift)
    if np.isfinite(cdrift) and cdrift > 0.5 * cam_period:
        warnings.append(f"camera container timestamps drift {cdrift:.3f}s from index/fps (possible VFR)")
    if np.isfinite(gdrift) and gdrift > 0.5 * gaze_period:
        warnings.append(f"gaze container timestamps drift {gdrift:.3f}s from index/fps (possible VFR)")
    n_unmatched = int((~df["gps_matched"].astype(bool)).sum()) if len(df) else 0
    if n_unmatched:
        warnings.append(f"{n_unmatched} rows without GPS within {cfg.gps_assoc_max_gap_sec}s")
    for key, label in (("gps_speed_invalid", "speed"), ("gps_course_invalid", "course"),
                       ("gps_position_invalid", "position")):
        if track.stats.get(key, 0):
            warnings.append(f"{track.stats[key]} GPS records with invalid {label}")
    if track.stats.get("gps_dup_timestamps", 0):
        warnings.append(f"{track.stats['gps_dup_timestamps']} duplicate GPS timestamps dropped")
    n_empty = int(df["gaze_empty"].fillna(False).astype(bool).sum()) if len(df) else 0
    if n_empty:
        warnings.append(f"{n_empty} empty gaze frames")
    bar_max = int(np.nanmax(df["gaze_bar_max"])) if len(df) and df["gaze_bar_max"].notna().any() else 0
    bar_frac = df["gaze_bar_mass_frac"] if len(df) else pd.Series(dtype=float)
    bar_frac_max = float(bar_frac.max()) if bar_frac.notna().any() else 0.0
    n_bar_rows = int((bar_frac > cfg.gaze_bar_mass_warn).sum())
    if n_bar_rows:
        warnings.append(f"{n_bar_rows} rows with >{cfg.gaze_bar_mass_warn:.0%} of gaze mass in the "
                        f"cropped-out bars (max {bar_frac_max:.1%})")

    stats = {
        "n_rows": len(df),
        "t_last_sec": _r(df["t_sec"].max()) if len(df) else None,
        "cam_fps": cam_fps, "gaze_fps": gaze_fps,
        "cam_frames_manifest": cam_n, "cam_frames_decoded": cam_decoded,
        "gaze_frames_manifest": gaze_n, "gaze_frames_decoded": gaze_decoded,
        "n_cam_fail": n_cam_fail, "n_gaze_fail": n_gaze_fail,
        "max_abs_cam_residual_sec": _r(_maxabs(df["cam_residual_sec"]) if len(df) else np.nan),
        "max_abs_gaze_residual_sec": _r(_maxabs(df["gaze_residual_sec"]) if len(df) else np.nan),
        "max_abs_cam_gaze_dt_sec": _r(_maxabs(df["cam_gaze_dt_sec"]) if len(df) else np.nan),
        "max_cam_container_drift_sec": _r(cdrift), "max_gaze_container_drift_sec": _r(gdrift),
        "n_gps_unmatched_rows": n_unmatched,
        "max_gps_gap_sec": _r(df["gps_gap_sec"].max()) if len(df) else None,
        "n_rows_speed_invalid": int((df["gps_matched"].astype(bool) & ~df["gps_speed_valid"].astype(bool)).sum()) if len(df) else 0,
        "n_gaze_empty_rows": n_empty, "gaze_bar_max": bar_max,
        "gaze_bar_mass_frac_max": round(bar_frac_max, 5),
        "gaze_bar_mass_frac_mean": round(float(bar_frac.mean()), 5) if bar_frac.notna().any() else 0.0,
        "n_rows_bar_mass_over_warn": n_bar_rows,
        **track.stats,
        "problems": problems, "warnings": warnings,
        "elapsed_sec": round(time.perf_counter() - t0, 3),
    }
    stats["_qc_frames"] = qc_frames if (want_qc or problems) else None
    stats["_qc_reason"] = "problem" if problems else ("sample" if want_qc else None)
    return df, stats


# ----------------------------------------------------------------------------------
# checkpoints
# ----------------------------------------------------------------------------------
def checkpoint_is_valid(cfg: PreprocessConfig, split: str, vid: str, config_hash: str,
                        statuses: tuple[str, ...] = ("done",)) -> bool:
    ck_path = cfg.checkpoints_dir(split) / f"{vid}.json"
    csv_path = cfg.sequence_records_dir(split) / f"{vid}.csv"
    if not ck_path.exists() or not csv_path.exists():
        return False
    try:
        ck = json.loads(ck_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (ck.get("status") in statuses and ck.get("config_hash") == config_hash
            and ck.get("split") == split and str(ck.get("video_id")) == vid
            and ck.get("csv_sha256") == sha256_file(csv_path))


def run_one(row: pd.Series, cfg: PreprocessConfig, config_hash: str) -> dict:
    split, vid = str(row["split"]), str(row["video_id"])
    started = datetime.now().isoformat(timespec="seconds")
    ck_path = cfg.checkpoints_dir(split) / f"{vid}.json"
    csv_path = cfg.sequence_records_dir(split) / f"{vid}.csv"
    try:
        df, stats = process_sequence(row, cfg)
    except Exception as e:                     # keep going; the sequence is retried next run
        ck = {"status": "failed", "split": split, "video_id": vid, "config_hash": config_hash,
              "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
              "started": started, "finished": datetime.now().isoformat(timespec="seconds")}
        atomic_write_json(ck, ck_path, cfg)
        return ck

    qc_frames, qc_reason = stats.pop("_qc_frames", None), stats.pop("_qc_reason", None)
    qc_path = None
    if qc_frames is not None:
        cam, gaze, t_ms = qc_frames
        label = f"{split}/{vid}  t={t_ms/1000:.1f}s  ({qc_reason})"
        img = make_qc_triptych(cam, gaze, cfg, label)
        qc_path = cfg.qc_overlay_dir(split) / f"{vid}_t{t_ms:05d}.jpg"
        cfg.assert_safe_output(qc_path)
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), cfg.qc_jpeg_quality])
        if ok:
            tmp = qc_path.with_name(qc_path.name + ".tmp")
            tmp.write_bytes(buf.tobytes())
            os.replace(tmp, qc_path)

    status = "done" if not stats["problems"] else "problem"
    # CSV first, checkpoint second: a checkpoint never points to a missing/partial CSV.
    atomic_write_csv(df, csv_path, cfg)
    ck = {"status": status, "split": split, "video_id": vid, "config_hash": config_hash,
          "schema_version": SCHEMA_VERSION, "csv_sha256": sha256_file(csv_path),
          "qc_overlay": str(qc_path.relative_to(cfg.processed_root)) if qc_path else None,
          "started": started, "finished": datetime.now().isoformat(timespec="seconds"),
          **stats}
    atomic_write_json(ck, ck_path, cfg)
    return ck


# ----------------------------------------------------------------------------------
# merge
# ----------------------------------------------------------------------------------
def merge_split(cfg: PreprocessConfig, manifest: pd.DataFrame, split: str, config_hash: str) -> dict:
    """Merge every valid per-sequence CSV of this split (from any previous run)."""
    seqs = sort_sequences(manifest[manifest["split"] == split])
    frames, stats_rows, missing, problem = [], [], [], []
    for vid in seqs["video_id"]:
        ck_path = cfg.checkpoints_dir(split) / f"{vid}.json"
        if not checkpoint_is_valid(cfg, split, vid, config_hash):
            if ck_path.exists():
                try:
                    st = json.loads(ck_path.read_text(encoding="utf-8")).get("status")
                except json.JSONDecodeError:
                    st = "corrupt"
                (problem if st in ("problem", "failed") else missing).append(vid)
            else:
                missing.append(vid)
            continue
        ck = json.loads(ck_path.read_text(encoding="utf-8"))
        d = pd.read_csv(cfg.sequence_records_dir(split) / f"{vid}.csv", dtype={"video_id": str})
        if len(d) != ck["n_rows"]:
            raise RuntimeError(f"{split}/{vid}: CSV has {len(d)} rows, checkpoint says {ck['n_rows']}")
        frames.append(d)
        stats_rows.append({k: (v if not isinstance(v, list) else " | ".join(v))
                           for k, v in ck.items() if k not in ("traceback",)})
    if not frames:
        return {"split": split, "merged_sequences": 0, "manifest_sequences": len(seqs),
                "missing": len(missing), "problem_or_failed": len(problem), "rows": 0}
    merged = pd.concat(frames, ignore_index=True)
    # integrity: one split, unique (video_id, t_ms), consecutive sample_idx
    assert (merged["split"] == split).all(), "rows from another split in merge"
    dup = merged.duplicated(["video_id", "t_ms"])
    assert not dup.any(), f"duplicate (video_id, t_ms): {merged.loc[dup, ['video_id', 't_ms']].head()}"
    atomic_write_csv(merged, cfg.aligned_records_path(split), cfg)
    atomic_write_csv(pd.DataFrame(stats_rows), cfg.sequence_stats_path(split), cfg)
    atomic_write_json({"schema_version": SCHEMA_VERSION, "config_hash": config_hash,
                       "config": {k: str(v) for k, v in vars(cfg).items()},
                       "merged_at": datetime.now().isoformat(timespec="seconds"),
                       "merged_sequences": len(frames), "manifest_sequences": len(seqs),
                       "rows": len(merged), "missing": missing, "problem_or_failed": problem},
                      cfg.metadata_dir(split) / "_schema.json", cfg)
    return {"split": split, "merged_sequences": len(frames), "manifest_sequences": len(seqs),
            "missing": len(missing), "problem_or_failed": len(problem), "rows": len(merged)}


# ----------------------------------------------------------------------------------
# guards, logging, CLI
# ----------------------------------------------------------------------------------
def guard_old_layout(cfg: PreprocessConfig, splits) -> None:
    """Stop if the pilot-v1 layout (JPEG frames, old CSV) is still in place."""
    bad = []
    for s in splits:
        if (cfg.split_dir(s) / "frames").exists():
            bad.append(cfg.split_dir(s) / "frames")
        old_csv = cfg.aligned_records_path(s)
        if old_csv.exists() and not (cfg.metadata_dir(s) / "_schema.json").exists():
            bad.append(old_csv)
    if bad:
        raise SystemExit(
            "Pilot-v1 outputs found (JPEG frames / old aligned_records.csv):\n  "
            + "\n  ".join(map(str, bad))
            + "\nArchive them first (step0_housekeeping.ps1 moves them to data/processed/_archive_pilot_v1/).")


def setup_logging(cfg: PreprocessConfig) -> Path:
    log_dir = cfg.processed_root / "logs"
    cfg.assert_safe_output(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return path


def append_processing_log(cfg: PreprocessConfig, ck: dict) -> None:
    path = cfg.logs_dir(ck["split"]) / "processing_log.csv"
    cfg.assert_safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "finished": ck.get("finished"), "video_id": ck["video_id"], "status": ck["status"],
        "n_rows": ck.get("n_rows"), "elapsed_sec": ck.get("elapsed_sec"),
        "problems": " | ".join(ck.get("problems", [])) or ck.get("error", ""),
        "warnings": " | ".join(ck.get("warnings", [])), "config_hash": ck.get("config_hash"),
    }])
    row.to_csv(path, mode="a", header=not path.exists(), index=False, lineterminator="\n")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true", help="extended pilot subset (3/split + edge cases)")
    ap.add_argument("--split", choices=["training", "validation", "test"])
    ap.add_argument("--videos", nargs="+", help="explicit video ids")
    ap.add_argument("--limit", type=int, help="process at most N sequences")
    ap.add_argument("--force", action="store_true", help="recompute even if a valid checkpoint exists")
    ap.add_argument("--qc-fraction", type=float, help="override share of sequences with a QC overlay")
    ap.add_argument("--no-merge", action="store_true", help="skip building aligned_records.csv")
    ap.add_argument("--dry-run", action="store_true", help="show selection and row estimate, write nothing")
    ap.add_argument("--processed-root", type=Path, help="override output root (tests)")
    ap.add_argument("--bdda-root", type=Path, help="override raw data root (tests)")
    ap.add_argument("--manifest", type=Path, help="override manifest path (tests)")
    return ap.parse_args(argv)


def build_config(args) -> PreprocessConfig:
    kw = {}
    if args.processed_root:
        kw["processed_root"] = args.processed_root
    if args.bdda_root:
        kw["bdda_root"] = args.bdda_root
    if args.manifest:
        kw["manifest_csv"] = args.manifest
    if args.qc_fraction is not None:
        kw["qc_fraction"] = args.qc_fraction
    return PreprocessConfig(**kw)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    manifest = load_manifest(cfg)
    sel = select_sequences(manifest, split=args.split, video_ids=args.videos,
                           pilot=args.pilot, limit=args.limit)
    config_hash = cfg.output_hash()
    n_rows_est = sum(len(sample_plan(int(r.cam_frames), float(r.cam_fps), int(r.gaze_frames),
                                     float(r.gaze_fps), cfg.sampling_fps)) for r in sel.itertuples())
    if args.dry_run:
        print(f"selected {len(sel)} sequences {sel['split'].value_counts().to_dict()}, "
              f"expected rows {n_rows_est}, config hash {config_hash}")
        print(sel[["split", "video_id", "cam_fps", "cam_duration"]].to_string(index=False))
        return 0

    splits = list(dict.fromkeys(sel["split"]))
    cfg.assert_safe_output(cfg.processed_root)
    guard_old_layout(cfg, splits)
    log_path = setup_logging(cfg)
    log.info("DriveSight Phase 2 | schema %s | config %s | %d sequences %s | expected rows %d",
             SCHEMA_VERSION, config_hash, len(sel), sel["split"].value_counts().to_dict(), n_rows_est)

    counts = {"done": 0, "problem": 0, "failed": 0, "skipped": 0}
    t_start = time.perf_counter()
    try:
        for i, row in enumerate(sel.itertuples(index=False), 1):
            row = pd.Series(row._asdict())
            split, vid = row["split"], row["video_id"]
            # "problem" sequences are not retried: the data, not the run, is at fault.
            # "failed" (exception) sequences are retried on every run.
            if not args.force and checkpoint_is_valid(cfg, split, vid, config_hash, ("done", "problem")):
                counts["skipped"] += 1
                continue
            ck = run_one(row, cfg, config_hash)
            append_processing_log(cfg, ck)
            counts[ck["status"]] += 1
            el = time.perf_counter() - t_start
            msg = (f"[{i}/{len(sel)}] {split}/{vid}: {ck['status']} "
                   f"rows={ck.get('n_rows')} t={ck.get('elapsed_sec')}s")
            if ck.get("problems"):
                msg += " | PROBLEMS: " + "; ".join(ck["problems"])
            if ck.get("error"):
                msg += " | ERROR: " + ck["error"]
            if ck.get("warnings"):
                msg += " | " + "; ".join(ck["warnings"])
            log.info(msg)
            if i % 25 == 0:
                rate = el / max(counts["done"] + counts["problem"] + counts["failed"], 1)
                log.info("  ~%.1f s/sequence, ETA %.1f min", rate, rate * (len(sel) - i) / 60)
    except KeyboardInterrupt:
        log.warning("Interrupted. Finished sequences are checkpointed; rerun the same command to resume.")
        return 130

    log.info("processing finished: %s in %.1f min", counts, (time.perf_counter() - t_start) / 60)
    if not args.no_merge:
        for s in splits:
            res = merge_split(cfg, manifest, s, config_hash)
            log.info("merge %s: %s", s, res)
    summary = {"config_hash": config_hash, "counts": counts, "log": str(log_path),
               "selected": len(sel), "expected_rows": n_rows_est}
    atomic_write_json(summary, log_path.with_name(log_path.stem + "_summary.json"), cfg)
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
