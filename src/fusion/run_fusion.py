"""
DriveSight Phase 5, step 1: fuse tracks (Phase 4), gaze and GPS (Phase 2) for every sequence, and
compute the braking labels. No video is decoded and no GPU is needed (about 0.1 s per sequence).

Reads (read-only, must be complete and verified):
    data/processed/{split}/metadata/aligned_records.csv + _schema.json     Phase 2
    data/processed/{split}/tracks/tracks.parquet + _schema.json            Phase 4
Writes data/processed/{split}/fusion/:
    frame_features.parquet    one row per Phase 2 row: features (src/fusion/fusion_core.py) + label
    object_features.parquet   one row per tracked box: per-object features
    braking_events.csv        one row per sequence
    _schema.json              fusion hash (settings + upstream hashes), input SHA-256, coverage

A split whose _schema.json already has the same fusion hash is skipped (use --force to redo it).

Run from the repository root:
    python -m src.fusion.run_fusion                    # all splits
    python -m src.fusion.run_fusion --split validation
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.fusion.fusion_config import (FUSION_SCHEMA_VERSION, SPLITS, FusionConfig, assert_safe_output,
                                      file_sha256)
from src.fusion.fusion_core import FRAME_COLUMNS, OBJECT_COLUMNS, fuse_sequence
from src.fusion.gps_labels import EVENT_COLUMNS


# ------------------------------------------------------------------------------------------ io helpers
def _atomic(path: Path, write) -> None:
    assert_safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    _atomic(path, lambda p: df.to_parquet(p, index=False))


def write_csv(df: pd.DataFrame, path: Path) -> None:
    _atomic(path, lambda p: df.to_csv(p, index=False))


def write_json(obj: dict, path: Path) -> None:
    def w(p):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
    _atomic(path, w)


# ------------------------------------------------------------------------------------------- upstream
def upstream_info(split: str, cfg: FusionConfig) -> dict:
    """Phase 2 and Phase 4 hashes of a split; raises if an input is missing or incomplete."""
    need = [cfg.records_path(split), cfg.records_schema_path(split), cfg.tracks_path(split),
            cfg.tracks_schema_path(split)]
    missing = [str(p) for p in need if not Path(p).exists()]
    if missing:
        raise SystemExit(f"[{split}] missing input(s): {missing}. Run Phases 2-4 first.")
    s2 = json.loads(Path(cfg.records_schema_path(split)).read_text(encoding="utf-8"))
    s4 = json.loads(Path(cfg.tracks_schema_path(split)).read_text(encoding="utf-8"))
    if s2.get("missing") or s4.get("missing"):
        raise SystemExit(f"[{split}] Phase 2 or Phase 4 output is incomplete (missing sequences).")
    return {"phase2_config_hash": s2.get("config_hash"), "phase2_schema": s2.get("schema_version"),
            "phase2_records_sha256": file_sha256(cfg.records_path(split)),
            "phase4_config_hash": s4.get("config_hash"), "phase4_schema": s4.get("schema_version"),
            "phase4_tracks_sha256": file_sha256(cfg.tracks_path(split))}


def is_up_to_date(split: str, cfg: FusionConfig, fhash: str) -> bool:
    p = cfg.fusion_schema_path(split)
    if not p.exists():
        return False
    s = json.loads(p.read_text(encoding="utf-8"))
    if s.get("fusion_hash") != fhash:
        return False
    for key, path in (("frame_features_sha256", cfg.frame_features_path(split)),
                      ("object_features_sha256", cfg.object_features_path(split)),
                      ("events_sha256", cfg.events_path(split))):
        if not Path(path).exists() or file_sha256(path) != s.get(key):
            return False
    return True


# ---------------------------------------------------------------------------------------------- run
def load_inputs(split: str, cfg: FusionConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(cfg.records_path(split), dtype={"video_id": str, "split": str})
    tracks = pd.read_parquet(cfg.tracks_path(split))
    tracks["video_id"] = tracks["video_id"].astype(str)
    return rows, tracks


def fuse_split(split: str, cfg: FusionConfig, videos: list[str] | None = None, force: bool = False,
               write: bool = True) -> dict:
    up = upstream_info(split, cfg)
    fhash = cfg.fusion_hash(up)
    if write and not force and videos is None and is_up_to_date(split, cfg, fhash):
        print(f"[{split}] up to date (fusion hash {fhash}) - skipped")
        return {"split": split, "skipped": True, "fusion_hash": fhash}
    t0 = time.time()
    rows, tracks = load_inputs(split, cfg)
    by_vid = {v: g for v, g in tracks.groupby("video_id", sort=False)}
    empty = tracks.iloc[0:0]
    frames, objects, events = [], [], []
    vids = sorted(rows["video_id"].unique(), key=lambda v: int(v) if v.isdigit() else v)
    if videos is not None:
        vids = [v for v in vids if v in set(videos)]
    for i, vid in enumerate(vids, 1):
        fr, ob, ev = fuse_sequence(split, vid, rows[rows["video_id"] == vid], by_vid.get(vid, empty), cfg)
        frames.append(fr)
        objects.append(ob)
        events.append(ev)
        if i % 100 == 0 or i == len(vids):
            print(f"[{split}] {i}/{len(vids)} sequences ({time.time() - t0:.0f} s)")
    fr = pd.concat(frames, ignore_index=True)[FRAME_COLUMNS]
    ob = pd.concat([o for o in objects if len(o)], ignore_index=True)[OBJECT_COLUMNS] if objects else \
        pd.DataFrame(columns=OBJECT_COLUMNS)
    ev = pd.DataFrame(events).reindex(columns=EVENT_COLUMNS)
    stats = coverage(fr, ev, rows, tracks)
    if not write:
        return {"split": split, "frames": fr, "objects": ob, "events": ev, "stats": stats, "fusion_hash": fhash}
    write_parquet(fr, cfg.frame_features_path(split))
    write_parquet(ob, cfg.object_features_path(split))
    write_csv(ev, cfg.events_path(split))
    schema = {"schema_version": FUSION_SCHEMA_VERSION, "fusion_hash": fhash, "upstream": up,
              "params": cfg.fusion_params(), "stats": stats,
              "frame_features_sha256": file_sha256(cfg.frame_features_path(split)),
              "object_features_sha256": file_sha256(cfg.object_features_path(split)),
              "events_sha256": file_sha256(cfg.events_path(split)),
              "created_at": datetime.now().isoformat(timespec="seconds"),
              "seconds": round(time.time() - t0, 1)}
    write_json(schema, cfg.fusion_schema_path(split))
    print(f"[{split}] {stats['sequences']} sequences, {stats['frames']} frames, {stats['in_scope']} in scope, "
          f"{stats['positives']} positive ({100 * stats['positive_rate']:.1f} %), "
          f"{stats['with_event']} clips with a hard braking - {time.time() - t0:.0f} s")
    return {"split": split, "skipped": False, "fusion_hash": fhash, "stats": stats}


def coverage(fr: pd.DataFrame, ev: pd.DataFrame, rows: pd.DataFrame, tracks: pd.DataFrame) -> dict:
    s = fr[fr["in_scope"]]
    e = ev[ev["has_event"].astype(bool)]
    return {"sequences": int(fr["video_id"].nunique()), "frames": int(len(fr)), "phase2_rows": int(len(rows)),
            "tracked_boxes": int(len(tracks)), "in_scope": int(len(s)), "positives": int(s["y"].sum()),
            "positive_rate": float(s["y"].mean()) if len(s) else float("nan"),
            "with_event": int(len(e)), "onset_median_sec": float(e["onset_sec"].median()) if len(e) else None,
            "onset_share_4p5_8p5": float(e["onset_sec"].between(4.5, 8.5).mean()) if len(e) else None,
            "peak_decel_median": float(e["peak_decel"].median()) if len(e) else None}


def build_config(args) -> FusionConfig:
    kw = {}
    if args.processed_root:
        kw["processed_root"] = args.processed_root
    if args.reports_dir:
        kw["reports_dir"] = args.reports_dir
    if args.no_event_policy:
        kw["no_event_policy"] = args.no_event_policy
    return FusionConfig(**kw)


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--processed-root", default=None)
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--no-event-policy", choices=("exclude", "negative"), default=None)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=SPLITS, action="append")
    ap.add_argument("--force", action="store_true")
    add_common_args(ap)
    args = ap.parse_args(argv)
    cfg = build_config(args)
    for split in args.split or SPLITS:
        fuse_split(split, cfg, force=args.force)


if __name__ == "__main__":
    main()
