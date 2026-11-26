"""
Phase 4 threshold tuning on the VALIDATION split only (95 sequences). Never run it on test.

For every setting of a small grid, the tracker is run in memory on the validation sequences (nothing is
written to data/processed) and the indicators of evaluate_tracking.metrics are computed.

Selection rule (written down before looking at the results, so it cannot be bent to a preferred outcome):
    score = coverage of conf >= 0.5 boxes  -  share of 1-2 frame tracks  -  share of consecutive pairs with
            no overlap  -  share of tracks with a class change
Each term is a share in [0, 1] and gets the same weight: more confident detections linked into tracks is
good; fragmentation, impossible jumps and mixed classes are bad. The table keeps every setting, so the
choice can be checked or overruled with a written reason. The chosen setting is then frozen (defaults in
src/tracking/track_config.py, or the matching command-line options) and used unchanged on all splits.

Run:  python -m src.tracking.tune_tracking            (full grid, a few minutes)
      python -m src.tracking.tune_tracking --quick    (12 settings, to check that it runs)
-> reports/video_dataset/phase4_tuning_validation.md and .csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.features.preprocess_config import PreprocessConfig
from src.features.manifest_loader import load_manifest, select_sequences
from src.detection.detect_config import DetectConfig
from src.tracking.track_config import TrackConfig, ULTRALYTICS_DEFAULTS
from src.tracking.tracker_core import track_sequence, summarize_tracks
from src.tracking.evaluate_tracking import metrics, flat
from src.tracking.run_tracking import phase3_status, phase3_files

FULL = {"track_high_thresh": [0.25, 0.40, 0.50], "new_track_thresh": [0.25, 0.40, 0.50, 0.60],
        "match_thresh": [0.70, 0.80, 0.90], "track_buffer_sec": [0.4, 1.0, 2.0]}
QUICK = {"track_high_thresh": [0.25, 0.50], "new_track_thresh": [0.25, 0.60],
         "match_thresh": [0.80], "track_buffer_sec": [0.4, 1.0, 2.0]}


def grid(spec: dict) -> list[dict]:
    keys = list(spec)
    out = []
    for vals in itertools.product(*(spec[k] for k in keys)):
        d = dict(zip(keys, vals))
        if d["new_track_thresh"] < d["track_high_thresh"]:      # a new track never starts below the first-stage score
            continue
        out.append(d)
    return out


def score(row: dict) -> float:
    return (row["all_coverage_conf50"] - row["all_short_frac"] - row["all_no_overlap_frac"]
            - row["all_class_change_frac"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--limit", type=int, help="first N validation sequences only (for a dry run)")
    ap.add_argument("--processed-root", type=Path)
    ap.add_argument("--bdda-root", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    pkw = {k: v for k, v in (("processed_root", a.processed_root), ("bdda_root", a.bdda_root),
                             ("manifest_csv", a.manifest)) if v}
    pre = PreprocessConfig(**pkw)
    base = TrackConfig(det=DetectConfig(pre=pre), **ULTRALYTICS_DEFAULTS)   # untuned values for the fixed parameters
    sel = select_sequences(load_manifest(pre), split="validation", limit=a.limit)   # validation ONLY

    data = []
    for r in sel.itertuples(index=False):
        vid = str(r.video_id)
        ok, _, why = phase3_status(base, "validation", vid)
        if not ok:
            print(f"skip validation/{vid}: {why}")
            continue
        p = phase3_files(base, "validation", vid)
        b = pd.read_parquet(p["boxes"])
        b["video_id"] = b["video_id"].astype(str)
        f = pd.read_csv(p["frames"], usecols=["sample_idx", "t_ms"])
        data.append((vid, b, f))
    if not data:
        raise SystemExit("No usable validation sequence (Phase 3 outputs missing?).")
    all_boxes = pd.concat([b for _, b, _ in data], ignore_index=True)

    settings = grid(QUICK if a.quick else FULL)
    print(f"{len(settings)} settings x {len(data)} validation sequences")
    rows = []
    for i, s in enumerate(settings, 1):
        cfg = replace(base, **s)
        t0 = time.perf_counter()
        parts = []
        for vid, b, f in data:
            t, _ = track_sequence(b, f, cfg)
            t["split"], t["video_id"] = "validation", vid
            parts.append(t)
        tracks = pd.concat(parts, ignore_index=True)
        summ = summarize_tracks(tracks)
        row = {**s, "track_buffer_frames": cfg.track_buffer, **flat(metrics(tracks, summ, all_boxes))}
        row["score"] = score(row)
        row["seconds"] = round(time.perf_counter() - t0, 1)
        rows.append(row)
        print(f"[{i}/{len(settings)}] {s} score={row['score']:.3f} cov50={row['all_coverage_conf50']:.3f} "
              f"short={row['all_short_frac']:.3f} no_overlap={row['all_no_overlap_frac']:.3f} "
              f"class_change={row['all_class_change_frac']:.3f} ({row['seconds']}s)")

    d0 = TrackConfig(**ULTRALYTICS_DEFAULTS)
    df = pd.DataFrame(rows)
    # ties are broken in favour of the Ultralytics defaults, then of the setting closest to them
    df["dist_to_default"] = (abs(df["track_high_thresh"] - d0.track_high_thresh) + abs(df["new_track_thresh"] - d0.new_track_thresh)
                             + abs(df["match_thresh"] - d0.match_thresh) + abs(df["track_buffer_sec"] - d0.track_buffer_sec))
    df = df.sort_values(["score", "dist_to_default"], ascending=[False, True]).reset_index(drop=True)
    out = a.out or (pre.reports_dir / "phase4_tuning_validation.md")
    pre.assert_safe_output(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    best = df.iloc[0]
    default = df[(df["track_high_thresh"] == d0.track_high_thresh) & (df["new_track_thresh"] == d0.new_track_thresh)
                 & (df["match_thresh"] == d0.match_thresh) & (df["track_buffer_sec"] == d0.track_buffer_sec)]
    show = ["track_high_thresh", "new_track_thresh", "match_thresh", "track_buffer_sec", "score", "all_coverage_conf50",
            "all_short_frac", "all_no_overlap_frac", "all_class_change_frac", "all_median_len", "vru_coverage_conf50",
            "vru_short_frac", "all_tracks"]
    fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
    L = ["# Phase 4 tuning on the validation split", "",
         f"{len(data)} validation sequences, {len(df)} settings, track_low_thresh {base.track_low_thresh} and "
         f"fuse_score {base.fuse_score} fixed (Ultralytics defaults).", "",
         "Selection rule (fixed in advance): score = coverage(conf >= 0.5) - share of 1-2 frame tracks - share of "
         "pairs with no overlap - share of tracks with a class change. Ties go to the Ultralytics defaults, then to "
         "the setting closest to them.", "",
         "## Recommended setting", "",
         "```", json.dumps({k: (float(best[k])) for k in ("track_high_thresh", "new_track_thresh", "match_thresh",
                                                          "track_buffer_sec")}), "```", "",
         f"Score {best['score']:.3f}" + (f" vs Ultralytics defaults {default.iloc[0]['score']:.3f}" if len(default) else ""), "",
         "Command for every split (frozen setting):", "",
         "```",
         f"python -m src.tracking.run_tracking --track-high-thresh {best['track_high_thresh']} --new-track-thresh "
         f"{best['new_track_thresh']} --match-thresh {best['match_thresh']} --track-buffer-sec {best['track_buffer_sec']}",
         "```", "", "## All settings (best first)", "",
         "| " + " | ".join(show) + " |", "|" + "---|" * len(show)]
    for _, r in df.iterrows():
        L.append("| " + " | ".join(fmt(r[c]) for c in show) + " |")
    L += ["", "Write down why the chosen setting is kept (or why another is preferred) in docs/phase4_tracking.md."]
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
