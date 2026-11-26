"""
Phase 4 evaluation WITHOUT ground truth (BDDA has no annotated trajectories).

These are indirect indicators, reported as such. None of them proves that an identity is correct.
  length         track length in frames (5 Hz); share of very short tracks (1-2 frames) = fragmentation
  class changes  share of tracks whose boxes have more than one class (sign of two objects merged, or of
                 detector class confusion such as motorcycle/bicycle)
  jumps          share of consecutive box pairs inside a track whose centre moves more than JUMP_FRAC of the
                 image (normalised units), or whose boxes do not overlap at all (IoU = 0)
  coverage       share of stored boxes with conf >= 0.5 that end up in a track (and the same for VRUs)
  gaps           share of tracks with at least one missing frame bridged by the tracker
Each indicator is given for all tracks, VRU tracks and vehicle tracks.

Run (reads the merged outputs of run_tracking.py):
    python -m src.tracking.evaluate_tracking --split validation
-> reports/video_dataset/phase4_evaluation_{split}.md and .csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.preprocess_config import PreprocessConfig
from src.detection.detect_config import DetectConfig
from src.tracking.track_config import TrackConfig

JUMP_FRAC = 0.10          # centre displacement between two consecutive boxes of a track, normalised image units
CONF_HI = 0.50
VEHICLES = ("car", "truck", "bus")


def _pairs(t: pd.DataFrame) -> pd.DataFrame:
    o = t.sort_values(["video_id", "track_id", "t_ms"])
    nxt = o.groupby(["video_id", "track_id"]).shift(-1)
    m = nxt["t_ms"].notna()
    a, b = o[m], nxt[m]
    jump = np.hypot(b["cx_n"].to_numpy() - a["cx_n"].to_numpy(), b["cy_n"].to_numpy() - a["cy_n"].to_numpy())
    ix = np.clip(np.minimum(a["x2"].to_numpy(), b["x2"].to_numpy()) - np.maximum(a["x1"].to_numpy(), b["x1"].to_numpy()), 0, None)
    iy = np.clip(np.minimum(a["y2"].to_numpy(), b["y2"].to_numpy()) - np.maximum(a["y1"].to_numpy(), b["y1"].to_numpy()), 0, None)
    return pd.DataFrame({"video_id": a["video_id"].to_numpy(), "track_id": a["track_id"].to_numpy(),
                         "jump": jump, "overlap": ix * iy > 0})


def metrics(tracks: pd.DataFrame, summary: pd.DataFrame, boxes: pd.DataFrame) -> dict:
    """Indicators for one set of tracks. boxes = the Phase 3 boxes of the same sequences."""
    out = {}
    hi = boxes[boxes["conf"] >= CONF_HI]
    tk = set(zip(tracks["video_id"], tracks["t_ms"], tracks["det_idx"]))
    pairs = _pairs(tracks) if len(tracks) else pd.DataFrame(columns=["video_id", "track_id", "jump", "overlap"])
    groups = {"all": summary, "vru": summary[summary["is_vru"].astype(bool)],
              "vehicle": summary[summary["cls_name"].isin(VEHICLES)]}
    for name, s in groups.items():
        keys = set(zip(s["video_id"], s["track_id"]))
        p = pairs[[k in keys for k in zip(pairs["video_id"], pairs["track_id"])]] if len(pairs) else pairs
        if name == "all":
            h = hi
        elif name == "vru":
            h = hi[hi["is_vru"].astype(bool)]
        else:
            h = hi[hi["cls_name"].isin(VEHICLES)]
        cov = sum(k in tk for k in zip(h["video_id"], h["t_ms"], h["det_idx"]))
        n = s["n_boxes"] if len(s) else pd.Series(dtype=float)
        out[name] = {
            "tracks": int(len(s)),
            "median_len": float(n.median()) if len(n) else np.nan,
            "mean_len": float(n.mean()) if len(n) else np.nan,
            "short_frac": float((n <= 2).mean()) if len(n) else np.nan,
            "long_frac": float((n >= 10).mean()) if len(n) else np.nan,
            "class_change_frac": float((s["n_classes"] > 1).mean()) if len(s) else np.nan,
            "gap_frac": float((s["n_gaps"] > 0).mean()) if len(s) else np.nan,
            "pairs": int(len(p)),
            "jump_frac": float((p["jump"] > JUMP_FRAC).mean()) if len(p) else np.nan,
            "no_overlap_frac": float((~p["overlap"].astype(bool)).mean()) if len(p) else np.nan,
            "coverage_conf50": float(cov / len(h)) if len(h) else np.nan,
            "boxes_conf50": int(len(h)),
        }
    return out


def flat(m: dict) -> dict:
    return {f"{g}_{k}": v for g, d in m.items() for k, v in d.items()}


def write_report(m: dict, split: str, params: dict | None, out_md: Path, cfg: TrackConfig) -> None:
    rows = [("tracks", "Tracks", "{:.0f}"), ("median_len", "Median length (frames)", "{:.0f}"),
            ("mean_len", "Mean length (frames)", "{:.1f}"), ("short_frac", "Tracks of 1-2 frames", "{:.1%}"),
            ("long_frac", "Tracks of >= 10 frames (2 s)", "{:.1%}"), ("class_change_frac", "Tracks with a class change", "{:.1%}"),
            ("gap_frac", "Tracks with a bridged gap", "{:.1%}"),
            ("jump_frac", f"Consecutive pairs with centre jump > {JUMP_FRAC}", "{:.1%}"),
            ("no_overlap_frac", "Consecutive pairs with no overlap (IoU = 0)", "{:.1%}"),
            ("coverage_conf50", f"Boxes with conf >= {CONF_HI} in a track", "{:.1%}")]
    L = [f"# Phase 4 evaluation without ground truth - {split}", "",
         "BDDA has no annotated trajectories. These indicators describe the tracks; they do not measure identity "
         "accuracy (that needs ground truth, e.g. the BDD100K MOT set).", ""]
    if params:
        L += [f"Tracker settings: `{params}`", ""]
    L += ["| Indicator | All | VRU | Vehicles |", "|---|---|---|---|"]
    for k, lab, f in rows:
        cells = []
        for g in ("all", "vru", "vehicle"):
            v = m[g][k]
            cells.append("n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v))
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L += ["", f"- Jump threshold {JUMP_FRAC} (normalised image units) and confidence {CONF_HI} are reporting choices, "
          "not tuned values.", "- At 5 Hz, a near object can move most of its own width between two frames; "
          "IoU-based association then fails (measured on synthetic boxes: tracking holds up to ~60 % of the box "
          "width per frame). Fast, close objects are the expected source of fragmentation."]
    cfg.pre.assert_safe_output(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    import json
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="validation", choices=["training", "validation", "test"])
    ap.add_argument("--processed-root", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    pre = PreprocessConfig(**({"processed_root": a.processed_root} if a.processed_root else {}))
    cfg = TrackConfig(det=DetectConfig(pre=pre))
    if not cfg.tracks_path(a.split).exists():
        raise SystemExit(f"{cfg.tracks_path(a.split)} not found: run python -m src.tracking.run_tracking --split {a.split}")
    t = pd.read_parquet(cfg.tracks_path(a.split))
    t["video_id"] = t["video_id"].astype(str)
    s = pd.read_csv(cfg.summary_path(a.split), dtype={"video_id": str})
    b = pd.read_parquet(cfg.det.detections_path(a.split))
    b["video_id"] = b["video_id"].astype(str)
    b = b[b["video_id"].isin(set(s["video_id"]) | set(t["video_id"]))]
    m = metrics(t, s, b)
    sj = cfg.tracks_dir(a.split) / "_schema.json"
    params = json.loads(sj.read_text(encoding="utf-8")).get("params") if sj.exists() else None
    out = a.out or (pre.reports_dir / f"phase4_evaluation_{a.split}.md")
    write_report(m, a.split, params, out, cfg)
    pd.DataFrame([flat(m)]).to_csv(out.with_suffix(".csv"), index=False)
    print(out.read_text(encoding="utf-8"))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
