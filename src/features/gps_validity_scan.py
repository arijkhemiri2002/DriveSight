"""
Step 1 of Phase 2: read-only scan of every BDDA GPS file for invalid values.

The Phase 1 audit checked GPS timing (coverage, intervals, ordering) but not the values.
Validation/1744 is known to contain speed = -1.0 (iOS "unknown" sentinel); this scan
tells us whether it is an isolated case before the pipeline relies on GPS fields.

Outputs (new files, nothing else is touched):
  reports/video_dataset/gps_validity_scan.csv   one row per GPS file
  reports/video_dataset/gps_validity_scan.md    summary

Run:  python -m src.features.gps_validity_scan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.preprocess_config import PreprocessConfig
from src.features.manifest_loader import load_manifest


def scan_file(path: Path, max_speed: float) -> dict:
    out = {"file_ok": True, "error": ""}
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # unreadable JSON is itself a finding
        return {"file_ok": False, "error": f"{type(e).__name__}: {e}"}
    df = pd.DataFrame(meta.get("locations") or [])
    start = meta.get("startTime")
    out["n_records"] = len(df)
    out["start_time_missing"] = start is None
    if df.empty:
        return out

    def col(c):
        return pd.to_numeric(df[c], errors="coerce").to_numpy(float) if c in df else np.full(len(df), np.nan)

    ts, sp, co = col("timestamp"), col("speed"), col("course")
    lat, lon, acc = col("latitude"), col("longitude"), col("accuracy")
    out.update(
        n_dup_timestamps=int(pd.Series(ts).duplicated().sum()),
        n_unsorted=int((np.diff(ts) < 0).sum()) if len(ts) > 1 else 0,
        n_speed_negative=int((sp < 0).sum()),
        n_speed_missing=int(np.isnan(sp).sum()),
        n_speed_too_high=int((sp > max_speed).sum()),
        n_course_invalid=int((np.isfinite(co) & ((co < 0) | (co >= 360))).sum() + np.isnan(co).sum()),
        n_accuracy_negative=int((acc < 0).sum()),
        n_position_zero=int(((lat == 0) & (lon == 0)).sum()),
        n_position_out_of_range=int(((np.abs(lat) > 90) | (np.abs(lon) > 180)).sum()),
        speed_min=float(np.nanmin(sp)) if np.isfinite(sp).any() else np.nan,
        speed_max=float(np.nanmax(sp)) if np.isfinite(sp).any() else np.nan,
        negative_speed_values=";".join(sorted({f"{v:g}" for v in sp[sp < 0]})),
    )
    if start is not None and len(ts):
        t = (np.sort(ts) - float(start)) / 1000.0
        out["first_offset_sec"] = round(float(t[0]), 3)
        out["max_interval_sec"] = round(float(np.diff(t).max()), 3) if len(t) > 1 else np.nan
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bdda-root", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args(argv)
    kw = {}
    if args.bdda_root:
        kw["bdda_root"] = args.bdda_root
    if args.manifest:
        kw["manifest_csv"] = args.manifest
    cfg = PreprocessConfig(**kw)
    out_dir = args.out_dir or cfg.reports_dir
    cfg.assert_safe_output(out_dir)

    clean = set()
    try:
        m = load_manifest(cfg)
        clean = set(zip(m["split"], m["video_id"]))
    except FileNotFoundError:
        print("manifest not found: in_clean_manifest will be False everywhere")

    rows = []
    for split in cfg.splits:
        for p in sorted((cfg.bdda_root / split / "gps_jsons").glob("*.json")):
            r = {"split": split, "video_id": p.stem, "in_clean_manifest": (split, p.stem) in clean}
            r.update(scan_file(p, cfg.gps_max_speed_m_s))
            rows.append(r)
    if not rows:
        print(f"No GPS files found under {cfg.bdda_root}")
        return 1
    df = pd.DataFrame(rows)
    issue_cols = ["n_dup_timestamps", "n_unsorted", "n_speed_negative", "n_speed_missing",
                  "n_speed_too_high", "n_course_invalid", "n_accuracy_negative",
                  "n_position_zero", "n_position_out_of_range"]
    for c in issue_cols:
        if c not in df:
            df[c] = 0
    df[issue_cols] = df[issue_cols].fillna(0).astype(int)
    df["any_issue"] = (df[issue_cols].sum(axis=1) > 0) | ~df["file_ok"].astype(bool)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "gps_validity_scan.csv", index=False)

    c = df[df["in_clean_manifest"]]
    lines = ["# GPS validity scan", "",
             f"Files scanned: {len(df)} (clean manifest: {len(c)})", "",
             "Rules: speed valid if 0 <= speed <= "
             f"{cfg.gps_max_speed_m_s} m/s; course valid if 0 <= course < 360; position valid if "
             "lat/lon in range, not (0,0), accuracy >= 0.", "",
             "## Records with each issue", "",
             "| Issue | All files: records | All files: files | Clean manifest: records | Clean manifest: files |",
             "|---|---|---|---|---|"]
    for col in issue_cols:
        lines.append(f"| {col} | {int(df[col].sum())} | {int((df[col] > 0).sum())} | "
                     f"{int(c[col].sum())} | {int((c[col] > 0).sum())} |")
    lines += ["", f"Unreadable files: {int((~df['file_ok'].astype(bool)).sum())}", "",
              "## Clean-manifest sequences with any issue", ""]
    bad = c[c["any_issue"]]
    if bad.empty:
        lines.append("None.")
    else:
        lines.append("| Split | Video | " + " | ".join(issue_cols) + " | negative speed values |")
        lines.append("|---|---|" + "---|" * (len(issue_cols) + 1))
        for _, r in bad.iterrows():
            lines.append(f"| {r['split']} | {r['video_id']} | " + " | ".join(str(int(r[k])) for k in issue_cols)
                         + f" | {r.get('negative_speed_values', '')} |")
    lines += ["", "Handling in the pipeline: invalid values are set to NaN with a *_valid flag; "
              "acceleration is only computed between two valid consecutive records."]
    (out_dir / "gps_validity_scan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out_dir / 'gps_validity_scan.csv'}\n-> {out_dir / 'gps_validity_scan.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
