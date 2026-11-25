"""
Independent verification of Phase 2 outputs (does not trust the runner's own summary).

Checks, per split in scope:
  coverage      every selected manifest sequence has a "done" checkpoint and its CSV hash matches
  row counts    rows per sequence == sampling plan recomputed from the manifest; merged == sum
  integrity     unique (video_id, t_ms); 200 ms steps; sample_idx 0..n-1; split column correct
  separation    each video_id appears in exactly one split, the one given by the manifest
  frames        all decodes ok; |residual| <= half a frame; |camera-gaze dt| <= half cam + half gaze frame
                last target inside both videos; decoded frame counts >= manifest
  GPS           no negative / implausible speed, invalid values NaN and flagged, course in [0, 360),
                association gap <= threshold
  storage       no bulk frame folders; size of outputs; number of QC overlays
  protection    weights/final SHA-256 == weights/final/SHA256.txt

Writes reports/video_dataset/phase2_verification_<scope>.md and exits 1 on any failure.
Run:  python -m src.features.verify_preprocessing --pilot
      python -m src.features.verify_preprocessing
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.features.preprocess_config import PreprocessConfig, PROJECT_ROOT
from src.features.manifest_loader import load_manifest, select_sequences
from src.features.frame_extractor import sample_plan, last_frame_time
from src.features.run_preprocessing import sha256_file

EPS = 1e-5


class Report:
    def __init__(self):
        self.lines: list[str] = []
        self.failures = 0

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        self.failures += int(not ok)
        self.lines.append(f"- {'PASS' if ok else '**FAIL**'} — {label}" + (f": {detail}" if detail else ""))

    def info(self, text: str) -> None:
        self.lines.append(text)


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def verify_weights(rep: Report) -> None:
    wdir = PROJECT_ROOT / "weights" / "final"
    ref = wdir / "SHA256.txt"
    if not ref.exists():
        rep.info("- (weights/final/SHA256.txt not found: weight check skipped)")
        return
    bad = []
    for line in ref.read_text().splitlines():
        if not line.strip():
            continue
        h, name = line.split()[0].lower(), line.split()[-1].lstrip("*")
        p = wdir / name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            bad.append(name)
    rep.check(not bad, "trained weights unchanged (SHA-256)", ", ".join(bad) or "3/3 match")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--split")
    ap.add_argument("--videos", nargs="+")
    ap.add_argument("--processed-root", type=Path)
    ap.add_argument("--bdda-root", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    kw = {k: v for k, v in (("processed_root", a.processed_root), ("bdda_root", a.bdda_root),
                            ("manifest_csv", a.manifest)) if v}
    cfg = PreprocessConfig(**kw)
    man = load_manifest(cfg)
    sel = select_sequences(man, split=a.split, video_ids=a.videos, pilot=a.pilot)
    scope = "pilot" if a.pilot else ("custom" if (a.split or a.videos) else "full")
    h = cfg.output_hash()
    rep = Report()
    rep.info(f"# Phase 2 verification — {scope}")
    rep.info("")
    rep.info(f"Config hash `{h}` | sequences in scope: {len(sel)} {sel['split'].value_counts().to_dict()}")
    rep.info("")

    seen_ids: dict[str, str] = {}
    totals = {"rows": 0, "expected": 0, "warn_speed": 0, "gaze_empty": 0}
    for split in [s for s in cfg.splits if s in set(sel["split"])]:
        rep.info(f"## {split}")
        rep.info("")
        ssel = sel[sel["split"] == split]
        ck_dir, rec_dir = cfg.checkpoints_dir(split), cfg.sequence_records_dir(split)
        status, hash_bad, expected = {}, [], {}
        cks = {}
        for r in ssel.itertuples():
            vid = r.video_id
            expected[vid] = len(sample_plan(int(r.cam_frames), float(r.cam_fps), int(r.gaze_frames),
                                            float(r.gaze_fps), cfg.sampling_fps))
            p = ck_dir / f"{vid}.json"
            if not p.exists():
                status[vid] = "missing"
                continue
            ck = json.loads(p.read_text(encoding="utf-8"))
            cks[vid] = ck
            status[vid] = ck.get("status") if ck.get("config_hash") == h else "stale-config"
            csv = rec_dir / f"{vid}.csv"
            if ck.get("status") == "done" and (not csv.exists() or sha256_file(csv) != ck.get("csv_sha256")):
                hash_bad.append(vid)
        cnt = pd.Series(status).value_counts().to_dict()
        rep.check(cnt.get("done", 0) == len(ssel), "every sequence has a done checkpoint (current config)", str(cnt))
        rep.check(not hash_bad, "per-sequence CSVs match their checkpoint hash", ", ".join(hash_bad[:20]))
        for vid, st in status.items():
            if st in ("problem", "failed"):
                ck = cks[vid]
                rep.info(f"  - {vid}: {st} — {'; '.join(ck.get('problems', [])) or ck.get('error', '')}")

        ap_path = cfg.aligned_records_path(split)
        if not ap_path.exists():
            rep.check(False, "aligned_records.csv exists")
            continue
        df = pd.read_csv(ap_path, dtype={"video_id": str})
        df = df[df["video_id"].isin(set(ssel["video_id"]))]
        done = [v for v, s in status.items() if s == "done"]
        per = df.groupby("video_id").size()
        wrong_n = {v: (int(per.get(v, 0)), expected[v]) for v in done if per.get(v, 0) != expected[v]}
        rep.check(not wrong_n, "rows per sequence == recomputed sampling plan",
                  f"{len(done)} sequences" if not wrong_n else str(dict(list(wrong_n.items())[:10])))
        rep.check(bool((df["split"] == split).all()), "split column consistent")
        rep.check(not df.duplicated(["video_id", "t_ms"]).any(), "unique (video_id, t_ms)")
        for v in df["video_id"].unique():
            if v in seen_ids and seen_ids[v] != split:
                rep.check(False, "video in one split only", f"{v} in {seen_ids[v]} and {split}")
            seen_ids[v] = split
        mans = man.set_index("video_id")["split"]
        rep.check(bool((df["video_id"].map(mans) == split).all()), "video split matches manifest")
        g = df.sort_values(["video_id", "t_ms"]).groupby("video_id")
        steps_ok = bool(g["t_ms"].diff().dropna().eq(round(1000 / cfg.sampling_fps)).all())
        idx_ok = bool((g.cumcount() == df.sort_values(["video_id", "t_ms"])["sample_idx"]).all())
        rep.check(steps_ok and idx_ok, f"{1000 / cfg.sampling_fps:.0f} ms steps and consecutive sample_idx")

        rep.check(bool(df["cam_ok"].all() and df["gaze_ok"].all()), "all camera and gaze decodes ok",
                  f"cam fail {(~df['cam_ok']).sum()}, gaze fail {(~df['gaze_ok']).sum()}")
        fps = df["video_id"].map(man.set_index("video_id")["cam_fps"]).astype(float)
        gfps = df["video_id"].map(man.set_index("video_id")["gaze_fps"]).astype(float)
        rep.check(bool((df["cam_residual_sec"].abs() <= 0.5 / fps + EPS).all()),
                  "|camera residual| <= half a camera frame", f"max {df['cam_residual_sec'].abs().max():.4f}s")
        rep.check(bool((df["gaze_residual_sec"].abs() <= 0.5 / gfps + EPS).all()),
                  "|gaze residual| <= half a gaze frame", f"max {df['gaze_residual_sec'].abs().max():.4f}s")
        rep.check(bool((df["cam_gaze_dt_sec"].abs() <= 0.5 / fps + 0.5 / gfps + EPS).all()),
                  "|camera - gaze frame time| <= half cam + half gaze frame",
                  f"max {df['cam_gaze_dt_sec'].abs().max():.4f}s")
        mrow = man.set_index("video_id")
        t_end = df.groupby("video_id")["t_sec"].max()
        lim = {v: min(last_frame_time(int(mrow.at[v, "cam_frames"]), float(mrow.at[v, "cam_fps"])),
                      last_frame_time(int(mrow.at[v, "gaze_frames"]), float(mrow.at[v, "gaze_fps"])))
               for v in t_end.index}
        rep.check(all(t_end[v] <= lim[v] + EPS for v in t_end.index), "last sample inside both videos")
        dec_bad = [v for v, ck in cks.items() if status.get(v) == "done" and
                   (ck["cam_frames_decoded"] < ck["cam_frames_manifest"] or
                    ck["gaze_frames_decoded"] < ck["gaze_frames_manifest"])]
        rep.check(not dec_bad, "decoded frame counts >= manifest counts", ", ".join(dec_bad[:20]))

        sp = df["gps_speed_m_s"]
        rep.check(bool((sp.dropna() >= 0).all() and (sp.dropna() <= cfg.gps_max_speed_m_s).all()),
                  "no negative or implausible speed values", f"min {sp.min()}, max {sp.max()}")
        rep.check(bool(df.loc[df["gps_speed_valid"].astype(bool), "gps_speed_m_s"].notna().all()
                       and df.loc[~df["gps_speed_valid"].astype(bool) & df["gps_matched"].astype(bool),
                                  "gps_speed_m_s"].isna().all()),
                  "speed NaN exactly where flagged invalid")
        rep.check(bool((df["gps_speed_interp_m_s"].dropna() >= 0).all()), "interpolated speed >= 0")
        co = df["gps_course_deg"].dropna()
        rep.check(bool(((co >= 0) & (co < 360)).all()), "course in [0, 360)")
        rep.check(bool((df["gps_gap_sec"].dropna() <= cfg.gps_assoc_max_gap_sec + EPS).all()),
                  f"GPS gap <= {cfg.gps_assoc_max_gap_sec}s", f"max {df['gps_gap_sec'].max()}")
        n_inv = int((df["gps_matched"].astype(bool) & ~df["gps_speed_valid"].astype(bool)).sum())
        n_unm = int((~df["gps_matched"].astype(bool)).sum())
        n_empty = int(df["gaze_empty"].astype(bool).sum())
        rep.info(f"- info: rows {len(df)}, rows with invalid speed {n_inv} "
                 f"({df.loc[df['gps_matched'].astype(bool) & ~df['gps_speed_valid'].astype(bool), 'video_id'].nunique()} sequences), "
                 f"rows without GPS {n_unm}, empty gaze rows {n_empty}, "
                 f"max |cam container drift| {max((c.get('max_cam_container_drift_sec') or 0) for c in cks.values()) if cks else 0:.4f}s")
        totals["rows"] += len(df)
        totals["expected"] += sum(expected.values())
        totals["warn_speed"] += n_inv
        totals["gaze_empty"] += n_empty
        rep.info("")

    rep.info("## Storage and protection")
    rep.info("")
    frames_dirs = [p for p in cfg.processed_root.rglob("frames") if p.is_dir() and "_archive" not in str(p)]
    rep.check(not frames_dirs, "no bulk frame folders in data/processed", ", ".join(map(str, frames_dirs)))
    size = sum(dir_size(cfg.split_dir(s)) for s in cfg.splits)
    n_qc = sum(len(list(cfg.qc_overlay_dir(s).glob("*.jpg"))) for s in cfg.splits)
    rep.info(f"- info: data/processed size (excluding archive) {size / 1e6:.1f} MB, QC overlays {n_qc}")
    rep.check(totals["rows"] == totals["expected"], "total rows == expected from manifest",
              f"{totals['rows']} / {totals['expected']}")
    verify_weights(rep)
    rep.info("")
    rep.info(f"**Result: {'ALL CHECKS PASSED' if rep.failures == 0 else f'{rep.failures} CHECK(S) FAILED'}**")

    out = a.out or (cfg.reports_dir / f"phase2_verification_{scope}.md")
    cfg.assert_safe_output(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rep.lines) + "\n", encoding="utf-8")
    print("\n".join(rep.lines))
    print(f"\n-> {out}")
    return 0 if rep.failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
