"""
Independent verification of Phase 4 outputs (does not import the tracker).

  coverage    every sequence in scope has a done checkpoint whose file hashes match, and its Phase 3
              lineage (phase3_boxes_sha256 / phase3_frames_sha256) still matches the Phase 3 files on disk
  link        every tracked box exists in Phase 3 (video_id, t_ms, det_idx) with the same class, confidence
              and coordinates; every tracked frame is a Phase 3 frame
  tracks      no track has two boxes at the same instant; no box belongs to two tracks; track ids are positive;
              track_frame / is_birth consistent with time order; each track lives in one sequence
  summary     one summary row per track; n_boxes, first/last time, majority class agree with the box table
  separation  each video in one split, the manifest's
  protection  Phase 2 and Phase 3 sequence files unchanged (their own checkpoint hashes); weights unchanged
Also reports descriptive track statistics for the documentation.
Run:  python -m src.tracking.verify_tracking --pilot   |   python -m src.tracking.verify_tracking
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.preprocess_config import PreprocessConfig
from src.features.manifest_loader import load_manifest, select_sequences
from src.features.run_preprocessing import sha256_file
from src.features.verify_preprocessing import Report, verify_weights
from src.detection.detect_config import DetectConfig
from src.tracking.track_config import TrackConfig

ATTRS = ["cls_id", "cls_name", "conf", "x1", "y1", "x2", "y2"]


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
    pkw = {k: v for k, v in (("processed_root", a.processed_root), ("bdda_root", a.bdda_root),
                             ("manifest_csv", a.manifest)) if v}
    pre = PreprocessConfig(**pkw)
    cfg = TrackConfig(det=DetectConfig(pre=pre))
    det = cfg.det
    man = load_manifest(pre)
    sel = select_sequences(man, split=a.split, video_ids=a.videos, pilot=a.pilot)
    scope = "pilot" if a.pilot else ("custom" if (a.split or a.videos) else "full")
    rep = Report()
    rep.info(f"# Phase 4 verification - {scope}")
    rep.info("")
    rep.info(f"Sequences in scope: {len(sel)} {sel['split'].value_counts().to_dict()}")
    rep.info("")
    hashes, p3hashes, params, all_sum = set(), set(), set(), []
    seen: dict[str, str] = {}

    for split in [s for s in pre.splits if s in set(sel["split"])]:
        rep.info(f"## {split}")
        rep.info("")
        vids = set(sel.loc[sel["split"] == split, "video_id"])
        status, bad_hash, bad_lineage, bad_upstream = {}, [], [], []
        for v in sorted(vids):
            p = cfg.checkpoints_dir(split) / f"{v}.json"
            if not p.exists():
                status[v] = "missing"
                continue
            ck = json.loads(p.read_text(encoding="utf-8"))
            status[v] = ck.get("status")
            if ck.get("status") != "done":
                rep.info(f"  - {v}: {ck.get('status')} - {'; '.join(ck.get('problems', [])) or ck.get('error', '')}")
                continue
            hashes.add(ck["config_hash"])
            p3hashes.add(ck.get("phase3_config_hash"))
            t = cfg.seq_tracks_dir(split) / f"{v}.parquet"
            s = cfg.seq_tracks_dir(split) / f"{v}_summary.csv"
            if not (t.exists() and s.exists() and sha256_file(t) == ck["tracks_sha256"]
                    and sha256_file(s) == ck["summary_sha256"]):
                bad_hash.append(v)
            b3 = det.seq_det_dir(split) / f"{v}.parquet"
            f3 = det.seq_det_dir(split) / f"{v}_frames.csv"
            if not (b3.exists() and sha256_file(b3) == ck.get("phase3_boxes_sha256")
                    and sha256_file(f3) == ck.get("phase3_frames_sha256")):
                bad_lineage.append(v)
            # upstream untouched: Phase 3 and Phase 2 files still match their OWN checkpoints
            c3 = det.checkpoints_dir(split) / f"{v}.json"
            c2 = pre.checkpoints_dir(split) / f"{v}.json"
            try:
                k3 = json.loads(c3.read_text(encoding="utf-8"))
                ok3 = sha256_file(b3) == k3.get("boxes_sha256") and sha256_file(f3) == k3.get("frames_sha256")
                k2 = json.loads(c2.read_text(encoding="utf-8")) if c2.exists() else None
                p2csv = pre.sequence_records_dir(split) / f"{v}.csv"
                ok2 = True if k2 is None else (sha256_file(p2csv) == k3.get("phase2_csv_sha256"))
            except (OSError, json.JSONDecodeError):
                ok3 = ok2 = False
            if not (ok3 and ok2):
                bad_upstream.append(v)
        cnt = pd.Series(status, dtype=object).value_counts().to_dict()
        rep.check(cnt.get("done", 0) == len(vids), "every sequence has a done checkpoint", str(cnt))
        rep.check(not bad_hash, "per-sequence files match checkpoint hashes", ", ".join(bad_hash[:20]))
        rep.check(not bad_lineage, "Phase 3 lineage unchanged (phase3_*_sha256)", ", ".join(bad_lineage[:20]))
        rep.check(not bad_upstream, "Phase 2 and Phase 3 sequence files unchanged (their own hashes)",
                  ", ".join(bad_upstream[:20]))

        if not (cfg.tracks_path(split).exists() and cfg.summary_path(split).exists()):
            rep.check(False, "merged tracks.parquet and track_summary.csv exist")
            continue
        sj = cfg.tracks_dir(split) / "_schema.json"
        if sj.exists():
            params.add(json.dumps(json.loads(sj.read_text(encoding="utf-8")).get("params"), sort_keys=True))
        tr = pd.read_parquet(cfg.tracks_path(split))
        tr["video_id"] = tr["video_id"].astype(str)
        sm = pd.read_csv(cfg.summary_path(split), dtype={"video_id": str})
        tr, sm = tr[tr["video_id"].isin(vids)], sm[sm["video_id"].isin(vids)]
        b3 = pd.read_parquet(det.detections_path(split))
        b3["video_id"] = b3["video_id"].astype(str)
        b3 = b3[b3["video_id"].isin(vids)]
        f3 = pd.read_csv(det.frame_summary_path(split), dtype={"video_id": str}, usecols=["video_id", "t_ms"])

        j = tr.merge(b3[["video_id", "t_ms", "det_idx"] + ATTRS], on=["video_id", "t_ms", "det_idx"],
                     how="left", suffixes=("", "_p3"), indicator=True)
        rep.check(bool((j["_merge"] == "both").all()), "every tracked box exists in Phase 3 (video_id, t_ms, det_idx)",
                  f"{int((j['_merge'] != 'both').sum())} missing")
        same = ((j["cls_id"] == j["cls_id_p3"]) & (j["cls_name"] == j["cls_name_p3"])
                & np.isclose(j["conf"], j["conf_p3"]) & np.isclose(j["x1"], j["x1_p3"]) & np.isclose(j["y1"], j["y1_p3"])
                & np.isclose(j["x2"], j["x2_p3"]) & np.isclose(j["y2"], j["y2_p3"]))
        rep.check(bool(same.all()), "tracked boxes carry the Phase 3 class, confidence and coordinates",
                  f"{int((~same).sum())} differ")
        rep.check(set(zip(tr["video_id"], tr["t_ms"])) <= set(zip(f3["video_id"], f3["t_ms"])),
                  "every tracked frame is a Phase 3 frame")
        rep.check(not tr.duplicated(["video_id", "track_id", "t_ms"]).any(), "no track has two boxes at one instant")
        rep.check(not tr.duplicated(["video_id", "t_ms", "det_idx"]).any(), "no box belongs to two tracks")
        rep.check(bool((tr["track_id"] > 0).all()), "track ids are positive")
        o = tr.sort_values(["video_id", "track_id", "t_ms"])
        tf_ok = bool((o.groupby(["video_id", "track_id"]).cumcount().to_numpy() == o["track_frame"].to_numpy()).all())
        first = o.groupby(["video_id", "track_id"])["t_ms"].transform("min")
        birth_ok = bool(((o["t_ms"] == first) == o["is_birth"].astype(bool)).all())
        rep.check(tf_ok and birth_ok, "track_frame and is_birth follow time order")
        g = tr.groupby(["video_id", "track_id"])
        agg = g.agg(n=("t_ms", "size"), first=("t_ms", "min"), last=("t_ms", "max")).reset_index()
        maj = (tr.groupby(["video_id", "track_id", "cls_name"]).size().rename("k").reset_index()
               .merge(tr.groupby(["video_id", "track_id", "cls_name"])["conf"].mean().rename("m").reset_index())
               .sort_values(["k", "m"], ascending=False).drop_duplicates(["video_id", "track_id"]))
        agg = agg.merge(maj[["video_id", "track_id", "cls_name"]], on=["video_id", "track_id"])
        m = agg.merge(sm, on=["video_id", "track_id"], how="outer", indicator=True, suffixes=("", "_s"))
        rep.check(bool((m["_merge"] == "both").all()) and len(sm) == len(agg), "one summary row per track",
                  f"{len(sm)} summary rows, {len(agg)} tracks")
        both = m[m["_merge"] == "both"]
        cons = ((both["n"] == both["n_boxes"]) & (both["first"] == both["first_t_ms"]) & (both["last"] == both["last_t_ms"])
                & (both["cls_name"] == both["cls_name_s"]))
        rep.check(bool(cons.all()), "summary agrees with the box table (n_boxes, first/last time, majority class)",
                  f"{int((~cons).sum())} differ")
        for v in vids:
            if v in seen and seen[v] != split:
                rep.check(False, "video in one split only", v)
            seen[v] = split
        rep.check(bool((tr["video_id"].map(man.set_index("video_id")["split"]) == split).all())
                  and bool((tr["split"].astype(str) == split).all()), "split matches manifest")
        hi = b3[b3["conf"] >= 0.5]
        cov = len(set(zip(hi["video_id"], hi["t_ms"], hi["det_idx"])) & set(zip(tr["video_id"], tr["t_ms"], tr["det_idx"])))
        rep.info(f"- info: tracks {len(sm)} (VRU {int(sm['is_vru'].sum())}), tracked boxes {len(tr)} of {len(b3)} "
                 f"stored; boxes with conf >= 0.5 in a track: {cov}/{len(hi)} ({100 * cov / max(len(hi), 1):.1f} %); "
                 f"median track length {sm['n_boxes'].median():.0f} frames")
        rep.info("")
        all_sum.append(sm)

    rep.info("## Totals")
    rep.info("")
    rep.check(len(hashes) <= 1, "one config hash across all sequences", ", ".join(sorted(hashes)))
    rep.check(len(p3hashes) <= 1, "one Phase 3 config hash across all sequences", ", ".join(sorted(map(str, p3hashes))))
    rep.check(len(params) <= 1, "one tracker setting across splits")
    if params:
        rep.info(f"- tracker settings: {next(iter(params))}")
    if all_sum:
        S = pd.concat(all_sum)
        n = S["n_boxes"]
        rep.info(f"- tracks {len(S)}; length (frames at 5 Hz): median {n.median():.0f}, mean {n.mean():.1f}, "
                 f"share with 1-2 frames {100 * (n <= 2).mean():.1f} %; tracks with a class change "
                 f"{100 * (S['n_classes'] > 1).mean():.1f} %")
    verify_weights(rep)
    rep.info("")
    rep.info(f"**Result: {'ALL CHECKS PASSED' if rep.failures == 0 else f'{rep.failures} CHECK(S) FAILED'}**")
    out = a.out or (pre.reports_dir / f"phase4_verification_{scope}.md")
    pre.assert_safe_output(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rep.lines) + "\n", encoding="utf-8")
    print("\n".join(rep.lines))
    print(f"\n-> {out}")
    return 0 if rep.failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
