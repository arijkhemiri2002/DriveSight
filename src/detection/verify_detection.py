"""
Independent verification of Phase 3 outputs.

  coverage    every sequence in scope has a done checkpoint whose file hashes match, and its
              Phase 2 lineage (phase2_csv_sha256) still matches the Phase 2 CSV on disk
  join        frame_summary keys == Phase 2 aligned_records keys (same rows, same frame index);
              every box key exists in frame_summary; n_det agrees with the box table
  boxes       ids/names consistent with the V2 class list, conf in [floor, 1],
              0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H, unique (video_id, t_ms, det_idx)
  separation  each video in one split, the manifest's
  protection  weights/final SHA-256 unchanged; no bulk frame folders
Also reports detection statistics (per class, per frame, timing) for the documentation.

Run:  python -m src.detection.verify_detection --pilot      |      python -m src.detection.verify_detection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.features.preprocess_config import PreprocessConfig
from src.features.manifest_loader import load_manifest, select_sequences
from src.features.run_preprocessing import sha256_file
from src.features.verify_preprocessing import Report, verify_weights
from src.detection.detect_config import DetectConfig, CLASS_NAMES

EPS = 0.5     # px


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
    cfg = DetectConfig(pre=pre)
    man = load_manifest(pre)
    sel = select_sequences(man, split=a.split, video_ids=a.videos, pilot=a.pilot)
    scope = "pilot" if a.pilot else ("custom" if (a.split or a.videos) else "full")
    W, H = pre.expected_camera_w, pre.expected_camera_h
    rep = Report()
    rep.info(f"# Phase 3 verification - {scope}")
    rep.info("")
    rep.info(f"Sequences in scope: {len(sel)} {sel['split'].value_counts().to_dict()}")
    rep.info("")
    all_boxes, all_frames, cks_all, hashes = [], [], [], set()
    seen: dict[str, str] = {}

    for split in [s for s in pre.splits if s in set(sel["split"])]:
        rep.info(f"## {split}")
        rep.info("")
        vids = set(sel.loc[sel["split"] == split, "video_id"])
        status, bad_hash, bad_lineage = {}, [], []
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
            cks_all.append(ck)
            hashes.add(ck["config_hash"])
            b = cfg.seq_det_dir(split) / f"{v}.parquet"
            f = cfg.seq_det_dir(split) / f"{v}_frames.csv"
            if not (b.exists() and f.exists() and sha256_file(b) == ck["boxes_sha256"]
                    and sha256_file(f) == ck["frames_sha256"]):
                bad_hash.append(v)
            if sha256_file(pre.sequence_records_dir(split) / f"{v}.csv") != ck.get("phase2_csv_sha256"):
                bad_lineage.append(v)
        cnt = pd.Series(status).value_counts().to_dict()
        rep.check(cnt.get("done", 0) == len(vids), "every sequence has a done checkpoint", str(cnt))
        rep.check(not bad_hash, "per-sequence files match checkpoint hashes", ", ".join(bad_hash[:20]))
        rep.check(not bad_lineage, "Phase 2 lineage unchanged (phase2_csv_sha256)", ", ".join(bad_lineage[:20]))

        if not (cfg.detections_path(split).exists() and cfg.frame_summary_path(split).exists()):
            rep.check(False, "merged detections.parquet and frame_summary.csv exist")
            continue
        boxes = pd.read_parquet(cfg.detections_path(split))
        boxes["video_id"] = boxes["video_id"].astype(str)
        frames = pd.read_csv(cfg.frame_summary_path(split), dtype={"video_id": str})
        boxes = boxes[boxes["video_id"].isin(vids)]
        frames = frames[frames["video_id"].isin(vids)]
        p2 = pd.read_csv(pre.aligned_records_path(split), dtype={"video_id": str},
                         usecols=["video_id", "t_ms", "cam_frame_idx"])
        p2 = p2[p2["video_id"].isin(vids)]
        k2 = set(zip(p2["video_id"], p2["t_ms"], p2["cam_frame_idx"]))
        kf = set(zip(frames["video_id"], frames["t_ms"], frames["cam_frame_idx"]))
        rep.check(k2 == kf, "frame_summary rows == Phase 2 rows (video_id, t_ms, cam_frame_idx)",
                  f"{len(kf)} / {len(k2)}; missing {len(k2 - kf)}, extra {len(kf - k2)}")
        rep.check(not frames.duplicated(["video_id", "t_ms"]).any(), "unique frame keys")
        kb = set(zip(boxes["video_id"], boxes["t_ms"]))
        rep.check(kb <= set(zip(frames["video_id"], frames["t_ms"])), "every box belongs to an analysed frame")
        nb = boxes.groupby(["video_id", "t_ms"]).size()
        nf = frames.set_index(["video_id", "t_ms"])["n_det"]
        rep.check(bool((nb.reindex(nf.index, fill_value=0) == nf).all()), "n_det agrees with the box table")
        rep.check(not boxes.duplicated(["video_id", "t_ms", "det_idx"]).any(), "unique (video_id, t_ms, det_idx)")
        ids_ok = bool(boxes["cls_id"].between(0, len(CLASS_NAMES) - 1).all()) and \
            bool((boxes["cls_name"] == boxes["cls_id"].map(dict(enumerate(CLASS_NAMES)))).all())
        rep.check(ids_ok, "class ids and names match the V2 class list")
        rep.check(bool(boxes["conf"].between(cfg.conf_floor - 1e-4, 1.0001).all()),
                  f"confidence in [{cfg.conf_floor}, 1]", f"min {boxes['conf'].min():.3f}")
        geo = ((boxes["x1"] >= -EPS) & (boxes["y1"] >= -EPS) & (boxes["x2"] <= W + EPS) & (boxes["y2"] <= H + EPS)
               & (boxes["x2"] > boxes["x1"]) & (boxes["y2"] > boxes["y1"]))
        bad = boxes.loc[~geo, ["video_id", "t_ms", "det_idx", "cls_name", "conf", "x1", "y1", "x2", "y2"]]
        rep.check(bool(geo.all()), "boxes inside the image with positive size",
                  f"{len(bad)} bad" + ("" if bad.empty else ": " + bad.head(5).to_dict("records").__repr__()))
        vru_ok = bool((boxes["is_vru"] == boxes["cls_name"].isin(["pedestrian", "rider", "motorcycle", "bicycle"])).all())
        rep.check(vru_ok, "is_vru flag consistent with class")
        for v in vids:
            if v in seen and seen[v] != split:
                rep.check(False, "video in one split only", v)
            seen[v] = split
        rep.check(bool((frames["video_id"].map(man.set_index("video_id")["split"]) == split).all()),
                  "split matches manifest")
        b25 = boxes[boxes["conf"] >= 0.25]
        rep.info(f"- info: frames {len(frames)}, boxes {len(boxes)} (conf>=0.25: {len(b25)}), "
                 f"boxes/frame {len(boxes) / max(len(frames), 1):.1f}, frames with a VRU at conf>=0.25: "
                 f"{b25[b25['is_vru']].groupby(['video_id', 't_ms']).ngroups}")
        rep.info("")
        all_boxes.append(boxes)
        all_frames.append(frames)

    rep.info("## Totals")
    rep.info("")
    rep.check(len(hashes) <= 1, "one config hash across all sequences", ", ".join(sorted(hashes)))
    if all_boxes:
        B = pd.concat(all_boxes)
        F = pd.concat(all_frames)
        tab = pd.DataFrame({"conf>=0.05": B["cls_name"].value_counts(),
                            "conf>=0.25": B.loc[B["conf"] >= 0.25, "cls_name"].value_counts(),
                            "conf>=0.50": B.loc[B["conf"] >= 0.50, "cls_name"].value_counts()}).reindex(CLASS_NAMES).fillna(0).astype(int)
        rep.info("Boxes per class (descriptive; the operating threshold is chosen later on validation):")
        rep.info("")
        rep.info("```")
        rep.info(tab.to_string())
        rep.info("```")
        el = sum(c.get("elapsed_sec", 0) for c in cks_all)
        inf = sum(c.get("infer_sec", 0) for c in cks_all)
        rep.info(f"- frames {len(F)}, total {el / 60:.1f} min ({len(F) / max(el, 1e-9):.1f} frames/s end-to-end; "
                 f"inference {inf / 60:.1f} min = {1000 * inf / max(len(F), 1):.1f} ms/frame)")
    frames_dirs = [p for p in pre.processed_root.rglob("frames") if p.is_dir() and "_archive" not in str(p)]
    rep.check(not frames_dirs, "no bulk frame folders")
    verify_weights(rep)
    rep.info("")
    rep.info(f"**Result: {'ALL CHECKS PASSED' if rep.failures == 0 else f'{rep.failures} CHECK(S) FAILED'}**")
    out = a.out or (pre.reports_dir / f"phase3_verification_{scope}.md")
    pre.assert_safe_output(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rep.lines) + "\n", encoding="utf-8")
    print("\n".join(rep.lines))
    print(f"\n-> {out}")
    return 0 if rep.failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
