"""
One-off repair of Phase 3 outputs: remove boxes that are degenerate after clipping
(x2 <= x1 or y2 <= y1), which the first full run stored (1 box in training).

Why a repair and not a re-run: the fix in box_features.box_table (drop degenerate boxes before
ordering) is the only code change, and its effect on stored outputs is fully determined. Removing
the box, renumbering det_idx in the existing confidence order and recomputing the frame summary
gives exactly the files the fixed code would produce, without a second 9-hour inference run.
The config hash is therefore kept, and every repaired checkpoint records what was removed.

For each affected sequence: rewrite {video_id}.parquet and {video_id}_frames.csv atomically,
update the checkpoint (hashes, counts, "repairs" entry), then rebuild the merged files of the
split. Writes reports/video_dataset/phase3_repair_degenerate_boxes.md.

Run:  python -m src.detection.repair_degenerate_boxes            (add --dry-run to only list)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.features.preprocess_config import PreprocessConfig
from src.features.run_preprocessing import sha256_file, atomic_write_csv, atomic_write_json
from src.detection.detect_config import DetectConfig, CLASS_NAMES
from src.detection.run_detection import atomic_write_parquet, typed


def repair_sequence(cfg: DetectConfig, split: str, vid: str, ck: dict, dry: bool) -> list[dict]:
    bpath = cfg.seq_det_dir(split) / f"{vid}.parquet"
    fpath = cfg.seq_det_dir(split) / f"{vid}_frames.csv"
    b = pd.read_parquet(bpath)
    bad = (b["x2"] <= b["x1"]) | (b["y2"] <= b["y1"])
    if not bad.any():
        return []
    removed = b.loc[bad, ["t_ms", "det_idx", "cls_name", "conf", "x1", "y1", "x2", "y2"]].to_dict("records")
    removed = [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in removed]
    if dry:
        return removed
    b = b.loc[~bad].sort_values(["sample_idx", "det_idx"], kind="stable").copy()
    b["det_idx"] = b.groupby("sample_idx").cumcount()
    b = typed(b.reset_index(drop=True))
    f = pd.read_csv(fpath, dtype={"video_id": str})
    g = b.groupby("t_ms")
    vru = b[b["is_vru"]].groupby("t_ms")
    f["n_det"] = f["t_ms"].map(g.size()).fillna(0).astype(int)
    f["n_vru"] = f["t_ms"].map(vru.size()).fillna(0).astype(int)
    f["max_conf"] = f["t_ms"].map(g["conf"].max()).astype(float)
    f["max_conf_vru"] = f["t_ms"].map(vru["conf"].max()).astype(float)
    atomic_write_parquet(b, bpath, cfg)
    atomic_write_csv(f, fpath, cfg.pre)
    by_cls = b["cls_name"].value_counts().to_dict()
    ck.update(boxes_sha256=sha256_file(bpath), frames_sha256=sha256_file(fpath), n_det=len(b),
              n_det_by_class={c: int(by_cls.get(c, 0)) for c in CLASS_NAMES},
              n_det_conf25=int((b["conf"] >= 0.25).sum()), n_vru=int(b["is_vru"].sum()))
    ck.setdefault("repairs", []).append({"date": datetime.now().isoformat(timespec="seconds"),
                                         "what": "removed boxes degenerate after clipping (x2<=x1 or y2<=y1)",
                                         "removed": removed})
    atomic_write_json(ck, cfg.checkpoints_dir(split) / f"{vid}.json", cfg.pre)
    return removed


def rebuild_merge(cfg: DetectConfig, split: str) -> dict:
    schema_p = cfg.det_dir(split) / "_schema.json"
    schema = json.loads(schema_p.read_text(encoding="utf-8"))
    h = schema["config_hash"]
    vids, boxes, frames = [], [], []
    for p in sorted(cfg.checkpoints_dir(split).glob("*.json")):
        ck = json.loads(p.read_text(encoding="utf-8"))
        if ck.get("status") == "done" and ck.get("config_hash") == h:
            v = str(ck["video_id"])
            vids.append(v)
            boxes.append(pd.read_parquet(cfg.seq_det_dir(split) / f"{v}.parquet"))
            frames.append(pd.read_csv(cfg.seq_det_dir(split) / f"{v}_frames.csv", dtype={"video_id": str}))
    key = lambda s: pd.to_numeric(s, errors="coerce")
    B = pd.concat(boxes, ignore_index=True)
    F = pd.concat(frames, ignore_index=True)
    B = B.sort_values(["video_id", "t_ms", "det_idx"], key=lambda s: key(s) if s.name == "video_id" else s).reset_index(drop=True)
    F = F.sort_values(["video_id", "t_ms"], key=lambda s: key(s) if s.name == "video_id" else s).reset_index(drop=True)
    assert not F.duplicated(["video_id", "t_ms"]).any()
    assert not B.duplicated(["video_id", "t_ms", "det_idx"]).any()
    atomic_write_parquet(typed(B), cfg.detections_path(split), cfg)
    atomic_write_csv(F, cfg.frame_summary_path(split), cfg.pre)
    schema.update(boxes=len(B), frames=len(F), merged_sequences=len(vids),
                  rebuilt_after_repair=datetime.now().isoformat(timespec="seconds"))
    atomic_write_json(schema, schema_p, cfg.pre)
    return {"split": split, "sequences": len(vids), "boxes": len(B), "frames": len(F)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--processed-root", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    pre = PreprocessConfig(**({"processed_root": a.processed_root} if a.processed_root else {}))
    cfg = DetectConfig(pre=pre)
    lines = ["# Phase 3 repair: degenerate boxes", "", f"Mode: {'dry run' if a.dry_run else 'repair'}", ""]
    total, touched = 0, set()
    for split in pre.splits:
        for p in sorted(cfg.checkpoints_dir(split).glob("*.json")):
            ck = json.loads(p.read_text(encoding="utf-8"))
            if ck.get("status") != "done":
                continue
            removed = repair_sequence(cfg, split, str(ck["video_id"]), ck, a.dry_run)
            if removed:
                total += len(removed)
                touched.add(split)
                lines.append(f"- {split}/{ck['video_id']}: {len(removed)} box(es) {removed}")
    lines.append("")
    lines.append(f"Total degenerate boxes: {total}")
    if not a.dry_run:
        for s in sorted(touched):
            lines.append(f"- merged files rebuilt: {rebuild_merge(cfg, s)}")
    out = a.out or (pre.reports_dir / "phase3_repair_degenerate_boxes.md")
    if not a.dry_run:
        pre.assert_safe_output(out)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
