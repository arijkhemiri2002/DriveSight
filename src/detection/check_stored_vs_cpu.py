"""
Do the stored Phase 3 detections (MX550 GPU, FP32, batch 4) match a CPU FP32 batch-1 reference?

Why: on the MX550, FP16 inference produced unrelated boxes (0 % agreement with FP32 in the
benchmark). The full run used FP32, and the QC images look right, but a numeric check on
the actual stored output is the evidence that the run is not affected.

Method: for a few sequences from each split, every 5th sampled frame is decoded again and run on
the CPU (FP32, batch 1, same imgsz / NMS / floor). Boxes are compared with a hysteresis that
avoids counting threshold flips as errors: a reference box with conf >= 0.30 must be found
among the stored boxes with conf >= 0.20 (IoU >= 0.9, same class), and the other way round.
Pass: >= 98 % in both directions.

Writes reports/video_dataset/phase3_gpu_vs_cpu_check.md. Read-only.
Run:  python -m src.detection.check_stored_vs_cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.preprocess_config import PreprocessConfig
from src.features.manifest_loader import load_manifest, sequence_paths
from src.features.frame_extractor import SequentialReader
from src.detection.detect_config import DetectConfig
from src.detection.check_detection_env import iou_matrix

DEFAULT = ["test:6", "test:25", "training:2", "training:4", "validation:1737", "validation:1744"]


def covered(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each box in a, is there a box in b with IoU >= 0.9 and the same class?"""
    if len(a) == 0:
        return np.zeros(0, bool)
    if len(b) == 0:
        return np.zeros(len(a), bool)
    return ((iou_matrix(a, b) >= 0.9) & (a[:, None, 5] == b[None, :, 5])).any(axis=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", nargs="+", default=DEFAULT, help="split:video_id")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--processed-root", type=Path)
    ap.add_argument("--bdda-root", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--weights-dir", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    pre = PreprocessConfig(**{k: v for k, v in (("processed_root", a.processed_root), ("bdda_root", a.bdda_root),
                                                 ("manifest_csv", a.manifest)) if v})
    dkw = {"weights_dir": a.weights_dir} if a.weights_dir else {}
    cfg = DetectConfig(pre=pre, device="cpu", half="false", **dkw)
    from src.detection.detector import YoloDetector
    det = YoloDetector(cfg)
    load_manifest(pre)

    rows, n_ref = [], 0
    for item in a.seqs:
        split, vid = item.split(":")
        stored = pd.read_parquet(cfg.seq_det_dir(split) / f"{vid}.parquet")
        grid = pd.read_csv(pre.sequence_records_dir(split) / f"{vid}.csv", usecols=["sample_idx", "t_ms", "cam_frame_idx"])
        grid = grid.iloc[::a.every]
        with SequentialReader(sequence_paths(split, vid, pre)["camera"]) as rd:
            for g in grid.itertuples():
                f = rd.read_at(int(g.cam_frame_idx))
                ref = det.predict([f])[0]
                s = stored[stored["t_ms"] == g.t_ms]
                st = s[["x1", "y1", "x2", "y2", "conf", "cls_id"]].to_numpy(np.float32)
                r_hi, r_lo = ref[ref[:, 4] >= 0.30], ref[ref[:, 4] >= 0.20]
                s_hi, s_lo = st[st[:, 4] >= 0.30], st[st[:, 4] >= 0.20]
                rows.append({"seq": item, "t_ms": g.t_ms,
                             "ref_boxes": len(r_hi), "ref_found_in_stored": int(covered(r_hi, s_lo).sum()),
                             "stored_boxes": len(s_hi), "stored_found_in_ref": int(covered(s_hi, r_lo).sum())})
                n_ref += 1
    df = pd.DataFrame(rows)
    nr, ns = df["ref_boxes"].sum(), df["stored_boxes"].sum()
    rr = df["ref_found_in_stored"].sum() / nr if nr else 1.0
    sr = df["stored_found_in_ref"].sum() / ns if ns else 1.0
    ok = rr >= 0.98 and sr >= 0.98 and (nr + ns) > 0
    per = df.groupby("seq")[["ref_boxes", "ref_found_in_stored", "stored_boxes", "stored_found_in_ref"]].sum()
    lines = ["# Stored GPU detections vs CPU FP32 reference", "",
             f"Frames re-run on CPU: {n_ref} ({len(a.seqs)} sequences, every {a.every}th sample)", "",
             f"- reference boxes (conf>=0.30) found in stored output (conf>=0.20): {rr:.2%} of {int(df['ref_boxes'].sum())}",
             f"- stored boxes (conf>=0.30) found in reference (conf>=0.20): {sr:.2%} of {int(df['stored_boxes'].sum())}",
             "", f"**Verdict: {'PASS - the stored detections reproduce the CPU FP32 reference' if ok else 'FAIL - stored detections differ from the reference'}**",
             "", "```", per.to_string(), "```"]
    out = a.out or (pre.reports_dir / "phase3_gpu_vs_cpu_check.md")
    pre.assert_safe_output(out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
