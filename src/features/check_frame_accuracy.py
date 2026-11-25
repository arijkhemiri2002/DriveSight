"""
Step 4 of Phase 2: does seek-based reading (used by the pilot) return the same frame as
sequential decoding (used by the new pipeline)?

For each sampled index the camera and gaze frames are read both ways and compared
pixel-for-pixel in memory. If they differ, the script searches the neighbouring frames
(idx-3 .. idx+3) to report the offset the seek landed on.

  * all identical -> the pilot's frame selection was right; only the end-of-video bug
    and the GPS issues affected it.
  * mismatches     -> seeking was unreliable; the sequential pipeline fixes it, and the
    offsets document by how much the pilot was wrong.

Writes reports/video_dataset/frame_accuracy_check.md (new file). Reads videos only.
Run:  python -m src.features.check_frame_accuracy            (extended pilot sequences)
      python -m src.features.check_frame_accuracy --videos 2 1744
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.features.preprocess_config import PreprocessConfig
from src.features.manifest_loader import load_manifest, select_sequences, sequence_paths
from src.features.frame_extractor import SequentialReader, sample_plan, seek_read


def _table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)          # needs `tabulate`
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def find_offset(path: Path, idx: int, frame: np.ndarray, radius: int = 3):
    with SequentialReader(path) as r:
        for j in range(max(0, idx - radius), idx + radius + 1):
            f = r.read_at(j)
            if f is not None and f.shape == frame.shape and np.array_equal(f, frame):
                return j - idx
    return None


def compare(path: Path, indices: list[int]) -> dict:
    cap = cv2.VideoCapture(str(path))
    res = {"n": 0, "identical": 0, "seek_failed": 0, "max_abs_diff": 0, "offsets": []}
    mismatches = []
    with SequentialReader(path) as r:
        for idx in indices:
            a = r.read_at(idx)
            b = seek_read(cap, idx)
            res["n"] += 1
            if b is None or a is None:
                res["seek_failed"] += int(b is None)
                continue
            if np.array_equal(a, b):
                res["identical"] += 1
            else:
                res["max_abs_diff"] = max(res["max_abs_diff"],
                                          int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()))
                mismatches.append((idx, b.copy()))
    cap.release()
    for idx, b in mismatches[:10]:                      # offsets for the first 10 mismatches
        res["offsets"].append((idx, find_offset(path, idx, b)))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+")
    ap.add_argument("--split")
    ap.add_argument("--bdda-root", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    kw = {k: v for k, v in (("bdda_root", a.bdda_root), ("manifest_csv", a.manifest)) if v}
    cfg = PreprocessConfig(**kw)
    m = load_manifest(cfg)
    sel = select_sequences(m, split=a.split, video_ids=a.videos, pilot=not a.videos)

    rows = []
    for r in sel.itertuples():
        plan = sample_plan(int(r.cam_frames), float(r.cam_fps), int(r.gaze_frames), float(r.gaze_fps),
                           cfg.sampling_fps)
        paths = sequence_paths(r.split, r.video_id, cfg)
        for mod, key in (("camera", "cam_idx"), ("gaze", "gaze_idx")):
            res = compare(paths[mod], [getattr(p, key) for p in plan])
            rows.append({"split": r.split, "video_id": r.video_id, "modality": mod,
                         "fps": r.cam_fps if mod == "camera" else r.gaze_fps,
                         "frames_compared": res["n"], "identical": res["identical"],
                         "seek_failed": res["seek_failed"], "max_abs_diff": res["max_abs_diff"],
                         "offsets_idx:off": " ".join(f"{i}:{o}" for i, o in res["offsets"])})
            print(rows[-1])
    df = pd.DataFrame(rows)
    ok = bool((df["identical"] == df["frames_compared"]).all())
    lines = ["# Frame accuracy check: seek (pilot) vs sequential decoding (Phase 2)", "",
             f"Sequences: {sel['video_id'].nunique()} | comparisons: {int(df['frames_compared'].sum())} | "
             f"identical: {int(df['identical'].sum())}", "",
             f"**Verdict: {'seek and sequential decoding agree on every sampled frame' if ok else 'MISMATCHES FOUND: seeking was not frame-accurate (see offsets)'}**",
             "", _table(df)]
    out = a.out or (cfg.reports_dir / "frame_accuracy_check.md")
    cfg.assert_safe_output(out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:5]), f"\n-> {out}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
