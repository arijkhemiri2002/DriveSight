"""
Phase 3 pre-flight: environment, model integrity and a speed/precision benchmark on real frames.

  1. Python / torch / CUDA / GPU / ultralytics versions
  2. weights SHA-256 == weights/final/SHA256.txt, checkpoint class order == V2 class list
  3. benchmark on 40 frames of a pilot sequence: ms/frame for several batch sizes, FP16 and FP32
  4. agreement checks that justify the run settings:
       * batch 1 vs batch 8 (must give the same detections)
       * FP16 vs FP32 (share of conf>=0.25 boxes matched with IoU >= 0.9 and same class)
  5. estimated duration of the full run (62,008 frames)

Writes reports/video_dataset/phase3_env_benchmark.md. Read-only on data and weights.
Run:  python -m src.detection.check_detection_env
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np

from src.features.preprocess_config import PreprocessConfig
from src.features.manifest_loader import load_manifest, select_sequences, sequence_paths
from src.features.frame_extractor import SequentialReader, sample_plan
from src.detection.detect_config import DetectConfig
from src.detection.run_detection import check_weights


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def agreement(r1: list, r2: list, conf=0.25, thr=0.9) -> tuple[float, int]:
    matched = total = 0
    for a, b in zip(r1, r2):
        a = a[a[:, 4] >= conf]; b = b[b[:, 4] >= conf]
        total += len(a)
        if len(a) and len(b):
            m = iou_matrix(a, b)
            same = a[:, None, 5] == b[None, :, 5]
            matched += int(((m >= thr) & same).any(axis=1).sum())
    return (matched / total if total else 1.0), total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bdda-root", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--weights-dir", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--frames", type=int, default=40)
    a = ap.parse_args(argv)
    import torch
    import ultralytics
    from src.detection.detector import YoloDetector

    pre = PreprocessConfig(**{k: v for k, v in (("bdda_root", a.bdda_root), ("manifest_csv", a.manifest)) if v})
    dkw = {"weights_dir": a.weights_dir} if a.weights_dir else {}
    lines = ["# Phase 3 environment and benchmark", ""]
    cuda = torch.cuda.is_available()
    lines += [f"- Python {platform.python_version()} on {platform.platform()}",
              f"- torch {torch.__version__} (CUDA build {torch.version.cuda}), CUDA available: {cuda}",
              f"- ultralytics {ultralytics.__version__}"]
    if cuda:
        p = torch.cuda.get_device_properties(0)
        lines.append(f"- GPU: {p.name}, {p.total_memory / 2**30:.1f} GiB")
    else:
        lines.append("- **No CUDA GPU visible to torch: inference would run on the CPU** "
                     "(install a CUDA build of torch, see requirements.txt)")
    cfg = DetectConfig(pre=pre, **dkw)
    sha = check_weights(cfg)
    has_ref = (cfg.weights_dir / "SHA256.txt").exists()
    lines.append(f"- weights {cfg.weights_path.name}: SHA-256 {sha[:16]}... "
                 + ("matches weights/final/SHA256.txt" if has_ref else "(no SHA256.txt to compare with)"))

    man = load_manifest(pre)
    r = select_sequences(man, pilot=True).iloc[0]
    plan = sample_plan(int(r.cam_frames), float(r.cam_fps), int(r.gaze_frames), float(r.gaze_fps), pre.sampling_fps)
    frames = []
    with SequentialReader(sequence_paths(r.split, r.video_id, pre)["camera"]) as rd:
        for p_ in plan[:a.frames]:
            frames.append(rd.read_at(p_.cam_idx))
    lines.append(f"- benchmark frames: {len(frames)} from {r.split}/{r.video_id}")
    lines += ["", "| precision | batch | ms/frame | peak GPU memory (GiB) |", "|---|---|---|---|"]

    results, timings, best = {}, {}, None
    for half in ([True, False] if cuda else [False]):
        det = YoloDetector(DetectConfig(pre=pre, half="true" if half else "false", **dkw))
        det.warmup(frames[0])
        for bs in (1, 4, 8, 16):
            try:
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                out = []
                for i in range(0, len(frames), bs):
                    out += det.predict(frames[i:i + bs])
                if cuda:
                    torch.cuda.synchronize()
                ms = 1000 * (time.perf_counter() - t0) / len(frames)
                mem = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
                results[(half, bs)] = out
                timings[(half, bs)] = ms
                lines.append(f"| {'FP16' if half else 'FP32'} | {bs} | {ms:.1f} | {mem:.2f} |")
                if best is None or ms < best[2]:
                    best = (half, bs, ms)
            except RuntimeError as e:          # CUDA out of memory
                lines.append(f"| {'FP16' if half else 'FP32'} | {bs} | failed: {str(e)[:60]} | |")
                if cuda:
                    torch.cuda.empty_cache()
        del det

    lines.append("")
    # Eligibility: a (precision, batch) setting may be used only if it reproduces the FP32
    # batch-1 reference (>= 99 % of conf>=0.25 boxes matched, IoU >= 0.9, same class).
    # On the MX550, FP16 produced unrelated boxes (0 % agreement), so speed alone must not decide.
    ref = results.get((False, 1))
    eligible = {}
    if ref is not None:
        lines += ["| setting | agreement with FP32 batch 1 | eligible |", "|---|---|---|"]
        for (hf, bs), out in sorted(results.items()):
            ag, n = agreement(ref, out)
            eligible[(hf, bs)] = ag >= 0.99
            lines.append(f"| {'FP16' if hf else 'FP32'} batch {bs} | {ag:.1%} of {n} | {'yes' if ag >= 0.99 else 'NO'} |")
        lines.append("")
        best = None
        for (hf, bs), ok in eligible.items():
            if ok and (hf, bs) in timings and (best is None or timings[(hf, bs)] < best[2]):
                best = (hf, bs, timings[(hf, bs)])
    h0 = True if cuda else False
    if (h0, 1) in results and (h0, 8) in results:
        ag, n = agreement(results[(h0, 1)], results[(h0, 8)], thr=0.99)
        lines.append(f"- batch 1 vs batch 8 ({'FP16' if h0 else 'FP32'}): {ag:.1%} of {n} boxes (conf>=0.25) identical "
                     f"(IoU >= 0.99, same class)")
    if cuda and (True, 8) in results and (False, 8) in results:
        ag, n = agreement(results[(False, 8)], results[(True, 8)])
        lines.append(f"- FP16 vs FP32: {ag:.1%} of {n} FP32 boxes (conf>=0.25) matched by FP16 (IoU >= 0.9, same class)")
    if best:
        est = 62008 * best[2] / 1000 / 60
        lines.append(f"- fastest: {'FP16' if best[0] else 'FP32'} batch {best[1]} = {best[2]:.1f} ms/frame -> "
                     f"inference for 62,008 frames ~ {est:.0f} min (+ video decoding, ~10 min)")
    out = a.out or (pre.reports_dir / "phase3_env_benchmark.md")
    import json
    out.with_suffix(".json").write_text(json.dumps({
        "cuda": cuda, "best_half": best[0] if best else None, "best_batch": best[1] if best else None,
        "best_ms_per_frame": round(best[2], 2) if best else None}, indent=2), encoding="utf-8")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
