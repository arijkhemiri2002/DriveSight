"""
Visual check of Phase 4 tracks: draw each track (box, id, trail of its last centres) on the camera frames.

Needs the BDDA camera videos of the chosen sequences (data/BDDA/{split}/camera_videos/{video_id}.mp4).
Frames are decoded with the same SequentialReader and frame indices as Phases 2 and 3, so what is drawn is
exactly what the tracker saw. Nothing is written under data/BDDA.

Output, per sequence, under data/processed/{split}/qc/tracks/:
    {video_id}_tracks.mp4     the sampled frames at 5 fps with the tracks drawn
    {video_id}_sheet.jpg      six evenly spaced frames on one image (for reports)

Run:  python -m src.tracking.draw_tracks --split validation --videos 1744 1802
      python -m src.tracking.draw_tracks --split validation --n 8          (first 8 validation sequences)
"""
from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.features.preprocess_config import PreprocessConfig
from src.features.manifest_loader import load_manifest, select_sequences, sequence_paths
from src.features.frame_extractor import SequentialReader
from src.detection.detect_config import DetectConfig
from src.tracking.track_config import TrackConfig

TRAIL = 10          # frames of trail (2 s at 5 Hz)
VRU = {"pedestrian", "rider", "motorcycle", "bicycle"}


def color(track_id: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((track_id * 0.61803) % 1.0, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_sequence(cfg: TrackConfig, split: str, vid: str) -> tuple[Path, Path] | None:
    tp = cfg.seq_tracks_dir(split) / f"{vid}.parquet"
    fp = cfg.det.seq_det_dir(split) / f"{vid}_frames.csv"
    video = sequence_paths(split, vid, cfg.pre)["camera"]
    if not (tp.exists() and fp.exists() and video.exists()):
        print(f"skip {split}/{vid}: tracks, frames or camera video missing")
        return None
    tr = pd.read_parquet(tp)
    frames = pd.read_csv(fp, usecols=["sample_idx", "t_ms", "cam_frame_idx"]).sort_values("t_ms")
    by_t = {t: g for t, g in tr.groupby("t_ms")}
    out_dir = cfg.pre.split_dir(split) / "qc" / "tracks"
    cfg.pre.assert_safe_output(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4, sheet = out_dir / f"{vid}_tracks.mp4", out_dir / f"{vid}_sheet.jpg"
    W, H = cfg.pre.expected_camera_w, cfg.pre.expected_camera_h
    vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), cfg.sampling_fps, (W, H))
    picks = set(np.linspace(0, len(frames) - 1, 6).round().astype(int).tolist()) if len(frames) else set()
    tiles, trails = [], {}
    with SequentialReader(video) as reader:
        for i, fr in enumerate(frames.itertuples(index=False)):
            img = reader.read_at(int(fr.cam_frame_idx))
            if img is None:
                break
            for b in by_t.get(int(fr.t_ms), tr.iloc[0:0]).itertuples():
                c = color(int(b.track_id))
                trails.setdefault(int(b.track_id), []).append((int((b.x1 + b.x2) / 2), int(b.y2)))
                pts = trails[int(b.track_id)][-TRAIL:]
                for p, q in zip(pts[:-1], pts[1:]):
                    cv2.line(img, p, q, c, 2, cv2.LINE_AA)
                cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), c, 3 if b.cls_name in VRU else 2)
                label = f"#{int(b.track_id)} {b.cls_name} {b.conf:.2f}"
                cv2.putText(img, label, (int(b.x1), max(int(b.y1) - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
            cv2.rectangle(img, (0, 0), (W, 28), (0, 0, 0), -1)
            cv2.putText(img, f"{split}/{vid}  t={fr.t_ms / 1000:.1f}s  ByteTrack {cfg.params()['track_buffer_frames']}-frame buffer",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            vw.write(img)
            if i in picks:
                tiles.append(cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA))
    vw.release()
    if tiles:
        while len(tiles) < 6:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[r * 3:(r + 1) * 3]) for r in range(2)])
        cv2.imwrite(str(sheet), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    print(f"{split}/{vid}: {mp4.name}, {sheet.name}")
    return mp4, sheet


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="validation", choices=["training", "validation", "test"])
    ap.add_argument("--videos", nargs="+")
    ap.add_argument("--n", type=int, default=5, help="number of sequences when --videos is not given")
    ap.add_argument("--processed-root", type=Path)
    ap.add_argument("--bdda-root", type=Path)
    ap.add_argument("--manifest", type=Path)
    a = ap.parse_args(argv)
    pkw = {k: v for k, v in (("processed_root", a.processed_root), ("bdda_root", a.bdda_root),
                             ("manifest_csv", a.manifest)) if v}
    cfg = TrackConfig(det=DetectConfig(pre=PreprocessConfig(**pkw)))
    sel = select_sequences(load_manifest(cfg.pre), split=a.split, video_ids=a.videos, limit=None if a.videos else a.n)
    done = [draw_sequence(cfg, a.split, str(v)) for v in sel["video_id"]]
    return 0 if any(done) else 1


if __name__ == "__main__":
    raise SystemExit(main())
