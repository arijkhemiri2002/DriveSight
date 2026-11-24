"""
Orchestrator for DriveSight video preprocessing.

Runs the full pipeline on selected sequences:
  1. extract camera frames at sampling FPS (timestamp-aligned)
  2. extract + crop + resize gaze frames at sampling FPS
  3. parse GPS, derive fields, align to target timestamps
  4. write aligned_records.csv per split
  5. generate QC overlays

Supports PILOT_MODE (small subset) and full-run mode.
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from preprocess_config import PreprocessConfig
from manifest_loader import load_manifest, select_sequences, sequence_paths
from frame_extractor import (extract_frame_from_capture, save_frame,
                             target_timestamps, frame_alignment_ok,
                             ExtractedFrame)
from gaze_crop import (process_gaze_frame, verify_crop_on_sequence,
                       save_qc_triptych, overlay_gaze_on_camera,
                       gaze_to_heatmap)
from gps_aligner import build_gps_records, nearest_gps, GpsRecord


@dataclass
class SequenceResult:
    split: str
    video_id: str
    n_camera_frames: int
    n_gaze_frames: int
    n_aligned_records: int
    n_gps_missing: int
    max_cam_residual_sec: float
    max_gaze_residual_sec: float
    max_gps_gap_sec: float
    camera_dim_ok: bool
    gaze_dim_ok: bool
    problems: list[str]


class VideoPreprocessor:
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self.results: list[SequenceResult] = []
        self.all_records: list[dict] = []
        self.bytes_written = 0

    # ------------------------------------------------------------------
    def run(self) -> None:
        cfg = self.config
        df = load_manifest(cfg)
        selected = select_sequences(df, cfg)
        print(f"Sequences selected for processing: {len(selected)}")
        print(f"Splits: {selected['split'].value_counts().to_dict()}")
        print(f"Sampling FPS: {cfg.sampling_fps}")

        # Storage estimate before extraction
        self._print_storage_estimate(selected, cfg)

        for split in cfg.splits:
            split_dir = cfg.split_dir(split)
            (split_dir / "frames" / "camera").mkdir(parents=True, exist_ok=True)
            (split_dir / "frames" / "gaze").mkdir(parents=True, exist_ok=True)
            (split_dir / "metadata" / "qc" / "overlays").mkdir(parents=True, exist_ok=True)
            cfg.metadata_dir(split).mkdir(parents=True, exist_ok=True)

        # process per split
        for split in cfg.splits:
            split_df = selected[selected["split"] == split]
            self._process_split(split, split_df)

        # write aligned_records per split
        for split in cfg.splits:
            split_records = [r for r in self.all_records if r["split"] == split]
            if split_records:
                rec_df = pd.DataFrame(split_records)
                rec_path = cfg.aligned_records_path(split)
                rec_df.to_csv(rec_path, index=False)
                print(f"  {split}: {len(rec_df)} aligned records -> {rec_path}")

        self._write_pilot_summary()

    # ------------------------------------------------------------------
    def _process_split(self, split: str, split_df: pd.DataFrame) -> None:
        cfg = self.config
        print(f"\n=== Processing split: {split} ({len(split_df)} sequences) ===")

        for _, row in split_df.iterrows():
            video_id = str(row["video_id"])
            paths = sequence_paths(split, video_id, cfg)
            res = self._process_sequence(split, video_id, paths, row)
            self.results.append(res)
            status = "OK" if not res.problems else "PROBLEMS"
            print(f"  {video_id}: {res.n_aligned_records} records "
                  f"(cam {res.n_camera_frames}, gaze {res.n_gaze_frames}, "
                  f"gps_missing {res.n_gps_missing}) [{status}]")
            if res.problems:
                for p in res.problems:
                    print(f"    - {p}")

    # ------------------------------------------------------------------
    def _process_sequence(self, split: str, video_id: str,
                          paths: dict[str, Path],
                          manifest_row: pd.Series) -> SequenceResult:
        cfg = self.config
        problems: list[str] = []

        cam_duration = float(manifest_row["cam_duration"])
        gaze_duration = float(manifest_row["gaze_duration"])
        cam_fps = float(manifest_row["cam_fps"])
        gaze_fps = float(manifest_row["gaze_fps"])

        duration = min(cam_duration, gaze_duration)
        targets = target_timestamps(duration, cfg.sampling_fps)

        # GPS
        gps_align = build_gps_records(paths["gps"])
        gps_align.split = split
        gps_align.video_id = video_id

        # Verify gaze crop on first sequence of each split (pilot QC)
        # (full visual verification done separately before global apply)

        cam_dir = cfg.frames_dir(split, "camera") / video_id
        gaze_dir = cfg.frames_dir(split, "gaze") / video_id

        cam_cap = cv2.VideoCapture(str(paths["camera"]))
        gaze_cap = cv2.VideoCapture(str(paths["gaze"]))
        if not cam_cap.isOpened() or not gaze_cap.isOpened():
            if cam_cap.isOpened():
                cam_cap.release()
            if gaze_cap.isOpened():
                gaze_cap.release()
            problems.append("could not open camera or gaze video")
            return SequenceResult(
                split=split, video_id=video_id,
                n_camera_frames=0, n_gaze_frames=0,
                n_aligned_records=0, n_gps_missing=len(targets),
                max_cam_residual_sec=0.0, max_gaze_residual_sec=0.0,
                max_gps_gap_sec=0.0,
                camera_dim_ok=False, gaze_dim_ok=False,
                problems=problems,
            )

        cam_extracted: list[ExtractedFrame] = []
        gaze_extracted: list[ExtractedFrame] = []
        cam_paths: list[Optional[Path]] = []
        gaze_paths: list[Optional[Path]] = []

        for t in targets:
            # --- camera ---
            cam_name = f"{int(round(t * 1000)):06d}.jpg"
            cam_out = cam_dir / cam_name
            cam_frame, cam_f = extract_frame_from_capture(
                cam_cap, "camera", t, cam_fps, cam_out)
            cam_f.video_id = video_id
            cam_f.split = split
            if cam_f.ok and cam_frame is not None:
                saved = save_frame(cam_frame, cam_out, cfg.jpeg_quality)
                if saved:
                    self.bytes_written += cam_out.stat().st_size
                cam_extracted.append(cam_f)
                cam_paths.append(cam_out)
            else:
                problems.append(f"camera frame failed at t={t}s")
                cam_extracted.append(cam_f)
                cam_paths.append(None)

            # --- gaze ---
            gaze_out = gaze_dir / cam_name
            gaze_frame, gaze_f = extract_frame_from_capture(
                gaze_cap, "gaze", t, gaze_fps, gaze_out)
            gaze_f.video_id = video_id
            gaze_f.split = split
            if gaze_f.ok and gaze_frame is not None:
                gaze_processed = process_gaze_frame(gaze_frame, cfg)
                saved = save_frame(
                    cv2.cvtColor(gaze_processed, cv2.COLOR_GRAY2BGR),
                    gaze_out, cfg.jpeg_quality)
                if saved:
                    self.bytes_written += gaze_out.stat().st_size
                gaze_extracted.append(gaze_f)
                gaze_paths.append(gaze_out)
            else:
                problems.append(f"gaze frame failed at t={t}s")
                gaze_extracted.append(gaze_f)
                gaze_paths.append(None)

        cam_cap.release()
        gaze_cap.release()

        # --- align records ---
        n_gps_missing = 0
        max_cam_res = 0.0
        max_gaze_res = 0.0
        max_gps_gap = 0.0

        # pair by target timestamp index
        for i, t in enumerate(targets):
            cam_f = cam_extracted[i] if i < len(cam_extracted) else None
            gaze_f = gaze_extracted[i] if i < len(gaze_extracted) else None

            gps_rec = nearest_gps(gps_align.records, t, cfg.gps_assoc_max_gap_sec)
            gps_missing = gps_rec is None
            if gps_missing:
                n_gps_missing += 1
            else:
                gap = abs(gps_rec.time_seconds - t)
                max_gps_gap = max(max_gps_gap, gap)

            cam_res = cam_f.residual_sec if cam_f and not np.isnan(cam_f.residual_sec) else 0.0
            gaze_res = gaze_f.residual_sec if gaze_f and not np.isnan(gaze_f.residual_sec) else 0.0
            max_cam_res = max(max_cam_res, cam_res)
            max_gaze_res = max(max_gaze_res, gaze_res)

            self.all_records.append({
                "split": split,
                "video_id": video_id,
                "time_seconds": round(t, 6),
                "camera_frame_index": cam_f.frame_index if cam_f else -1,
                "gaze_frame_index": gaze_f.frame_index if gaze_f else -1,
                "cam_frame_time_sec": round(cam_f.actual_t, 6) if cam_f and not np.isnan(cam_f.actual_t) else None,
                "gaze_frame_time_sec": round(gaze_f.actual_t, 6) if gaze_f and not np.isnan(gaze_f.actual_t) else None,
                "gps_record_time_sec": round(gps_rec.time_seconds, 6) if gps_rec else None,
                "cam_residual_sec": round(cam_res, 6),
                "gaze_residual_sec": round(gaze_res, 6),
                "gps_speed_m_s": gps_rec.speed_m_s if gps_rec else None,
                "gps_speed_kmh": gps_rec.speed_kmh if gps_rec else None,
                "gps_course": gps_rec.course if gps_rec else None,
                "gps_heading_sin": gps_rec.heading_sin if gps_rec else None,
                "gps_heading_cos": gps_rec.heading_cos if gps_rec else None,
                "gps_latitude": gps_rec.latitude if gps_rec else None,
                "gps_longitude": gps_rec.longitude if gps_rec else None,
                "gps_accuracy": gps_rec.accuracy if gps_rec else None,
                "gps_acceleration": gps_rec.acceleration if gps_rec else None,
                "gps_missing": gps_missing,
                "camera_frame_path": str(cam_paths[i]) if cam_paths[i] is not None else None,
                "gaze_frame_path": str(gaze_paths[i]) if gaze_paths[i] is not None else None,
            })

        # --- QC overlays for a few timestamps ---
        self._generate_qc_overlays(split, video_id, paths, targets, cam_fps, gaze_fps, cfg)

        cam_ok = bool(int(manifest_row["cam_w"]) == cfg.expected_camera_w
                      and int(manifest_row["cam_h"]) == cfg.expected_camera_h)
        gaze_ok = bool(int(manifest_row["gaze_w"]) == cfg.expected_gaze_w
                       and int(manifest_row["gaze_h"]) == cfg.expected_gaze_h)

        return SequenceResult(
            split=split, video_id=video_id,
            n_camera_frames=sum(1 for f in cam_extracted if f.ok),
            n_gaze_frames=sum(1 for f in gaze_extracted if f.ok),
            n_aligned_records=len(targets),
            n_gps_missing=n_gps_missing,
            max_cam_residual_sec=round(max_cam_res, 6),
            max_gaze_residual_sec=round(max_gaze_res, 6),
            max_gps_gap_sec=round(max_gps_gap, 6),
            camera_dim_ok=cam_ok,
            gaze_dim_ok=gaze_ok,
            problems=problems,
        )

    # ------------------------------------------------------------------
    def _generate_qc_overlays(self, split: str, video_id: str,
                              paths: dict[str, Path], targets: list[float],
                              cam_fps: float, gaze_fps: float,
                              cfg: PreprocessConfig) -> None:
        overlay_dir = cfg.qc_dir(split) / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)

        # pick up to 4 timestamps spread across the sequence
        if len(targets) <= 4:
            pick = targets
        else:
            pick = [targets[0], targets[len(targets) // 4],
                    targets[len(targets) // 2], targets[-1]]

        cap_c = cv2.VideoCapture(str(paths["camera"]))
        cap_g = cv2.VideoCapture(str(paths["gaze"]))
        if not cap_c.isOpened() or not cap_g.isOpened():
            if cap_c.isOpened():
                cap_c.release()
            if cap_g.isOpened():
                cap_g.release()
            return

        for t in pick:
            cap_c.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * cam_fps)))
            cap_g.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * gaze_fps)))
            ret_c, cam_f = cap_c.read()
            ret_g, gaze_f = cap_g.read()
            if not ret_c or not ret_g:
                continue
            cam_rgb = cv2.cvtColor(cam_f, cv2.COLOR_BGR2RGB)
            gaze_gray = cv2.cvtColor(gaze_f, cv2.COLOR_BGR2GRAY)
            gaze_resized = process_gaze_frame(gaze_f, cfg)
            overlay = overlay_gaze_on_camera(cam_rgb, gaze_resized)
            save_qc_triptych(cam_rgb, gaze_gray, gaze_resized, overlay,
                             overlay_dir / f"{video_id}_t{int(t)}.png",
                             title=f"{video_id} @ {t}s")

        cap_c.release()
        cap_g.release()

    # ------------------------------------------------------------------
    def _print_storage_estimate(self, selected: pd.DataFrame,
                                cfg: PreprocessConfig) -> None:
        est_frames_per_seq = 0
        for _, row in selected.iterrows():
            d = min(float(row["cam_duration"]), float(row["gaze_duration"]))
            n = int(np.floor(d * cfg.sampling_fps)) + 1
            est_frames_per_seq += n
        avg = est_frames_per_seq / max(len(selected), 1)
        total_sequences = len(selected)
        # scale to full dataset if pilot
        if cfg.pilot_mode:
            full_est = int(avg * 1161)
        else:
            full_est = total_sequences * int(avg)
        # two modalities, ~150KB per JPEG at 1280x720 q95
        kb_per_frame = 150
        pilot_mb = (est_frames_per_seq * 2 * kb_per_frame) / 1024
        full_gb = (full_est * 2 * kb_per_frame) / (1024 * 1024)
        print(f"\n--- Storage estimate ---")
        print(f"  Selected sequences: {total_sequences}")
        print(f"  Avg frames per sequence (one modality): {avg:.1f}")
        print(f"  Total frames (pilot, both modalities): {est_frames_per_seq * 2:,}")
        print(f"  Estimated pilot disk usage: ~{pilot_mb:.1f} MB")
        if cfg.pilot_mode:
            print(f"  Extrapolated full dataset (1,161 seq): ~{full_est:,} frames "
                  f"(both modalities), ~{full_gb:.1f} GB")
        print(f"  (assumes ~150 KB per 1280x720 JPEG at quality {cfg.jpeg_quality})")
        print("------------------------\n")

    # ------------------------------------------------------------------
    def _write_pilot_summary(self) -> None:
        cfg = self.config
        summary_dir = PROJECT_ROOT / "reports" / "video_dataset"
        summary_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append("# Preprocessing Pilot Summary")
        lines.append("")
        lines.append(f"- Sequences processed: {len(self.results)}")
        lines.append(f"- Sampling FPS: {cfg.sampling_fps}")
        lines.append(f"- GPS max association gap: {cfg.gps_assoc_max_gap_sec}s")
        lines.append(f"- Max frame alignment error: {cfg.max_frame_align_err_sec}s")
        lines.append("")
        total_cam = sum(r.n_camera_frames for r in self.results)
        total_gaze = sum(r.n_gaze_frames for r in self.results)
        total_rec = sum(r.n_aligned_records for r in self.results)
        total_gps_miss = sum(r.n_gps_missing for r in self.results)
        lines.append(f"- Camera frames extracted: {total_cam}")
        lines.append(f"- Gaze frames extracted: {total_gaze}")
        lines.append(f"- Aligned records: {total_rec}")
        lines.append(f"- GPS-missing records: {total_gps_miss}")
        lines.append(f"- Disk written (frames): ~{self.bytes_written / (1024*1024):.1f} MB")
        lines.append("")
        lines.append("## Per-sequence results")
        lines.append("")
        lines.append("| Split | Video ID | Cam frames | Gaze frames | Records | GPS missing | Max cam residual (s) | Max gaze residual (s) | Max GPS gap (s) | Problems |")
        lines.append("|-------|----------|------------|-------------|---------|-------------|----------------------|-----------------------|-----------------|----------|")
        for r in self.results:
            lines.append(
                f"| {r.split} | {r.video_id} | {r.n_camera_frames} | {r.n_gaze_frames} | "
                f"{r.n_aligned_records} | {r.n_gps_missing} | {r.max_cam_residual_sec} | "
                f"{r.max_gaze_residual_sec} | {r.max_gps_gap_sec} | "
                f"{' | '.join(r.problems) if r.problems else 'none'} |"
            )
        lines.append("")
        lines.append("## Problems")
        lines.append("")
        any_prob = False
        for r in self.results:
            if r.problems:
                any_prob = True
                lines.append(f"- {r.split}/{r.video_id}: " + "; ".join(r.problems))
        if not any_prob:
            lines.append("- None detected.")
        (summary_dir / "pilot_preprocessing_summary.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        print(f"\nPilot summary -> {summary_dir / 'pilot_preprocessing_summary.md'}")


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    cfg = PreprocessConfig()
    pre = VideoPreprocessor(cfg)
    pre.run()


if __name__ == "__main__":
    main()
