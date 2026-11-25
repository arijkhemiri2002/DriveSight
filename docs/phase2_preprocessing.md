# Phase 2 — On-the-fly multimodal preprocessing

Status: implementation complete and tested on synthetic data. **The real-data pilot and the full run are still pending.**

## 1. Purpose

Turn each clean BDDA sequence (camera video, gaze-map video, GPS JSON) into one row per 200 ms. Each row holds aligned timing, gaze descriptors and GPS context. No image frames are stored. Phase 3 (YOLO) and Phase 5 (fusion) decode the videos again with the same frame iterator, so the timestamp→frame mapping is defined in one place only (`frame_extractor.sample_plan`).

Input: `reports/video_dataset/dataset_manifest.csv` (1,161 sync_ok sequences: 800 training / 95 validation / 266 test). The raw data in `data/BDDA/` is read-only; the code refuses to write under it.

## 2. Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Frame access | **Sequential decoding**: grab every frame, keep only sampled ones | Exact by construction; seeking (`CAP_PROP_POS_FRAMES`) is decoder-dependent on H.264, and the pilot's residual (from `CAP_PROP_POS_MSEC`) could not detect a wrong seek. Clips are short (≈10 s, max 32 s), so decoding everything costs little. `check_frame_accuracy.py` measures whether the pilot's seeks were right. |
| Sample times | t_k = k / 5 Hz while t_k ≤ min(last camera frame time, last gaze frame time) | The pilot's last sample (t = duration) fell past the last camera frame and was clamped silently (0.033 s error at 30 fps, and the final QC overlay was missing in all 9 sequences). |
| Frame per sample | nearest frame, `floor(t·fps + 0.5)` | Bounded error of half a frame: ≤ 16.7 ms at 30 fps, ≤ 8.3 ms at 60 fps, ≤ 17.2 ms for gaze (29 fps). |
| Frame time | `index / fps` (nominal, constant frame rate) | Container timestamps are also recorded per sequence (`max_*_container_drift_sec`); a drift above half a frame is logged as a possible variable-frame-rate warning. |
| Memory | 2 decoded frames at a time (≈5 MB) | No bulk frame storage. |
| Invalid GPS | value → NaN + `*_valid` flag, never passed on | iOS writes −1 for unknown speed/course (seen in validation/1744). |
| GPS at 5 Hz | nearest 1 Hz record (`gps_speed_m_s`) **and** linear interpolation between two valid records ≤ 2 s apart (`gps_speed_interp_m_s`) | Nearest is the raw measurement (a step function); interpolation is a smoother derived signal. Both are kept and labelled; there is no extrapolation outside the GPS coverage (≈0.5 s at each end). |
| Acceleration | backward difference between two consecutive valid records, dt ≤ 2 s | 1 s finite difference of phone GPS speed: a coarse proxy, not a measured acceleration. |
| Gaze descriptors | computed on the cropped 1024×576 map, at ¼ resolution | Cheap QC signals and candidate features. Full maps are re-decoded in Phase 5. |
| QC images | one triptych (camera, gaze, overlay) at the middle sample for a deterministic ~5 % of sequences + every sequence with a problem | The pilot's 4 overlays per sequence would be ≈1 GB for the full set. |
| Resume | per-sequence CSV + checkpoint JSON (config hash + CSV SHA-256), atomic writes | Interruptions lose at most one sequence; a change of any output-affecting setting invalidates old checkpoints. |

## 3. Important methodological caveat: what the gaze map is

BDD-A gaze maps come from an **in-lab protocol**: drivers watched recorded braking-event clips, and the maps average several independent observers (Xia et al., *Predicting Driver Attention in Critical Situations*, ACCV 2018). They show where attentive human drivers looked for that scene. They are **not** the gaze of the person driving the recording car. DriveSight must therefore not claim that a recorded driver was distracted. A defensible framing is to compare scene hazards with a human attention reference. The clips are also selected braking events, so risk base rates are not those of normal driving.

## 4. Output layout

```
data/processed/
├── logs/run_<timestamp>.log, run_<timestamp>_summary.json
└── {split}/
    ├── metadata/
    │   ├── aligned_records.csv              merged rows of every valid sequence
    │   ├── sequence_stats.csv               one row per sequence (from checkpoints)
    │   ├── _schema.json                     schema version, config + hash, coverage
    │   ├── sequence_records/{video_id}.csv
    │   └── sequence_checkpoints/{video_id}.json
    ├── qc/overlays/{video_id}_t{ms}.jpg
    └── logs/processing_log.csv              append-only, one row per processed sequence
```

The pilot-v1 outputs (JPEG frames, old CSV, crop QC) are archived under `data/processed/_archive_pilot_v1/`.

## 5. `aligned_records.csv` columns (schema 2.0)

| Column | Meaning |
|---|---|
| `split`, `video_id`, `sample_idx`, `t_sec`, `t_ms` | identity and target time from video start; (`video_id`, `t_ms`) is unique |
| `cam_frame_idx`, `cam_frame_t_sec`, `cam_residual_sec`, `cam_ok` | selected camera frame, its nominal time, signed difference from `t_sec`, decode success |
| `cam_mean_intensity` | mean pixel value (QC: black/corrupt frames, night) |
| `gaze_frame_idx`, `gaze_frame_t_sec`, `gaze_residual_sec`, `gaze_ok` | same for the gaze video (29 fps) |
| `cam_gaze_dt_sec` | camera frame time − gaze frame time |
| `gaze_max`, `gaze_mean` | intensity statistics of the cropped gaze map (0–255) |
| `gaze_active_frac` | fraction of pixels above the background threshold (20) |
| `gaze_peak_x/y`, `gaze_centroid_x/y` | normalised [0, 1] image coordinates (same in the 1280×720 camera frame) |
| `gaze_spread` | RMS distance of the gaze mass from its centroid (normalised units) |
| `gaze_n_regions` | number of separate attention regions (≥ 0.1 % of the frame); several are expected because maps aggregate several observers |
| `gaze_empty` | max ≤ 10: no usable attention signal |
| `gaze_bar_max` | max value in the cropped-out black bars (should be 0) |
| `gps_matched`, `gps_rec_t_sec`, `gps_gap_sec` | nearest GPS record within 1.0 s, its time, the gap |
| `gps_speed_m_s`, `gps_speed_kmh`, `gps_speed_valid` | nearest-record speed (NaN if invalid) |
| `gps_speed_interp_m_s` | interpolated speed (derived) |
| `gps_accel_m_s2` | backward-difference acceleration of the nearest record (derived, coarse) |
| `gps_course_deg`, `gps_heading_sin/cos`, `gps_course_valid` | heading |
| `gps_lat`, `gps_lon`, `gps_accuracy_m`, `gps_position_valid` | position |

## 6. How to run (from the repository root)

```powershell
python -m pytest -q tests                                  # 19 tests on synthetic data
python -m src.features.gps_validity_scan                   # step 1: GPS value scan (read-only)
python -m src.features.check_frame_accuracy                # step 4: seek vs sequential
python -m src.features.run_preprocessing --pilot           # extended pilot
python -m src.features.verify_preprocessing --pilot
python -m src.features.run_preprocessing                   # full run (resumes automatically)
python -m src.features.verify_preprocessing                # full verification
```

Useful options: `--dry-run` (selection and expected rows only), `--split`, `--videos`, `--limit`, `--force`, `--qc-fraction`. Stop a run with Ctrl+C at any time and run the same command again to resume.

## 7. Known limitations

- Frame times assume a constant frame rate. The container drift is recorded and flagged, but the samples are not corrected for it.
- Camera (29.97/30/59.94/60 fps) and gaze (29 fps) videos are assumed to start together and play in real time. Their durations differ by a constant ≈0.06 s, which is below two gaze frames.
- GPS is 1 Hz phone GPS: speed is a step function at 5 Hz, and acceleration is noisy.
- The validation split has only 95 sequences, so thresholds tuned on it will be fragile.
