# Phase 4 — Tracking (ByteTrack on the stored detections)

Status: **complete**. 1,161 sequences, 47,221 tracks, 716,659 tracked boxes, all verification checks passed.
See `reports/video_dataset/phase4_completion_report.md`.

## 1. Purpose

Give every detected object a stable identifier (`track_id`) from one sampled frame to the next, inside each
sequence. Phase 5 uses the tracks to follow a road user over time (approach, persistence, attention on it)
and to join it to the gaze and GPS rows.

Input: only the verified Phase 3 output (`detections/sequence_detections/{video_id}.parquet` and
`{video_id}_frames.csv`). No video is decoded, YOLO is not re-run, no GPU is needed.

## 2. Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Tracker | Ultralytics `BYTETracker` 8.4.189 (`ultralytics/trackers/byte_tracker.py`) on the stored boxes | Recommended in the hand-over guide; uses the low-score boxes (kept down to 0.05 in Phase 3) in its second association step |
| Input object | `Detections` (`tracker_core.py`): NumPy `conf`, `cls`, `xywh` (box **centre**, width, height), `len()`, boolean indexing | What `BYTETracker.update` reads (`_split_detections`, `init_track`, `parse_bboxes`), checked in the source, not guessed |
| One tracker per sequence | new `BYTETracker` for every clip | Track ids restart at 1 per clip; a track can never cross two sequences |
| Every frame is passed | including frames with no box (`{video_id}_frames.csv`) | Lost tracks age by frame count; skipping empty frames would keep them alive too long (tested) |
| Time base | `track_buffer = round(track_buffer_sec × 5 Hz)` | **In 8.4.189 `BYTETracker` has no `frame_rate` argument**: `max_frames_lost = args.track_buffer`, in frames. The default 30 means 1 s at 30 fps but 6 s at our 5 Hz. The buffer is therefore set in seconds and converted (1.0 s → 5 frames) |
| Link to Phase 3 | `det_idx` of the matched box, read from the track's `idx` | Boxes are passed in `det_idx` order, so the position `idx` maps back to `det_idx`. This is the key Phase 5 uses (`video_id`, `t_ms`, `det_idx`) |
| Birth box | kept (`keep_birth_box`) | `update()` returns only confirmed tracks; a new track is confirmed when matched again in the next frame, so its first box is not in the output. The tracker state after each update contains it; it is kept for every track that is confirmed later. Tracks never confirmed (an isolated box) are dropped and counted (`n_unconfirmed_dropped`) |
| Box values | copied from Phase 3 (not the Kalman-smoothed box) | The tracked table must agree exactly with the detections; the verifier checks it |
| Thresholds | tuned on **validation** only (`tune_tracking.py`), then frozen as the defaults of `track_config.py`; Ultralytics `bytetrack.yaml` values kept in `ULTRALYTICS_DEFAULTS` as the reference | The test split is used once, with frozen settings |
| Resume | per-sequence parquet + checkpoint JSON (config hash incl. the Phase 3 hash, file hashes, lineage), atomic writes | Same pattern as Phases 2 and 3; a change of setting or of the Phase 3 files invalidates old tracks |

## 3. Output layout

```
data/processed/{split}/tracks/
├── tracks.parquet                 merged tracked boxes of the split
├── track_summary.csv              merged, one row per track
├── _schema.json                   config hash, tracker settings, environment, coverage
├── sequence_tracks/{video_id}.parquet, {video_id}_summary.csv
└── sequence_checkpoints/{video_id}.json
data/processed/{split}/qc/tracks/{video_id}_tracks.mp4, {video_id}_sheet.jpg     (draw_tracks.py, needs the videos)
```

`tracks.parquet` columns: `split, video_id, sample_idx, t_ms, det_idx` (link to the Phase 3 box), `track_id`,
`track_frame` (0 for the first box of the track), `is_birth`, then the Phase 3 box columns
`cls_id, cls_name, is_vru, conf, x1, y1, x2, y2, cx_n, cy_n, w_n, h_n, area_frac, bottom_y_n, x_offset_n, touches_border`.

`track_summary.csv` columns: majority class (`cls_name`, ties → higher mean confidence), `cls_majority_frac`,
`n_classes`, `is_vru`, `first_t_ms`, `last_t_ms`, `duration_sec`, `n_boxes`, `span_frames`, `n_gaps`,
`max_gap_frames`, `mean_conf`, `max_conf`, first/last/difference of `cx_n`, `cy_n`, `bottom_y_n`, `h_n`,
`log_area_ratio` (last/first box area), `max_center_jump`, `min_consecutive_iou`, `touches_border_frac`.
Displacements are in normalised image units; like `bottom_y_n` and `h_n` in Phase 3 they are ordinal
proxies, not metres.

## 4. Evaluation without ground truth

BDDA has no annotated trajectories, so identity accuracy (IDF1, MOTA, HOTA) cannot be measured on it.
`evaluate_tracking.py` reports indirect indicators, for all tracks, VRU tracks and vehicle tracks:
track length and share of 1–2 frame tracks (fragmentation), share of tracks with a class change (mixing,
or detector confusion such as motorcycle/bicycle), share of consecutive pairs whose centre jumps more than
0.10 of the image or whose boxes do not overlap, share of tracks with a bridged gap, and the share of
boxes with conf ≥ 0.5 that end up in a track. The 0.10 and 0.5 are reporting choices, not tuned values.
A visual check is done with `draw_tracks.py` on a few sequences.

An evaluation with ground truth is possible on the BDD100K MOT set (annotated at 5 fps): run our YOLOv8s
and this tracker on a small part of its validation split and score with TrackEval. Check the download size
and frame rate on the BDD100K site before committing to it.

## 5. Tuning (validation only)

`tune_tracking.py` runs a grid in memory on the 95 validation sequences (`track_high_thresh` 0.25/0.40/0.50,
`new_track_thresh` 0.25–0.60 and never below the first-stage threshold, `match_thresh` 0.70/0.80/0.90,
`track_buffer_sec` 0.4/1.0/2.0; `track_low_thresh` 0.10 and `fuse_score` fixed). The selection rule is fixed
before looking at the results:

    score = coverage(conf ≥ 0.5) − share of 1–2 frame tracks − share of pairs with no overlap − share of tracks with a class change

Ties go to the Ultralytics defaults, then to the closest setting. The full table is kept
(`phase4_tuning_validation.md` / `.csv`); the chosen setting and the reason are written in §8 below, then the
same setting is used for every split.

## 6. How to run (from the repository root)

```powershell
python -m pytest -q tests                                   # includes tests/test_phase4_tracking.py
python -m src.features.verify_preprocessing                 # Phase 2 intact
python -m src.detection.verify_detection                    # Phase 3 intact
python -m src.tracking.run_tracking --pilot                 # 9 sequences
python -m src.tracking.verify_tracking --pilot
python -m src.tracking.tune_tracking                        # validation only (~20 min); --quick to check it runs
python -m src.tracking.run_tracking                         # all splits, frozen setting (defaults)
python -m src.tracking.verify_tracking                      # must print ALL CHECKS PASSED
python -m src.tracking.evaluate_tracking --split validation
python -m src.tracking.draw_tracks --split validation --n 5      # needs the camera videos
```

Measured on the real data: 3.7 min for the 1,161 sequences on one CPU core (0.2 s per sequence);
tuning 81 settings on the 95 validation sequences: about 17 min.

## 7. Known limitations

- **5 Hz is coarse.** Association uses box overlap (IoU) plus a Kalman prediction. On synthetic boxes,
  tracking holds while an object moves up to about 60 % of its own width per frame and fails beyond
  (no overlap left before the tracker has learned the velocity). Close, fast objects are the expected
  source of fragmentation; the evaluation measures it.
- **Detector recall bounds tracking.** VRU-4 recall was 43.8 % on the V2 validation set. The tracker can
  bridge short gaps (up to `track_buffer` frames) and recover low-score boxes, but it cannot create an
  object that is never detected.
- **ByteTrack is class-agnostic.** A track can contain several classes; this is measured, and the majority
  class is reported.
- **No identity ground truth on BDDA.** The indicators of §4 are indirect.
- The clips are braking events and the gaze map is an in-lab attention reference, not the recording
  driver's gaze (see Phase 2): this matters for Phase 5, not for tracking itself.

## 8. Results on real data

**Inputs intact.** Before any tracking, `verify_preprocessing` and `verify_detection` printed ALL CHECKS PASSED
on the extracted `drivesight_processed.zip` (62,008 rows; 2,540,124 boxes).

**Pilot** (11 sequences, Ultralytics defaults): all checks passed; 0.15–0.46 s per sequence.

**Tuning on validation** (`phase4_tuning_validation.md`, 81 settings × 95 sequences). Score of the
Ultralytics defaults (1 s buffer): 0.488. Best: 0.598 for

    track_high_thresh 0.25, new_track_thresh 0.50, match_thresh 0.90, track_buffer_sec 0.4 (2 frames)

| Validation, all tracks | Ultralytics defaults | Frozen setting |
|---|---|---|
| Tracks | 6,679 | 4,153 |
| Median length (frames) | 6 | 10 |
| Tracks of 1–2 frames | 24.4 % | 10.8 % |
| Boxes with conf ≥ 0.5 in a track | 82.1 % | 83.0 % |
| Pairs with no overlap | 0.06 % | 0.12 % |
| Tracks with a class change | 8.8 % | 12.3 % |
| VRU: tracks of 1–2 frames | 31.4 % | 19.2 % |
| VRU: tracks with a class change | 12.8 % | 16.1 % |

Why this setting is kept:
- It is the result of the rule fixed before the tuning, not a hand pick. The ten best settings all lie within
  0.01 of each other (0.589–0.598) and share `match_thresh` 0.9 and `new_track_thresh` 0.4–0.5: the choice is
  driven by those two parameters, the buffer and first-stage threshold matter little.
- `match_thresh` 0.9 is at the edge of the grid, so 0.95 was checked afterwards (same other values): score
  0.600 with a 0.4 s buffer and 0.594 with 1 s, i.e. a plateau, while class changes keep rising (15.4 %
  and 17.4 %). 0.9 is kept: a larger value would trade more mixed-class tracks for no measurable gain.
- `new_track_thresh` 0.5 (vs 0.25) starts tracks only from confident boxes. This halves fragmentation but means
  an object seen only at conf 0.25–0.5 is never tracked (its boxes can still join an existing track through the
  first and second association). This is the main cost, and it falls more on VRUs, which the detector finds
  with lower confidence: VRU coverage of conf ≥ 0.5 boxes is 65.6 % against 92.3 % for vehicles.
- A 0.4 s buffer (2 frames) is enough at 5 Hz: longer buffers bridged slightly more gaps but also produced more
  class changes, with the same score.
- The trade-off accepted: fewer, longer, less fragmented tracks (better for following a road user in Phase 5)
  against +3.5 points of tracks with a class change. The majority class and `cls_majority_frac` are stored so
  Phase 5 can filter mixed tracks.

**Full run** (frozen setting, `phase4_verification_full.md`): 1,161/1,161 sequences done, ALL CHECKS PASSED.

| Split | Sequences | Tracks (VRU) | Tracked boxes | Conf ≥ 0.5 boxes in a track | Median length |
|---|---|---|---|---|---|
| training | 800 | 32,435 (3,557) | 490,988 | 84.3 % | 11 |
| validation | 95 | 4,153 (583) | 61,213 | 83.0 % | 10 |
| test | 266 | 10,633 (1,182) | 164,458 | 84.8 % | 11 |

**Evaluation without ground truth** (`phase4_evaluation_validation.md`, `phase4_evaluation_training.md`).
Training, which played no part in the tuning, gives the same picture as validation (1–2 frame tracks 10.3 %
vs 10.8 %, class changes 11.3 % vs 12.3 %, coverage 84.3 % vs 83.0 %), so the setting is not fitted to the 95
validation clips. VRU tracks are shorter (median 6 frames vs 12 for vehicles) and more often fragmented
(18.6 % vs 6.6 % of 1–2 frame tracks), as expected from the detector's VRU recall and from fast, close
pedestrians at 5 Hz. Impossible jumps are rare (no-overlap pairs 0.1 %, centre jumps > 0.1 of the image 0.2 %).
The test-split indicators are left for the final evaluation.

**Not done yet:**
- visual check (`draw_tracks.py`), which needs the BDDA camera videos of a few sequences;
- evaluation with ground truth on the BDD100K MOT set (optional).
