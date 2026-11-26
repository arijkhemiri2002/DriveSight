# Phase 4 completion report - tracking (ByteTrack on the stored Phase 3 detections)

Date: 2026-11-26 | Tracking schema 4.0 | Config hash `e08ed2dc6a9e7e22` | Phase 3 config hash `09359d4da3a284ef` |
Status: **COMPLETE** (all verification checks passed)

## 1. Output

| Split | Sequences | Tracks | VRU tracks | Tracked boxes | Stored boxes (conf ≥ 0.05) | Conf ≥ 0.5 boxes in a track | Median length (frames) |
|---|---|---|---|---|---|---|---|
| training | 800 | 32,435 | 3,557 | 490,988 | 1,724,959 | 442,111 / 524,465 (84.3 %) | 11 |
| validation | 95 | 4,153 | 583 | 61,213 | 228,700 | 55,042 / 66,302 (83.0 %) | 10 |
| test | 266 | 10,633 | 1,182 | 164,458 | 586,465 | 147,875 / 174,471 (84.8 %) | 11 |
| **total** | **1,161** | **47,221** | **5,322** | **716,659** | 2,540,124 | | 11 |

- Tracker: Ultralytics `BYTETracker` 8.4.189 on the stored boxes, one tracker per sequence, every sampled frame
  passed in time order (including frames without boxes).
- Frozen setting (chosen on validation, §3): `track_high_thresh` 0.25, `track_low_thresh` 0.10,
  `new_track_thresh` 0.50, `match_thresh` 0.90, `fuse_score` true, `track_buffer_sec` 0.4 (2 frames at 5 Hz).
- Over all tracks: length median 11 frames (2.2 s), mean 15.2; 10.2 % of tracks have 1–2 frames; 11.5 % contain
  more than one class.
- Majority class of the tracks (training): car 21,223; traffic sign 3,984; pedestrian 3,154; traffic light 2,527;
  truck 861; bus 283; motorcycle 177; bicycle 136; rider 90.
- Run time: 3.7 min for 1,161 sequences on one CPU core; outputs 113 MB (`data/processed/{split}/tracks/`).

## 2. Verification (`phase4_verification_full.md`, `phase4_verification_pilot.md`)

Inputs first: `verify_preprocessing` and `verify_detection` printed ALL CHECKS PASSED on the extracted archive.
Then, for the tracks, all checks passed:
- every sequence has a done checkpoint whose file hashes match; its Phase 3 lineage (`phase3_boxes_sha256`,
  `phase3_frames_sha256`) is unchanged, and the Phase 2 and Phase 3 sequence files still match their own hashes
- every tracked box exists in Phase 3 under (`video_id`, `t_ms`, `det_idx`) with the same class, confidence and
  coordinates; every tracked frame is a Phase 3 frame
- no track has two boxes at one instant; no box belongs to two tracks; track ids positive; `track_frame` and
  `is_birth` follow time order; each track lives in one sequence (ids restart per clip)
- one summary row per track, consistent with the box table (number of boxes, first/last time, majority class)
- each video in its manifest split only; one config hash, one Phase 3 hash and one tracker setting everywhere;
  trained weights unchanged

## 3. Tuning on validation (`phase4_tuning_validation.md` / `.csv`)

81 settings on the 95 validation sequences, with a selection rule fixed before the run:
score = coverage of conf ≥ 0.5 boxes − share of 1–2 frame tracks − share of consecutive pairs with no overlap −
share of tracks with a class change.

| Validation | Ultralytics defaults (1 s buffer) | Frozen setting |
|---|---|---|
| Score | 0.488 (rank 53/81) | 0.598 (rank 1) |
| Tracks | 6,679 | 4,153 |
| Median length (frames) | 6 | 10 |
| Tracks of 1–2 frames | 24.4 % | 10.8 % |
| Conf ≥ 0.5 boxes in a track | 82.1 % | 83.0 % |
| Tracks with a class change | 8.8 % | 12.3 % |

The ten best settings lie within 0.01 (0.589–0.598) and all use `match_thresh` 0.9 and `new_track_thresh`
0.4–0.5. Because 0.9 is the edge of the grid, `match_thresh` 0.95 was checked afterwards: 0.600 (0.4 s buffer)
and 0.594 (1 s), a plateau with more class changes (15.4 %, 17.4 %), so 0.9 is kept. The written justification and
the accepted trade-off (less fragmentation against more mixed-class tracks; tracks start only from conf ≥ 0.5) are in
`docs/phase4_tracking.md` §8.

## 4. Evaluation without ground truth (`phase4_evaluation_validation.md`, `phase4_evaluation_training.md`)

| Indicator | Validation all | Validation VRU | Validation vehicles | Training all | Training VRU | Training vehicles |
|---|---|---|---|---|---|---|
| Median length (frames) | 10 | 6 | 12 | 11 | 6 | 12 |
| Tracks of 1–2 frames | 10.8 % | 19.2 % | 7.2 % | 10.3 % | 18.6 % | 6.6 % |
| Tracks of ≥ 10 frames | 53.3 % | 33.4 % | 59.2 % | 55.4 % | 34.9 % | 60.3 % |
| Tracks with a class change | 12.3 % | 16.1 % | 13.9 % | 11.3 % | 16.4 % | 12.5 % |
| Tracks with a bridged gap | 22.6 % | 29.2 % | 24.2 % | 20.8 % | 23.1 % | 23.2 % |
| Pairs with centre jump > 0.1 | 0.3 % | 0.3 % | 0.3 % | 0.2 % | 0.1 % | 0.3 % |
| Pairs with no overlap | 0.1 % | 0.6 % | 0.1 % | 0.1 % | 0.4 % | 0.1 % |
| Conf ≥ 0.5 boxes in a track | 83.0 % | 65.6 % | 92.3 % | 84.3 % | 66.9 % | 92.8 % |

Training played no part in the tuning and gives the same values as validation: the setting is not fitted to the
95 validation clips. The test split is evaluated once, at the end of the project.

## 5. Findings

| Finding | Evidence | Consequence |
|---|---|---|
| **`BYTETracker` 8.4.189 has no `frame_rate` argument**; `track_buffer` is a number of frames | `ultralytics/trackers/byte_tracker.py`: `max_frames_lost = args.track_buffer` | The hand-over guide's "frame_rate=5" cannot be passed. The buffer is set in seconds and converted at 5 Hz; the default 30 frames would have meant 6 s |
| `update()` returns only confirmed tracks, so the first box of every new object is missing from its output | source + unit test | The birth box is read from the tracker state and kept for every track confirmed later; 1–2 frame candidates never confirmed are dropped and counted per sequence |
| At 5 Hz, IoU association fails when an object moves more than about 60 % of its width between frames | synthetic test (40/50/60 px steps for an 80 px box) | Expected source of fragmentation for close, fast objects; VRU tracks are the most affected |
| VRU tracks are shorter and more fragmented than vehicle tracks | §4 | Phase 5 should not require long VRU tracks; detector recall (VRU-4 43.8 %) is the main limit |
| With `new_track_thresh` 0.5 an object detected only at conf 0.25–0.5 never starts a track | tuning trade-off | Lower VRU coverage of confident boxes (66 % vs 92 % for vehicles); Phase 5 can still use the untracked Phase 3 boxes |

## 6. Limitations carried forward

- No ground-truth trajectories on BDDA: the indicators are indirect. An evaluation on the BDD100K MOT set
  (annotated at 5 fps) is possible but not done.
- Visual check (`draw_tracks.py`) not done yet: it needs the BDDA camera videos of a few sequences.
- ByteTrack is class-agnostic: 11.5 % of tracks contain several classes; the majority class and its share are stored.
- Displacements (`d_cx_n`, `d_bottom_y_n`, `log_area_ratio`) are image-space, ordinal proxies, not distances or speeds.
- The clips are braking events, and the gaze map is an in-lab attention reference, not the recording driver's gaze.
