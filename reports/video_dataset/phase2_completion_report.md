# Phase 2 completion report - on-the-fly multimodal preprocessing

Date: 2026-11-26 | Schema 2.1 | Config hash `6f11bf7c59d912ee` | Status: **COMPLETE (all verification checks passed)**

## 1. Output

| Split | Sequences | Rows (5 Hz) |
|---|---|---|
| training | 800 | 42,553 |
| validation | 95 | 5,330 |
| test | 266 | 14,125 |
| **total** | **1,161** | **62,008** |

- Location: `data/processed/{split}/metadata/aligned_records.csv` (+ per-sequence CSVs, checkpoints, `sequence_stats.csv`, `_schema.json`), not tracked by Git.
- Size: 41.1 MB in total (≈257 bytes per row), 77 QC overlays. No image frames are stored. The original per-frame JPEG design was estimated at 18.8 GB.
- Processing: 18.6 min for 1,161 sequences (median 0.9 s per sequence) on the local CPU. 1,161 done, 0 problems, 0 failures.

## 2. Verification (`phase2_verification_full.md`, independent of the runner)

All checks passed in every split:
- every sequence has a valid checkpoint and its CSV matches the recorded SHA-256
- the rows per sequence match the sampling plan recomputed from the manifest (62,008 / 62,008)
- (video_id, t_ms) is unique, steps are 200 ms, each video is in one split only and in the manifest's split
- 0 camera / 0 gaze decode failures; decoded frame counts ≥ manifest counts
- max |camera residual| 16.6 ms (bound: half a frame at 29.97 fps = 16.7 ms); max |gaze residual| 13.8 ms; max |camera − gaze| 30.4 ms
- container timestamps never drift from index/fps (max 0.0 s): no variable-frame-rate clips
- no negative or implausible speeds; invalid values are NaN exactly where they are flagged; course in [0, 360)
- trained weights unchanged (3/3 SHA-256 match)

## 3. Supporting checks

| Check | Result | Report |
|---|---|---|
| GPS validity scan (1,435 files) | 27 records with speed = −1 in 16 clean sequences, 1 invalid course; nothing else. All are flagged and set to NaN. validation/1744 has 1 such record (5 rows at 5 Hz), not 5 records. | `gps_validity_scan.md` |
| Frame accuracy (11 sequences, 1,360 frames) | seek and sequential decoding are pixel-identical: the pilot-v1 frame selection was right | `frame_accuracy_check.md` |
| Gaze crop diagnostic (71 sequences, 3,925 frames) | see §4 | `gaze_bar_diagnostic.md/.png` |
| Extended pilot (11 sequences) | all checks passed | `phase2_verification_pilot.md` |

## 4. Gaze crop: finding and decision

The gaze map is 1024×768. The crop keeps rows 96:672, i.e. the 1024×576 (16:9) area in which the 1280×720 video is letterboxed. This geometry follows exactly from the two resolutions.

Findings:
- 0.69 % of all gaze mass lies in the bars (mean over 3,925 frames). The vertical profile is smooth across both boundaries and peaks slightly below the image centre (row ≈ 430), where the road ahead is.
- The top bar holds almost nothing. The bottom bar holds a decaying tail plus, in some frames, blobs centred on or just below the bottom edge (max value 167, worst frame 43.8 % of its mass in the bar). Only 53 % of the bar mass lies within 24 px of the boundary.
- Over the full set, 1,456 rows (2.3 %, in 297 sequences) have more than 5 % of their gaze mass in the bars.
- The QC overlays place the attention blobs on lead vehicles, brake lights, traffic lights and the area around pedestrians: no sign of a spatial shift.

Interpretation: the crop is geometrically correct. The bar signal is heat-map blur plus fixations recorded at, or slightly below, the bottom edge of the video. Those are plausibly eye-tracker error near the screen edge, or attention to the road just ahead. The automated verdict ("spill-over") passed its rule only narrowly, so this is not treated as fully explained.

Decision: keep the crop. `gaze_bar_mass_frac` is stored on every row. Phase 5 will treat rows above 5 % as "attention partly off-image" and run its gaze-overlap results with and without them (sensitivity analysis).

## 5. Data facts to carry into Phases 3-6

- The gaze maps aggregate several **in-lab observers** (BDD-A protocol). They are not the recording driver's gaze. 64 % of rows contain two or more separate attention regions.
- The clips are **braking events**: median GPS acceleration is −0.53 m/s² (1st percentile −4.85, 0.1th percentile −7.45 m/s²). Risk base rates are therefore not those of normal driving.
- Speed: median 20.5 km/h, 99th percentile 62.9 km/h; 4.6 % of rows are near-stationary (< 1 km/h).
- `gps_speed_valid` is true for 99.78 % of rows. Interpolated speed covers 90.4 % of rows (none in the first and last ≈0.5 s of a clip, where there is no GPS record to interpolate from).
- training/1660 has 5 rows without a GPS record within 1 s (its GPS starts late and ends early). training/1656, 1660 and 1684 have association gaps of 0.69-1.0 s.

## 6. Open items (not blocking Phase 3)

- The archived pilot-v1 outputs in `data/processed/_archive_pilot_v1/` (≈111 MB of JPEGs) can be deleted to free space. This is pending the owner's decision.
- `requirements.txt` pins differ from the installed environment (Python 3.12, numpy 2.4.6, pandas 2.2.3): to be resolved with a dedicated venv at the start of Phase 3.
