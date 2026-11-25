# Phase 2 verification - full

Config hash `6f11bf7c59d912ee` | sequences in scope: 1161 {'training': 800, 'test': 266, 'validation': 95}

## training

- PASS - every sequence has a done checkpoint (current config): {'done': 800}
- PASS - per-sequence CSVs match their checkpoint hash
- PASS - rows per sequence == recomputed sampling plan: 800 sequences
- PASS - split column consistent
- PASS - unique (video_id, t_ms)
- PASS - video split matches manifest
- PASS - 200 ms steps and consecutive sample_idx
- PASS - all camera and gaze decodes ok: cam fail 0, gaze fail 0
- PASS - |camera residual| <= half a camera frame: max 0.0166s
- PASS - |gaze residual| <= half a gaze frame: max 0.0138s
- PASS - |camera - gaze frame time| <= half cam + half gaze frame: max 0.0304s
- PASS - last sample inside both videos
- PASS - decoded frame counts >= manifest counts
- PASS - no negative or implausible speed values: min 0.0, max 32.05
- PASS - speed NaN exactly where flagged invalid
- PASS - interpolated speed >= 0
- PASS - course in [0, 360)
- PASS - GPS gap <= 1.0s: max 0.997
- PASS - schema 2.1 columns present (gaze_bar_mass_frac)
- info: gaze mass in cropped-out bars: mean 0.0049, p99 0.1163, max 0.4384; rows > 5%: 971 (184 sequences)
- info: rows 42553, rows with invalid speed 109 (11 sequences), rows without GPS 5, empty gaze rows 0, max |cam container drift| 0.0000s

## validation

- PASS - every sequence has a done checkpoint (current config): {'done': 95}
- PASS - per-sequence CSVs match their checkpoint hash
- PASS - rows per sequence == recomputed sampling plan: 95 sequences
- PASS - split column consistent
- PASS - unique (video_id, t_ms)
- PASS - video split matches manifest
- PASS - 200 ms steps and consecutive sample_idx
- PASS - all camera and gaze decodes ok: cam fail 0, gaze fail 0
- PASS - |camera residual| <= half a camera frame: max 0.0166s
- PASS - |gaze residual| <= half a gaze frame: max 0.0138s
- PASS - |camera - gaze frame time| <= half cam + half gaze frame: max 0.0304s
- PASS - last sample inside both videos
- PASS - decoded frame counts >= manifest counts
- PASS - no negative or implausible speed values: min 0.0, max 23.68
- PASS - speed NaN exactly where flagged invalid
- PASS - interpolated speed >= 0
- PASS - course in [0, 360)
- PASS - GPS gap <= 1.0s: max 0.51
- PASS - schema 2.1 columns present (gaze_bar_mass_frac)
- info: gaze mass in cropped-out bars: mean 0.0051, p99 0.1044, max 0.3250; rows > 5%: 216 (57 sequences)
- info: rows 5330, rows with invalid speed 5 (1 sequences), rows without GPS 0, empty gaze rows 0, max |cam container drift| 0.0000s

## test

- PASS - every sequence has a done checkpoint (current config): {'done': 266}
- PASS - per-sequence CSVs match their checkpoint hash
- PASS - rows per sequence == recomputed sampling plan: 266 sequences
- PASS - split column consistent
- PASS - unique (video_id, t_ms)
- PASS - video split matches manifest
- PASS - 200 ms steps and consecutive sample_idx
- PASS - all camera and gaze decodes ok: cam fail 0, gaze fail 0
- PASS - |camera residual| <= half a camera frame: max 0.0166s
- PASS - |gaze residual| <= half a gaze frame: max 0.0138s
- PASS - |camera - gaze frame time| <= half cam + half gaze frame: max 0.0304s
- PASS - last sample inside both videos
- PASS - decoded frame counts >= manifest counts
- PASS - no negative or implausible speed values: min 0.0, max 22.6
- PASS - speed NaN exactly where flagged invalid
- PASS - interpolated speed >= 0
- PASS - course in [0, 360)
- PASS - GPS gap <= 1.0s: max 0.599
- PASS - schema 2.1 columns present (gaze_bar_mass_frac)
- info: gaze mass in cropped-out bars: mean 0.0040, p99 0.0918, max 0.2766; rows > 5%: 269 (56 sequences)
- info: rows 14125, rows with invalid speed 20 (4 sequences), rows without GPS 0, empty gaze rows 0, max |cam container drift| 0.0000s

## Storage and protection

- PASS - no bulk frame folders in data/processed
- info: data/processed size (excluding archive) 41.1 MB, QC overlays 77
- PASS - total rows == expected from manifest: 62008 / 62008
- PASS - trained weights unchanged (SHA-256): 3/3 match

**Result: ALL CHECKS PASSED**
