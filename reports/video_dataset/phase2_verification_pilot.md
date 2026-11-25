# Phase 2 verification — pilot

Config hash `14e2a0c8ac197272` | sequences in scope: 11 {'test': 4, 'training': 4, 'validation': 3}

## training

- PASS — every sequence has a done checkpoint (current config): {'done': 4}
- PASS — per-sequence CSVs match their checkpoint hash
- PASS — rows per sequence == recomputed sampling plan: 4 sequences
- PASS — split column consistent
- PASS — unique (video_id, t_ms)
- PASS — video split matches manifest
- PASS — 200 ms steps and consecutive sample_idx
- PASS — all camera and gaze decodes ok: cam fail 0, gaze fail 0
- PASS — |camera residual| <= half a camera frame: max 0.0098s
- PASS — |gaze residual| <= half a gaze frame: max 0.0138s
- PASS — |camera - gaze frame time| <= half cam + half gaze frame: max 0.0234s
- PASS — last sample inside both videos
- PASS — decoded frame counts >= manifest counts
- PASS — no negative or implausible speed values: min 0.0, max 13.84
- PASS — speed NaN exactly where flagged invalid
- PASS — interpolated speed >= 0
- PASS — course in [0, 360)
- PASS — GPS gap <= 1.0s: max 0.5
- info: rows 310, rows with invalid speed 0 (0 sequences), rows without GPS 0, empty gaze rows 0, max |cam container drift| 0.0000s

## validation

- PASS — every sequence has a done checkpoint (current config): {'done': 3}
- PASS — per-sequence CSVs match their checkpoint hash
- PASS — rows per sequence == recomputed sampling plan: 3 sequences
- PASS — split column consistent
- PASS — unique (video_id, t_ms)
- PASS — video split matches manifest
- PASS — 200 ms steps and consecutive sample_idx
- PASS — all camera and gaze decodes ok: cam fail 0, gaze fail 0
- PASS — |camera residual| <= half a camera frame: max 0.0138s
- PASS — |gaze residual| <= half a gaze frame: max 0.0138s
- PASS — |camera - gaze frame time| <= half cam + half gaze frame: max 0.0274s
- PASS — last sample inside both videos
- PASS — decoded frame counts >= manifest counts
- PASS — no negative or implausible speed values: min 0.0, max 18.36
- PASS — speed NaN exactly where flagged invalid
- PASS — interpolated speed >= 0
- PASS — course in [0, 360)
- PASS — GPS gap <= 1.0s: max 0.5
- info: rows 170, rows with invalid speed 5 (1 sequences), rows without GPS 0, empty gaze rows 0, max |cam container drift| 0.0000s

## test

- PASS — every sequence has a done checkpoint (current config): {'done': 4}
- PASS — per-sequence CSVs match their checkpoint hash
- PASS — rows per sequence == recomputed sampling plan: 4 sequences
- PASS — split column consistent
- PASS — unique (video_id, t_ms)
- PASS — video split matches manifest
- PASS — 200 ms steps and consecutive sample_idx
- PASS — all camera and gaze decodes ok: cam fail 0, gaze fail 0
- PASS — |camera residual| <= half a camera frame: max 0.0098s
- PASS — |gaze residual| <= half a gaze frame: max 0.0138s
- PASS — |camera - gaze frame time| <= half cam + half gaze frame: max 0.0234s
- PASS — last sample inside both videos
- PASS — decoded frame counts >= manifest counts
- PASS — no negative or implausible speed values: min 0.0, max 10.23
- PASS — speed NaN exactly where flagged invalid
- PASS — interpolated speed >= 0
- PASS — course in [0, 360)
- PASS — GPS gap <= 1.0s: max 0.5
- info: rows 200, rows with invalid speed 0 (0 sequences), rows without GPS 0, empty gaze rows 0, max |cam container drift| 0.0000s

## Storage and protection

- PASS — no bulk frame folders in data/processed
- info: data/processed size (excluding archive) 1.3 MB, QC overlays 11
- PASS — total rows == expected from manifest: 680 / 680
- PASS — trained weights unchanged (SHA-256): 3/3 match

**Result: ALL CHECKS PASSED**
