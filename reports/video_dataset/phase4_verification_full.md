# Phase 4 verification - full

Sequences in scope: 1161 {'training': 800, 'test': 266, 'validation': 95}

## training

- PASS - every sequence has a done checkpoint: {'done': 800}
- PASS - per-sequence files match checkpoint hashes
- PASS - Phase 3 lineage unchanged (phase3_*_sha256)
- PASS - Phase 2 and Phase 3 sequence files unchanged (their own hashes)
- PASS - every tracked box exists in Phase 3 (video_id, t_ms, det_idx): 0 missing
- PASS - tracked boxes carry the Phase 3 class, confidence and coordinates: 0 differ
- PASS - every tracked frame is a Phase 3 frame
- PASS - no track has two boxes at one instant
- PASS - no box belongs to two tracks
- PASS - track ids are positive
- PASS - track_frame and is_birth follow time order
- PASS - one summary row per track: 32435 summary rows, 32435 tracks
- PASS - summary agrees with the box table (n_boxes, first/last time, majority class): 0 differ
- PASS - split matches manifest
- info: tracks 32435 (VRU 3557), tracked boxes 490988 of 1724959 stored; boxes with conf >= 0.5 in a track: 442111/524465 (84.3 %); median track length 11 frames

## validation

- PASS - every sequence has a done checkpoint: {'done': 95}
- PASS - per-sequence files match checkpoint hashes
- PASS - Phase 3 lineage unchanged (phase3_*_sha256)
- PASS - Phase 2 and Phase 3 sequence files unchanged (their own hashes)
- PASS - every tracked box exists in Phase 3 (video_id, t_ms, det_idx): 0 missing
- PASS - tracked boxes carry the Phase 3 class, confidence and coordinates: 0 differ
- PASS - every tracked frame is a Phase 3 frame
- PASS - no track has two boxes at one instant
- PASS - no box belongs to two tracks
- PASS - track ids are positive
- PASS - track_frame and is_birth follow time order
- PASS - one summary row per track: 4153 summary rows, 4153 tracks
- PASS - summary agrees with the box table (n_boxes, first/last time, majority class): 0 differ
- PASS - split matches manifest
- info: tracks 4153 (VRU 583), tracked boxes 61213 of 228700 stored; boxes with conf >= 0.5 in a track: 55042/66302 (83.0 %); median track length 10 frames

## test

- PASS - every sequence has a done checkpoint: {'done': 266}
- PASS - per-sequence files match checkpoint hashes
- PASS - Phase 3 lineage unchanged (phase3_*_sha256)
- PASS - Phase 2 and Phase 3 sequence files unchanged (their own hashes)
- PASS - every tracked box exists in Phase 3 (video_id, t_ms, det_idx): 0 missing
- PASS - tracked boxes carry the Phase 3 class, confidence and coordinates: 0 differ
- PASS - every tracked frame is a Phase 3 frame
- PASS - no track has two boxes at one instant
- PASS - no box belongs to two tracks
- PASS - track ids are positive
- PASS - track_frame and is_birth follow time order
- PASS - one summary row per track: 10633 summary rows, 10633 tracks
- PASS - summary agrees with the box table (n_boxes, first/last time, majority class): 0 differ
- PASS - split matches manifest
- info: tracks 10633 (VRU 1182), tracked boxes 164458 of 586465 stored; boxes with conf >= 0.5 in a track: 147875/174471 (84.8 %); median track length 11 frames

## Totals

- PASS - one config hash across all sequences: e08ed2dc6a9e7e22
- PASS - one Phase 3 config hash across all sequences: 09359d4da3a284ef
- PASS - one tracker setting across splits
- tracker settings: {"fuse_score": true, "keep_birth_box": true, "match_thresh": 0.9, "new_track_thresh": 0.5, "track_buffer_frames": 2, "track_buffer_sec": 0.4, "track_high_thresh": 0.25, "track_low_thresh": 0.1}
- tracks 47221; length (frames at 5 Hz): median 11, mean 15.2, share with 1-2 frames 10.2 %; tracks with a class change 11.5 %
- PASS - trained weights unchanged (SHA-256): 3/3 match

**Result: ALL CHECKS PASSED**
