# Phase 4 verification - pilot

Sequences in scope: 11 {'test': 4, 'training': 4, 'validation': 3}

## training

- PASS - every sequence has a done checkpoint: {'done': 4}
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
- PASS - one summary row per track: 204 summary rows, 204 tracks
- PASS - summary agrees with the box table (n_boxes, first/last time, majority class): 0 differ
- PASS - split matches manifest
- info: tracks 204 (VRU 11), tracked boxes 3305 of 10990 stored; boxes with conf >= 0.5 in a track: 3043/3646 (83.5 %); median track length 11 frames

## validation

- PASS - every sequence has a done checkpoint: {'done': 3}
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
- PASS - one summary row per track: 109 summary rows, 109 tracks
- PASS - summary agrees with the box table (n_boxes, first/last time, majority class): 0 differ
- PASS - split matches manifest
- info: tracks 109 (VRU 11), tracked boxes 1774 of 6036 stored; boxes with conf >= 0.5 in a track: 1623/1894 (85.7 %); median track length 12 frames

## test

- PASS - every sequence has a done checkpoint: {'done': 4}
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
- PASS - one summary row per track: 185 summary rows, 185 tracks
- PASS - summary agrees with the box table (n_boxes, first/last time, majority class): 0 differ
- PASS - split matches manifest
- info: tracks 185 (VRU 17), tracked boxes 3109 of 11506 stored; boxes with conf >= 0.5 in a track: 2779/3238 (85.8 %); median track length 13 frames

## Totals

- PASS - one config hash across all sequences: e08ed2dc6a9e7e22
- PASS - one Phase 3 config hash across all sequences: 09359d4da3a284ef
- PASS - one tracker setting across splits
- tracker settings: {"fuse_score": true, "keep_birth_box": true, "match_thresh": 0.9, "new_track_thresh": 0.5, "track_buffer_frames": 2, "track_buffer_sec": 0.4, "track_high_thresh": 0.25, "track_low_thresh": 0.1}
- tracks 498; length (frames at 5 Hz): median 12, mean 16.4, share with 1-2 frames 9.4 %; tracks with a class change 11.0 %
- PASS - trained weights unchanged (SHA-256): 3/3 match

**Result: ALL CHECKS PASSED**
