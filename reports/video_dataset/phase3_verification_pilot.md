# Phase 3 verification - pilot

Sequences in scope: 11 {'test': 4, 'training': 4, 'validation': 3}

## training

- PASS - every sequence has a done checkpoint: {'done': 4}
- PASS - per-sequence files match checkpoint hashes
- PASS - Phase 2 lineage unchanged (phase2_csv_sha256)
- PASS - frame_summary rows == Phase 2 rows (video_id, t_ms, cam_frame_idx): 310 / 310; missing 0, extra 0
- PASS - unique frame keys
- PASS - every box belongs to an analysed frame
- PASS - n_det agrees with the box table
- PASS - unique (video_id, t_ms, det_idx)
- PASS - class ids and names match the V2 class list
- PASS - confidence in [0.05, 1]: min 0.050
- PASS - boxes inside the image with positive size: 0 bad
- PASS - is_vru flag consistent with class
- PASS - split matches manifest
- info: frames 310, boxes 10990 (conf>=0.25: 5538), boxes/frame 35.5, frames with a VRU at conf>=0.25: 131

## validation

- PASS - every sequence has a done checkpoint: {'done': 3}
- PASS - per-sequence files match checkpoint hashes
- PASS - Phase 2 lineage unchanged (phase2_csv_sha256)
- PASS - frame_summary rows == Phase 2 rows (video_id, t_ms, cam_frame_idx): 170 / 170; missing 0, extra 0
- PASS - unique frame keys
- PASS - every box belongs to an analysed frame
- PASS - n_det agrees with the box table
- PASS - unique (video_id, t_ms, det_idx)
- PASS - class ids and names match the V2 class list
- PASS - confidence in [0.05, 1]: min 0.050
- PASS - boxes inside the image with positive size: 0 bad
- PASS - is_vru flag consistent with class
- PASS - split matches manifest
- info: frames 170, boxes 6036 (conf>=0.25: 2911), boxes/frame 35.5, frames with a VRU at conf>=0.25: 50

## test

- PASS - every sequence has a done checkpoint: {'done': 4}
- PASS - per-sequence files match checkpoint hashes
- PASS - Phase 2 lineage unchanged (phase2_csv_sha256)
- PASS - frame_summary rows == Phase 2 rows (video_id, t_ms, cam_frame_idx): 200 / 200; missing 0, extra 0
- PASS - unique frame keys
- PASS - every box belongs to an analysed frame
- PASS - n_det agrees with the box table
- PASS - unique (video_id, t_ms, det_idx)
- PASS - class ids and names match the V2 class list
- PASS - confidence in [0.05, 1]: min 0.050
- PASS - boxes inside the image with positive size: 0 bad
- PASS - is_vru flag consistent with class
- PASS - split matches manifest
- info: frames 200, boxes 11506 (conf>=0.25: 5493), boxes/frame 57.5, frames with a VRU at conf>=0.25: 134

## Totals

- PASS - one config hash across all sequences: 09359d4da3a284ef
Boxes per class (descriptive; the operating threshold is chosen later on validation):

```
               conf>=0.05  conf>=0.25  conf>=0.50
cls_name                                         
pedestrian           1552         648         352
rider                  50          18           7
car                 17079        8715        5835
truck                 541         273         167
bus                   182         137         102
motorcycle             62          18           8
bicycle                85          29          15
traffic light        4086        2026         997
traffic sign         4895        2078        1295
```
- frames 680, total 1.8 min (6.4 frames/s end-to-end; inference 1.3 min = 116.1 ms/frame)
- PASS - no bulk frame folders
- PASS - trained weights unchanged (SHA-256): 3/3 match

**Result: ALL CHECKS PASSED**
