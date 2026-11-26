# Phase 3 verification - full

Sequences in scope: 1161 {'training': 800, 'test': 266, 'validation': 95}

## training

- PASS - every sequence has a done checkpoint: {'done': 800}
- PASS - per-sequence files match checkpoint hashes
- PASS - Phase 2 lineage unchanged (phase2_csv_sha256)
- PASS - frame_summary rows == Phase 2 rows (video_id, t_ms, cam_frame_idx): 42553 / 42553; missing 0, extra 0
- PASS - unique frame keys
- PASS - every box belongs to an analysed frame
- PASS - n_det agrees with the box table
- PASS - unique (video_id, t_ms, det_idx)
- PASS - class ids and names match the V2 class list
- PASS - confidence in [0.05, 1]: min 0.050
- PASS - boxes inside the image with positive size: 0 bad
- PASS - is_vru flag consistent with class
- PASS - split matches manifest
- info: frames 42553, boxes 1724959 (conf>=0.25: 821145), boxes/frame 40.5, frames with a VRU at conf>=0.25: 20814

## validation

- PASS - every sequence has a done checkpoint: {'done': 95}
- PASS - per-sequence files match checkpoint hashes
- PASS - Phase 2 lineage unchanged (phase2_csv_sha256)
- PASS - frame_summary rows == Phase 2 rows (video_id, t_ms, cam_frame_idx): 5330 / 5330; missing 0, extra 0
- PASS - unique frame keys
- PASS - every box belongs to an analysed frame
- PASS - n_det agrees with the box table
- PASS - unique (video_id, t_ms, det_idx)
- PASS - class ids and names match the V2 class list
- PASS - confidence in [0.05, 1]: min 0.050
- PASS - boxes inside the image with positive size: 0 bad
- PASS - is_vru flag consistent with class
- PASS - split matches manifest
- info: frames 5330, boxes 228700 (conf>=0.25: 106670), boxes/frame 42.9, frames with a VRU at conf>=0.25: 2828

## test

- PASS - every sequence has a done checkpoint: {'done': 266}
- PASS - per-sequence files match checkpoint hashes
- PASS - Phase 2 lineage unchanged (phase2_csv_sha256)
- PASS - frame_summary rows == Phase 2 rows (video_id, t_ms, cam_frame_idx): 14125 / 14125; missing 0, extra 0
- PASS - unique frame keys
- PASS - every box belongs to an analysed frame
- PASS - n_det agrees with the box table
- PASS - unique (video_id, t_ms, det_idx)
- PASS - class ids and names match the V2 class list
- PASS - confidence in [0.05, 1]: min 0.050
- PASS - boxes inside the image with positive size: 0 bad
- PASS - is_vru flag consistent with class
- PASS - split matches manifest
- info: frames 14125, boxes 586465 (conf>=0.25: 275325), boxes/frame 41.5, frames with a VRU at conf>=0.25: 6649

## Totals

- PASS - one config hash across all sequences: 09359d4da3a284ef
Boxes per class (descriptive; the operating threshold is chosen later on validation):

```
               conf>=0.05  conf>=0.25  conf>=0.50
cls_name                                         
pedestrian         240765       95902       54075
rider               11833        5334        2920
car               1445281      735458      509035
truck               61410       34387       23298
bus                 20292       11082        7558
motorcycle          10149        4989        3009
bicycle             14568        5413        3012
traffic light      309368      135367       63522
traffic sign       426458      175208       98809
```
- frames 62008, total 528.5 min (2.0 frames/s end-to-end; inference 509.2 min = 492.8 ms/frame)
- PASS - no bulk frame folders
- PASS - trained weights unchanged (SHA-256): 3/3 match

**Result: ALL CHECKS PASSED**
