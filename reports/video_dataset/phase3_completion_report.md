# Phase 3 completion report - YOLOv8s detection on the Phase 2 frames

Date: 2026-11-26 | Detection schema 3.0 | Config hash `09359d4da3a284ef` | Status: **COMPLETE** (all verification checks passed; GPU-vs-CPU reproduction check: see §4)

## 1. Output

| Split | Sequences | Frames | Boxes (conf ≥ 0.05) | conf ≥ 0.25 | Frames with a VRU at conf ≥ 0.25 |
|---|---|---|---|---|---|
| training | 800 | 42,553 | 1,724,959 | 821,145 | 20,814 (48.9 %) |
| validation | 95 | 5,330 | 228,700 | 106,670 | 2,828 (53.1 %) |
| test | 266 | 14,125 | 586,465 | 275,325 | 6,649 (47.1 %) |
| **total** | **1,161** | **62,008** | **2,540,124** | **1,203,140** | 30,291 (48.8 %) |

- Model: `weights/final/yolov8s_v2_best.pt` (SHA-256 verified before and after the run), imgsz 1152, NMS IoU 0.70, conf floor 0.05, FP32, batch 4.
- Median 39 boxes per frame at conf ≥ 0.05 (max 170); 86 frames have no box.
- Per class at conf ≥ 0.25: car 735,458; traffic sign 175,208; traffic light 135,367; pedestrian 95,902; truck 34,387; bus 11,082; bicycle 5,413; rider 5,334; motorcycle 4,989.
- 83-87 % of sequences contain at least one VRU detection at conf ≥ 0.5. These are descriptive counts of detector output, not ground truth.

## 2. Verification (`phase3_verification_full.md`)

All checks passed:
- every sequence has a done checkpoint whose file hashes match, and its Phase 2 lineage (`phase2_csv_sha256`) is unchanged
- frame summary rows = Phase 2 rows exactly, on (video_id, t_ms, cam_frame_idx): 62,008 / 62,008
- every box belongs to an analysed frame; `n_det` agrees with the box table; (video_id, t_ms, det_idx) is unique
- class ids and names match the V2 list; conf in [0.05, 1]; all boxes inside the image with positive size
- one config hash across all sequences; each video in its manifest split; weights unchanged

**Repair** (`phase3_repair_degenerate_boxes.md`): the first verification found 1 box (training/47, t = 9.4 s, "traffic sign", conf 0.069) lying entirely below the image, which became zero-height once clipped. The cause was fixed in `box_table` (degenerate boxes are dropped). The stored box was removed, `det_idx` renumbered and the frame summary recomputed. A test shows that the repaired files are identical to the output of the fixed code, so no re-run was needed. The removal is recorded in the sequence checkpoint.

## 3. Hardware findings

| Finding | Evidence | Consequence |
|---|---|---|
| The run machine has an **NVIDIA MX550, 2 GB**, not the RTX 2050 named in `requirements.txt` | `nvidia-smi`, `_schema.json` | Hardware stated correctly in the docs; project `.venv` with torch 2.14.1+cu126 |
| **FP16 inference is broken on the MX550**: FP16 boxes match 0 % of FP32 boxes, and FP16 batch 1 vs batch 8 also disagree completely | `phase3_env_benchmark.md` | The run used **FP32** (fastest FP32 setting, batch 4). The benchmark now only allows settings that reproduce the FP32 batch-1 reference (≥ 99 %) |
| Typical speed: median **60 ms/frame** (benchmark: 89 ms/frame at FP32 batch 4) | per-sequence timings in the console log | Inference for the whole set takes about 1 h at normal speed |
| Wall time was 8.8 h because **16 % of sequences ran at > 200 ms/frame** (stalls, accounting for 7.6 of the 8.5 h of inference). They are consistent with the laptop sleeping or throttling overnight | console log timestamps | Results are unaffected; for long runs, keep the laptop plugged in and prevent sleep |
| On the CPU (torch 2.14.1+cpu), inference took 564 ms/frame | first benchmark | Kept as a fallback (~10 h, resumable) |

## 4. Is the stored output correct despite the FP16 problem?

- QC images (77 sequences, boxes at conf ≥ 0.25) show cars, buses, riders, pedestrians, traffic lights and signs in the right places, with occasional class confusions (e.g. motorcycle/bicycle).
- Numeric check: `check_stored_vs_cpu.py` re-runs every 5th frame of 6 sequences (2 per split) on the CPU in FP32 and compares the boxes with the stored GPU output. Result: *to be filled in from `phase3_gpu_vs_cpu_check.md`.*
- Reference boxes found in stored detections: 100.00% (1249/1249)
- Stored boxes found in CPU reference: 100.00% (1249/1249)
- Validation: PASS
- Frames tested: 64 across 6 sequences
- Reference: CPU FP32

## 5. Limitations carried forward

- Detector recall on VRUs is limited (V2 validation: VRU-4 recall 43.8 %, AP50-95 23.88). Missing detections are expected, and tracking can only bridge short gaps.
- There are no ground-truth boxes on BDDA, so detection quality on these videos is assessed qualitatively. A small manual annotation is planned for Phase 7.
- BDD100K (training images) and BDD-A (videos) come from the same Berkeley DeepDrive collection, so some overlap of drives is possible.
- `bottom_y_n` and `h_n` are ordinal distance proxies, not distances.
- The operating confidence threshold is not chosen yet. It will be chosen on the validation split in Phase 4/5, never on test.
