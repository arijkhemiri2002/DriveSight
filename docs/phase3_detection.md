# Phase 3 — YOLO detection on the Phase 2 frames

Status: **complete**. 1,161 sequences, 62,008 frames, 2,540,124 boxes, all verification checks passed. See `reports/video_dataset/phase3_completion_report.md`.

## 1. What it does

For each of the 1,161 sequences, `src/detection/run_detection.py`:
1. reads the frames to analyse from the verified Phase 2 output (`sequence_records/{video_id}.csv`: `sample_idx`, `t_ms`, `cam_frame_idx`)
2. decodes the camera video with the same `SequentialReader` (no frame is written to disk)
3. runs the selected **YOLOv8s** V2 checkpoint in batches
4. stores every box with conf ≥ 0.05, plus one summary row per frame, including frames with no box

Every detection therefore joins to a Phase 2 row by (`video_id`, `t_ms`). One function (`frame_extractor.sample_plan`) defines which frame belongs to which time stamp in both phases.

## 2. Settings and why

| Setting | Value | Reason |
|---|---|---|
| Model | `weights/final/yolov8s_v2_best.pt` (read-only; SHA-256 checked against `SHA256.txt` before every run) | Best VRU-4 AP50-95 (23.88) in the V2 comparison |
| Input size | 1152 | training and validation resolution (`configs/v2_train_recipe.yaml`) |
| NMS IoU | 0.70 | V2 evaluation / app setting |
| Confidence floor | 0.05 | storage floor, not the operating threshold. The threshold is chosen later on the **validation** split only, and a ByteTrack-style tracker (Phase 4) uses low-score boxes in its second association step. |
| Precision, batch | FP32, batch 4 (run); chosen by `check_detection_env.py` as the fastest setting that reproduces the FP32 batch-1 reference (≥ 99 % of boxes) | FP16 is broken on the MX550: 0 % agreement with FP32 |
| Batch | 8 (speed only) | batch 1 and batch 8 give identical boxes (tested) |
| Class order | checked at load time | a checkpoint with another class order is refused |

The config hash includes the weights' SHA-256, every setting above, the Ultralytics version, the device type and the **Phase 2 config hash**. Re-running Phase 2 with other settings therefore invalidates the detections automatically. Each checkpoint also records the SHA-256 of the Phase 2 CSV it read (lineage).

## 3. Outputs (per split, under `data/processed/{split}/`)

```
detections/
├── detections.parquet          one row per box (merged)
├── frame_summary.csv           one row per Phase 2 row: n_det, n_vru, max_conf, max_conf_vru
├── _schema.json                config, hash, environment (torch/CUDA/GPU/ultralytics), coverage
├── sequence_detections/{video_id}.parquet, {video_id}_frames.csv
└── sequence_checkpoints/{video_id}.json      status, hashes, lineage, timings, counts per class
qc/detections/{video_id}_t{ms}.jpg            boxes (conf >= 0.25) on the middle frame, ~5 % of sequences
```

Box columns: `split, video_id, sample_idx, t_ms, det_idx` (0 = most confident), `cls_id, cls_name, is_vru, conf`, pixel box `x1 y1 x2 y2` (clipped to 1280×720), and normalised geometry: `cx_n, cy_n, w_n, h_n, area_frac, aspect_wh, bottom_y_n, x_offset_n, touches_border`.

## 4. What the geometric columns mean, and do not mean

- `bottom_y_n` (box bottom) and `h_n` (box height) are **ordinal distance proxies**: on a roughly flat road, closer objects reach lower in the image and look taller. There is no camera calibration, the mounting differs per vehicle, slopes and pitch change the picture, and real heights vary by class and instance. They can rank objects within a clip and class; they are not distances in metres.
- `x_offset_n` is the offset from the image centre, not from the ego lane.
- `touches_border`: the box is cut by the image edge, so its size and bottom are unreliable.

## 5. Known limitations

- **Detector recall**: on the V2 validation set, VRU-4 recall was 43.8 % (AP50-95 23.88). A large share of pedestrians and cyclists will be missed in single frames. Phase 4 tracking can bridge short gaps; it cannot recover objects that are never detected. Any risk result has to state this.
- **No ground-truth boxes on BDDA**: detection quality on these videos is not measured directly. The V2 metrics come from BDD100K images, which are from the same crowd-sourced dashcam source and the same 1280×720 resolution. They are video keyframes, though, and BDDA frames are decoded video, which can be more compressed or blurred. The QC images give a qualitative check; a small manual annotation of BDDA frames would be needed for a quantitative one (Phase 7).
- **Possible overlap**: BDD100K and BDD-A are both drawn from the Berkeley DeepDrive video collection. If some YOLO training images come from the same drives as BDDA clips, detection on those clips may look better than it would on unseen data. This does not affect the risk evaluation design, but it should be mentioned.

## 6. Hardware and environment

- Run machine: NVIDIA GeForce MX550 (Turing, 2 GB), driver CUDA 13.3. The comment in `requirements.txt` (RTX 2050, 4 GB) describes the machine the models were trained on, not this one.
- The global Python had a CPU-only torch (2.14.1+cpu): CPU inference measured 564 ms/frame, about 10 h for 62,008 frames. A project `.venv` with a CUDA 12.6 build of torch is created by `scripts/setup_gpu_env.ps1`, and `pip freeze` is saved to `reports/video_dataset/venv_pip_freeze.txt`.

## 7. How to run

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_gpu_env.ps1   # once: .venv with CUDA torch
powershell -ExecutionPolicy Bypass -File scripts\phase3_run.ps1      # tests, env check, pilot, full run, packing
python -m src.detection.check_detection_env                          # environment + speed/precision benchmark
python -m src.detection.run_detection --pilot --qc-fraction 1.0
python -m src.detection.verify_detection --pilot
python -m src.detection.run_detection                                # all sequences, resumable
python -m src.detection.verify_detection
```
