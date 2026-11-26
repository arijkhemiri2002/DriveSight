# Phase 3 environment and benchmark

- Python 3.12.0 on Windows-11-10.0.26200-SP0
- torch 2.14.1+cu126 (CUDA build 12.6), CUDA available: True
- ultralytics 8.4.189
- GPU: NVIDIA GeForce MX550, 2.0 GiB
- weights yolov8s_v2_best.pt: SHA-256 0dde23367cd2f566... matches weights/final/SHA256.txt
- benchmark frames: 40 from test/6

| precision | batch | ms/frame | peak GPU memory (GiB) |
|---|---|---|---|
| FP16 | 1 | 143.6 | 0.06 |
| FP16 | 4 | 131.4 | 0.18 |
| FP16 | 8 | 136.1 | 0.33 |
| FP16 | 16 | 212.6 | 0.65 |
| FP32 | 1 | 326.6 | 0.13 |
| FP32 | 4 | 88.9 | 0.33 |
| FP32 | 8 | 101.3 | 0.59 |
| FP32 | 16 | 175.6 | 1.23 |

- batch 1 vs batch 8 (FP16): 0.0% of 5015 boxes (conf>=0.25) identical (IoU >= 0.99, same class)
- FP16 vs FP32: 0.0% of 1063 FP32 boxes (conf>=0.25) matched by FP16 (IoU >= 0.9, same class)
- fastest: FP32 batch 4 = 88.9 ms/frame -> inference for 62,008 frames ~ 92 min (+ video decoding, ~10 min)
