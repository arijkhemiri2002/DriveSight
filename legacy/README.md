# legacy/

Superseded scripts, kept for traceability. Do not run them.

| File | Was | Replaced by | Why |
|---|---|---|---|
| `bdd100k_to_yolo_v1_preprocessing.py` | `src/features/preprocessing.py` | `src/features/convert_bdd100k_to_yolo.py` | First BDD100K→YOLO conversion. Hardcoded absolute paths; uses `shutil.move`, `os.remove` and `rmtree` on `data/yolo/`. Its name suggested video preprocessing. |
| `bdd100k_to_yolo_v1_verification.py` | `src/features/verification.py` | (none needed for Phase 2) | Checks for the conversion above; hardcoded absolute paths. |
| `video_preprocessing_pilot_v1.py` | `src/features/video_preprocessing.py` | `src/features/run_preprocessing.py` | Phase 2 pilot orchestrator: saved JPEG frames, no checkpoint/resume, last sample past the end of the camera video, no GPS validity checks. It depends on functions that were removed from `frame_extractor.py`, so it no longer runs. |
