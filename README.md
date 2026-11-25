# DriveSight

Object detection for driving scenes on BDD100K, built for a risk-estimation use case in which recall on
**vulnerable road users (VRU-4: pedestrian, rider, motorcycle, bicycle)** matters more than overall accuracy.

**Start here:** [`notebooks/drivesight_v2/DriveSight_Object_Detection_Journey.ipynb`](notebooks/drivesight_v2/DriveSight_Object_Detection_Journey.ipynb)
is the single master report: dataset audit, V1 and V2 experiments, every decision with its evidence, and
(Section 17) the final three-model comparison. It is self-contained: every algorithm it uses (training, resume, evaluation,
bootstrap, speed benchmark) is defined in its own cells, so the repository needs no separate scripts.

## Objectives

1. Detect nine road-scene classes with a model small enough to train on a 4 GB laptop GPU (RTX 2050).
2. Maximise detection quality on the VRU-4 classes, which are rare and small.
3. Compare three architectures fairly (same data, same recipe, same evaluation) and select one.

## Dataset (V2)

BDD100K converted to YOLO format, 9 classes in this order: `pedestrian, rider, car, truck, bus, motorcycle,
bicycle, traffic light, traffic sign`.

| split | images | use |
|---|---|---|
| train | 2,744 | training |
| val | 686 | model selection and comparison |
| test | 830 | **protected**: not used for any selection in the three-model comparison |

The data is not part of this repository (`/data/` is git-ignored). Dataset definition: [`configs/v2_data.yaml`](configs/v2_data.yaml).
It is built by the notebooks `01_dataset_audit_eda` and `02_data_preprocessing_yolo_annotations` from the public BDD100K download.

## Models compared

| model | role | parameters | `best.pt` |
|---|---|---|---|
| YOLO11n (`v2_cls_pw025`) | existing V2 reference | 2.61 M | 5.6 MB |
| YOLOv8s | architecture from the friends' repo, retrained on V2 | 11.14 M | 22.6 MB |
| YOLO11s | architecture from the friends' repo, retrained on V2 | 9.43 M | 19.2 MB |

The architectures come from <https://github.com/arijkhemiri2002/DriveSight> (YOLOv8s, YOLO11s). That repository ships
no weights and trained on a 2,500-image subset at 640 px for 30 epochs, so both models were **retrained here with the V2
recipe**; the numbers in their repository are not comparable.

## V2 training configuration (identical for all three)

`imgsz 1152, batch 4, 150 epochs (patience 30), AdamW lr0 0.001, cls_pw 0.25 (class-weight exponent) with cls_remap,
no MixUp, seed 0, workers 0.` Full list: [`configs/v2_train_recipe.yaml`](configs/v2_train_recipe.yaml). A parity check in
Section 17 confirms that the saved `args.yaml` files differ only in model / run name / resume.
YOLOv8s ran 95 epochs (best epoch 65, early-stopped; resumed once after an external kill at epoch 44), YOLO11s 85 epochs
(best epoch 55, early-stopped).

## Evaluation method

One controlled evaluation of the three `best.pt` files on the **V2 validation split**: the same images, imgsz 1152,
NMS IoU 0.70, Ultralytics `val()` for official precision / recall / mAP50 / mAP50-95 and per-class AP, predictions cached so every
later number (confusion matrices, FP/FN, paired bootstrap of 2,000 image resamples) uses exactly the predictions behind the official
metrics. Checkpoints and split are hashed before and after. Speed: batch 1, 1152 px, FP32, 200 images after warm-up, RTX 2050.
Results live in `reports/drivesight_v2/evaluation/three_model_comparison/val/`.

## Results (V2 validation, 686 images)

| | YOLO11n | YOLOv8s | YOLO11s |
|---|---|---|---|
| mAP50-95 | 29.43 | 31.51 | **31.54** |
| mAP50 | 54.74 | **57.58** | 57.48 |
| Precision / recall | 70.5 / 48.4 | 71.8 / 51.9 | 66.5 / **53.0** |
| VRU-4 AP50-95 | 20.50 | **23.88** | 21.55 |
| VRU-4 recall (max-F1 point) | 38.8 | 43.8 | **46.2** |
| Latency, ms per image | **19.2** | 31.1 | 31.1 |

Paired 95 % bootstrap intervals: both "s" models beat YOLO11n on mAP50-95 (about +2.1 points, intervals exclude 0);
only YOLOv8s beats YOLO11n on VRU-4 AP50-95 (+3.4, [+0.8, +6.3]); YOLOv8s and YOLO11s are not distinguishable from each other.

### Selected model: **YOLOv8s**

It has the best VRU-4 AP50-95, the only VRU gain over the reference with an interval that excludes 0, and overall accuracy
level with YOLO11s. The price is about 1.6x the latency and 4x the size of YOLO11n, which remains the efficient option
(best mAP per millisecond). YOLO11s has the highest recall but is weak on motorcycles (12.1 AP50-95 vs 19.7). Caveats: one
training seed per model, validation used for both checkpoint choice and comparison, thin rare-class statistics (59 riders,
45 motorcycles, 74 bicycles in val). Weights: [`weights/final/`](weights/final/) (SHA-256 in `SHA256.txt`).

## Project structure

```text
notebooks/drivesight_v2/   master notebook + the two data notebooks (01, 02)
notebooks/                 07 resolution-ablation notebook, convert_to_yolo.py (V1 tooling the report cites)
app/                       demo web app (FastAPI server, photo page, video risk page)
configs/                   v2_data.yaml, v2_train_recipe.yaml
src/features/              dataset conversion & preprocessing utilities (BDD100K -> YOLO, BDDA video/gaze/GPS preprocessing)
src/risk/                  risk model: GPS braking labels, frame features, rule baseline, learned model (see its README)
weights/final/             the three final best.pt files; weights/*.pt pretrained starts are downloaded
reports/                   metrics, figures and written reports behind every notebook section (local only, git-ignored)
runs/drivesight_v2/        per-run args.yaml, results.csv (checkpoints and plots stay local)
data/                      local only (git-ignored): BDD100K dataset, not part of the repository
archive/                   local only (git-ignored): original scripts, docs, old runs and data not needed by the notebook
```

## Installation and execution

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                    # install a CUDA build of torch first for GPU use
# place the BDD100K data as described in notebooks 01/02 -> data/bdd100k/yolo_dataset/
jupyter lab notebooks/drivesight_v2/DriveSight_Object_Detection_Journey.ipynb
```

Running the notebook trains nothing: every training, evaluation or benchmark cell sits behind a `RUN_...` flag set to
`False` (`RUN_NEW_TRAINING`, `RUN_RESUME`, `RUN_TEST_EVAL`, `RUN_THREE_MODEL_TRAINING`, `RUN_THREE_MODEL_VAL_EVAL`,
`RUN_VAL_BOOTSTRAP`, `RUN_SPEED_BENCHMARK`), and saved results are read from `reports/`. To reproduce a step, set the flag
to `True` in the notebook and run that cell; evaluation outputs go to a new directory and never overwrite saved results.
Large caches (`predictions*.json`, `*.pkl`) are not in Git; the evaluation cell regenerates them.

## Demo app

A local web app to try the selected model: drop a road photo and get the detections back.

```bash
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000     # from the repository root
# then open http://127.0.0.1:8000
```

- Drag and drop, click or paste (`Ctrl+V`) a photo; vulnerable road users (pedestrian, rider, motorcycle, bicycle) get a glowing
  highlight and an alert card.
- Switch between **YOLOv8s** (selected), YOLO11s and YOLO11n, move the two confidence sliders (vulnerable road users default to 15%, other objects to 25%; a demo setting, not an evaluated profile), hide classes, hover a detection
  to find it in the image, compare with the original, and download the annotated image.
- **Video risk page** (`http://127.0.0.1:8000/video`, or the Photo / Video switch): drop a driving video (up to 60 s) and get a
  per-frame risk timeline (hard braking likely within 2 s) with the detections drawn on the video, an adjustable alarm threshold
  and the list of alerts. It uses the learned objects + motion model `data/risk/risk_model_vision.joblib` once
  `src/risk/train_risk_model.py` has produced it, and the rule-based score of `src/risk/rule_baseline.py` until then
  (see [`src/risk/README.md`](src/risk/README.md)). The video is written to a temporary file for decoding and deleted right after.
- Images are processed in memory and never saved; the server listens on `127.0.0.1` only. Weights come from `weights/final/`.
- It is a research demo, not a safety system. The detection settings are those of the V2 evaluation (imgsz 1152, NMS IoU 0.70).

## License

The code in this repository is released under the **GNU Affero General Public License v3.0** (see [`LICENSE`](LICENSE)).
It is the licence of [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (`ultralytics==8.4.189`, AGPL-3.0), on which the
training, evaluation and demo-app code depend; code that builds on it and is shared has to stay AGPL-3.0 compatible.
For use outside that licence, Ultralytics offers a separate enterprise licence.

- **Trained weights** (`weights/final/`) were trained on BDD100K and Ultralytics pretrained weights, and are shared for research and
  education only. Check the BDD100K terms before any other use.
- **The BDD100K data is not part of this repository** (`/data/` is git-ignored).

## Data attribution and credits

- **Dataset: BDD100K.** F. Yu, H. Chen, X. Wang, W. Xian, Y. Chen, F. Liu, V. Madhavan, T. Darrell, *BDD100K: A Diverse Driving Dataset
  for Heterogeneous Multitask Learning*, CVPR 2020 ([bdd-data.berkeley.edu](https://bdd-data.berkeley.edu)). The dataset is provided by
  the Berkeley DeepDrive consortium under its own terms, which restrict it to non-commercial research and education use; download it from the
  official site and re-confirm the current terms. The road-scene images shown in the notebook and in `reports/` are frames and crops of
  BDD100K and remain subject to those terms.
- **Models: Ultralytics.** YOLOv8s, YOLO11s and YOLO11n are Ultralytics architectures (AGPL-3.0).
- **Friends' repository.** The choice of the YOLOv8s and YOLO11s architectures and the reference training code printed in Section 17.2 of
  the notebook come from <https://github.com/arijkhemiri2002/DriveSight>; this project retrained both models with its own recipe.

## Rules the project follows

- The V2 test split stays closed for model selection; a one-shot confirmation needs an explicit decision.
- Existing checkpoints are never overwritten; evaluation outputs go to new directories.
- No V3 dataset or model is used in this repository.
