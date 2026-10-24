# DriveSight
Automatic estimation of the risk level of a road scene from images/videos and their annotations, built on the BDD100K dataset.

Project Idea

Driving images (or video sequences) are analyzed to detect the objects present (vehicles, pedestrians, cyclists, traffic lights, etc.). These detections are then combined with contextual information from the labels (weather, time of day, scene type) to produce a risk level: low, medium, or high.

Dataset: http://bdd-data.berkeley.edu/download.html 

 BDD100K is a large and highly diverse driving dataset (100,000 videos).

| Information | Use in the project |
|---|---|
| Bounding boxes (objects) | Detection training, object counting and positioning |
| Scene attributes: `weather` (clear, rainy, snowy, foggy...) | Contextual risk factor |
| Scene attributes: `timeofday` (daytime, night, dawn/dusk) | Risk factor (visibility) |
| Scene attributes: `scene` (highway, city street, residential...) | Risk factor (environment) |
| Occlusion and truncation attributes | Detection reliability |
| Drivable areas, lanes | Position of objects relative to the road |

## Pipeline

```
Image / Video  →  1. Detection (YOLO)  →  2. Feature extraction  →  3. Risk estimation  →  Risk level
                                                    ↑
                                           BDD100K labels
                                  (weather, time, scene, occlusion)
```

### Step 1: Object Detection
- **YOLO** is used as the base model (fine-tuned on BDD100K).
- Detected classes: car, truck, bus, pedestrian, cyclist, motorcycle, traffic light, traffic sign, etc.
- Output: class, position, and confidence for each object.

### Step 2: Scene Feature Extraction
From the detections and labels:
- Number of objects per class (scene density)
- Presence and proximity of vulnerable road users (pedestrians, cyclists)
- Bounding box size (proxy for distance to the object)
- Position in the image (central area = ego vehicle's path)
- Conditions: weather, time of day, scene type
- Occlusion rate

### Step 3: Risk Estimation
- Computation of a **risk score** combining the features above.
- Two possible approaches:
  - **Rule-based**: weighted score (simple, interpretable baseline)
  - **Learned**: classification model (Random Forest, XGBoost, small MLP) trained on the extracted features
- Output: risk level (low / medium / high).

### Step 4: Evaluation and Visualization
- Detection metrics: mAP, precision, recall
- Risk classification metrics: accuracy, F1-score, confusion matrix
- Bounding boxes and risk level overlaid on the image/video

## Repository Structure (proposed)

```
├── data/              # BDD100K dataset (not versioned)
├── notebooks/         # Exploration and experiments
├── src/
│   ├── detection/     # YOLO training and inference
│   ├── features/      # Feature extraction
│   ├── risk/          # Risk computation / model
│   └── visualize.py   # Results display
├── configs/
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/arijkhemiri2002/DriveSight.git
cd DriveSight
pip install -r requirements.txt
```

## Usage

```bash
# Train the detector
python src/detection/train.py --data configs/bdd100k.yaml

# Estimate risk on an image or video
python src/predict.py --source path/to/video.mp4
```

## Future Work
- Temporal reasoning (object tracking, relative speed, time-to-collision)
- Integration of depth or LiDAR data
- Risk estimation on sequences rather than single frames
