# Phase 5 completion report - fusion (tracks + gaze + GPS) and risk model

Date: 2026-11-26 | Fusion schema 5.0 | Phase 4 config hash `e08ed2dc6a9e7e22` | Phase 2 schema 2.1 |
Status: **fusion and validation COMPLETE** (all verification checks passed); **test split not scored yet**,
on purpose (one shot, at the very end).

## 1. Output

| Split | Sequences | Samples | In scope | Positive | Hard-braking clips | Onset exactly 5.5 s | Object rows | Fusion hash |
|---|---|---|---|---|---|---|---|---|
| training | 800 | 42,553 | 22,923 | 5,577 (24.3 %) | 564 (70.5 %) | 67.7 % | 490,988 | `1adc58e8a353e7e6` |
| validation | 95 | 5,330 | 2,803 | 710 (25.3 %) | 71 (74.7 %) | 60.6 % | 61,213 | `a6b78328948ffc7b` |
| test | 266 | 14,125 | 7,614 | 1,862 (24.5 %) | 188 (70.7 %) | 64.9 % | 164,458 | `6cd6d3c6cf5d1827` |

- Label: GPS deceleration ≥ 2.0 m/s² (speed before ≥ 1.5 m/s); positive if the onset is within 2 s. Median
  peak deceleration 3.1–3.4 m/s². Clips without a hard braking are negatives.
- 45 model features: objects 18, motion 10, attention 13, GPS 4. None of them reveals the clip time (§3 of
  `docs/phase5_fusion.md`).
- Run time: fusion about 80 s for 1,161 sequences; training about 90 s (CPU).

## 2. Verification (`phase5_verification.md`)

All 57 checks passed:

- the Phase 2 and Phase 4 inputs are unchanged since fusion, the outputs are intact, and the fusion hash
  matches the settings;
- one frame row per Phase 2 row, with unique keys;
- one object row per tracked box, with the same Phase 4 track ids;
- braking events recomputed with separate code, all identical;
- `y`, `lead_sec` and `in_scope` are consistent with the onset, the horizon and the warm-up;
- **no look-ahead**: 15 sequences per split were re-fused on inputs cut at a random time, and every feature
  before the cut was identical;
- no clip-time column among the features;
- value ranges are correct;
- the splits are separate.

Tests: `tests/test_phase5_fusion.py`, 15 tests. Full suite: **57 passed**.

The tests cover:

- labels, causal GPS, object motion, attention and confidence filtering;
- the same-time AUC;
- end-to-end fusion, verification and resume;
- the verifier failing on a tampered output, a changed input and a deliberately leaking feature;
- the guard against writing into upstream folders;
- training with the one-shot test lock;
- the demo app using the new model (skipped when the front-end integration is absent).

## 3. Risk model, validation (`phase5_model_validation.md`)

Model choice uses the **same-time AUC**: ROC-AUC among samples taken at the same clip time in different
clips. A time-only model scores 0.5 on it. Environment: `requirements.txt` (scikit-learn 1.7.2).

| model | same-time AUC [95 % CI] | AP |
|---|---|---|
| **boosting, all features (selected: lr 0.03, 31 leaves, l2 1.0)** | **0.744** [0.664, 0.817] | 0.519 |
| logistic, all features | 0.695 | 0.406 |
| boosting, vision only (demo app) | 0.628 [0.557, 0.700] | 0.369 |
| rule baseline | 0.578 [0.478, 0.682] | 0.362 |
| speed-only (reference) | 0.730 | 0.377 |
| time-only (reference) | 0.500 | 0.523 |

Ablation (same-time AUC):

| features | same-time AUC |
|---|---|
| objects | 0.593 |
| + motion (vision) | 0.628 |
| + attention | 0.635 |
| GPS only | 0.741 |
| all − attention | 0.738 |
| all − GPS | 0.635 |
| all | 0.744 |

By speed floor, same-time AUC:

| speed | fused model | vision only | speed-only |
|---|---|---|---|
| ≥ 15 km/h | 0.700 | 0.614 | 0.681 |
| ≥ 25 km/h | 0.614 | 0.588 | 0.495 |

Conclusions:

- The fused model is better than the rule. Bootstrap difference: [+0.056, +0.279].
- Most of its signal comes from the GPS (speed, and a deceleration that has already started).
- It keeps a signal beyond speed: 0.614 at ≥ 25 km/h, where speed alone is at chance.
- Vision alone is weak but above chance.
- The attention proxy adds little.
- Clip-level alarms at the F1 threshold come early (median 3.9 s before the onset), and 79 % of the clips
  without hard braking also raise one. The score is a risk ranking, not a reliable imminent-braking alarm.

Reproducibility: the tracks and the fusion output (same values, compared column by column; the parquet file hashes differ only through library metadata), the labels, the rule, the logistic model and the references are
identical on two machines. Only gradient boosting moves with the scikit-learn version (1.9.1: 0.748, vision
0.607), so the pinned versions of `requirements.txt` are the reference.

## 4. Fixes made during the phase

- **Clip-time leak found and removed.** `gps_age_sec` (time since the last 1 Hz record) revealed the clip
  time within each second, because records and onsets both fall at x.5 s. On its own it reached ROC-AUC 0.61.
  It is now stored but not used as a feature, and the verifier and tests check this.
- **Time bins.** Bins of 1 s still let the time-only model reach 0.61, because the label boundary at 3.5 s
  falls inside a bin. The metric now groups samples by the exact 200 ms sample time, and time-only gives
  exactly 0.500.
- **Speed artefact made visible.** A deceleration of 2 m/s² cannot happen at very low speed. A `speed-only`
  reference and the speed-floor table now separate this physical effect from real prediction.

## 5. Demo app (not in this branch)

`src/fusion/live.py` applies the same chain as BDDA to an arbitrary video: YOLOv8s detections, ByteTrack with
the frozen Phase 4 setting, then the Phase 5 vision features. The model to use is
`data/risk/risk_model_vision_v5.joblib`. Wiring it into the front-end (`app/`) is kept outside this branch;
the related test (`test_demo_app_uses_the_phase5_vision_model`) is skipped when the app does not have that
integration.

## 6. Remaining

- **Score the test split once**, when nothing else will change: `python -m src.fusion.train_risk --final-test`.
- Optional: re-decode the full gaze maps (needs the BDDA gaze videos) to measure the attention inside each
  box, instead of the centroid kernel.
- `src/risk/` steps 2–4 (old format) are superseded by `src/fusion/`.
