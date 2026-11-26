# Phase 5: fusion and risk model

## 1. Purpose

Join what the earlier phases produced for every 200 ms sample: the tracked road users (Phase 4), the
attention map descriptors and the GPS (Phase 2). Then learn a risk score: **will the driver brake hard
within the next 2 s?** The label comes from the GPS, so no manual annotation is needed.

Inputs: the verified `aligned_records.csv` (Phase 2, schema 2.1) and `tracks.parquet` (Phase 4,
schema 4.0) of each split. No video is decoded and no GPU is needed. The full fusion takes about 80 s
for the 1,161 sequences, and training takes about 90 s.

## 2. Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Label | Hard braking = consecutive 1 Hz GPS records with deceleration ≥ 2.0 m/s² and speed before ≥ 1.5 m/s. The strongest event is kept; its start is the onset. A sample is positive if `0 ≤ onset − t ≤ 2 s`. | Same rule as the earlier `src/risk`, rebuilt on the Phase 2 rows (distinct `gps_rec_t_sec`), so the raw BDDA folder is not needed. |
| Samples in scope | From the end of the warm-up (1.4 s) to the onset. Samples after the onset are excluded. Clips without a hard braking are negatives over their whole length (`no_event_policy = negative`). | After the onset the braking is already visible. The warm-up is where motion and gaze shifts are undefined. Clips without hard braking are mild braking events. See §3 for why they are needed. |
| GPS features | Rebuilt 1 Hz records with `t_rec ≤ t` only: last speed, last acceleration, 2 s speed trend. | Phase 2's nearest record can be up to 0.5 s **in the future**; those columns are never features. |
| `gps_age_sec` | Stored, **not** a feature | GPS records fall at x.5 s and so do the onsets. The age gives away the clip time within each second; on its own it reached ROC-AUC 0.61 on validation. |
| Object motion | For each track, the box of the **same track** 0.6–1.4 s earlier: log growth rate of the box height (1/s), contact-time proxy = 1/growth (capped at 10 s), lateral closing speed, bottom-edge speed. | Uses the Phase 4 identities. Heights are image proxies, not metres. |
| Ego path | A fixed corridor in the image (bottom edge ≥ 0.45 of the height, half-width 0.06 → 0.28). "Near path" widens it by 0.10 on each side. | There is no lane detection. Same corridor as the earlier `src/risk`. |
| Counted boxes | A tracked box counts at time t if conf ≥ 0.15 (VRU) or ≥ 0.25 (others). Its history uses all of its tracked boxes. | Same thresholds as the demo app. |
| Attention on an object | Gaussian of the distance from the gaze-map centroid to the box (0 inside), with σ = max(gaze spread, 0.05), in image-height units. | Phase 2 keeps the centroid, peak and spread, not the full map. This approximates "how much attention lands on the object" without decoding the maps again. |
| Threat (a priori) | path weight (1 / 0.5 / 0.15) × closeness (height ÷ 0.25, or ÷ 0.15 for a VRU) × (0.5 + 0.5 × growth ÷ 0.5) × 1.3 for a VRU. Used for `max_threat`, `unattended_threat` and the rule baseline. | Fixed constants, not fitted. |
| Clip time | Never a feature. Track age is capped at 1 s. | See §3. |
| Model choice | Logistic regression and gradient boosting (sklearn), class-balanced. Hyper-parameters and model are chosen on **validation same-time AUC**; the alarm threshold is the best F1 on validation. | The test split is scored once at the end (`--final-test`); a second run is refused unless `--allow-retest`. |
| Resume / lineage | One run per split. `_schema.json` holds the fusion hash (settings + Phase 2/4 hashes + input SHA-256) and output SHA-256. An identical run is skipped; any change recomputes. | The run is short, so per-sequence checkpoints are not needed. |

## 3. The BDDA shortcut, and how it is handled

BDDA clips are cut at a fixed offset from the braking event. Results on our data:

- **two thirds of the onsets are exactly at 5.5 s of clip time**;
- 82–93 % of the onsets fall between 4.5 and 8.5 s.

Clip time alone therefore "predicts" the label: a model that only sees the time reaches AP 0.52 on
validation. A model fed with any time-revealing feature would learn the cut, not the road.

Three measures handle this:

1. **No time feature.** Clip time is never an input. Track age is capped, the warm-up is excluded, and
   `gps_age_sec` is removed.
2. **Contrast at the same clip time.** Clips without a hard braking stay in as negatives.
3. **Same-time AUC as the main metric.** It is the ROC-AUC among samples taken at the same 200 ms
   sample time in different clips, averaged over sample times. The `time-only` reference scores
   exactly 0.5 on it. Model selection uses it.

A second effect is physical. A deceleration of 2 m/s² cannot happen at very low speed, so speed alone
separates many samples (same-time AUC 0.73). The report adds a `speed-only` reference and a table by
speed floor, to show what each model adds beyond speed.

The verifier checks that the features do not read the future. It re-runs the fusion on inputs cut at a
random time T and requires every feature at t ≤ T to be identical (§5).

## 4. Output layout

```
data/processed/{split}/fusion/
├── frame_features.parquet    one row per Phase 2 row: keys, 45 features, gps_age_sec, label columns
├── object_features.parquet   one row per tracked box: in_path, age, scale_rate, ttc, attention, threat, ...
├── braking_events.csv        one row per sequence: has_event, onset_sec, peak_decel, ...
└── _schema.json              fusion hash, upstream hashes, settings, coverage, output SHA-256
data/risk/
├── risk_model_fusion.joblib     selected model on all 45 features (+ threshold, hashes)
└── risk_model_vision_v5.joblib  same family on the 28 vision features (used by the demo app)
reports/video_dataset/
├── phase5_verification.md
├── phase5_model_validation.md / .json, figures/phase5_*.png
└── phase5_model_test.md / .json  (after --final-test only)
```

Feature groups (`src/fusion/fusion_core.py`):

- **objects (18):** counts; lead road user in the path (height, bottom edge, VRU or not, age); nearest
  VRU (height, bottom edge, lateral offset, near path, confidence); `max_threat`.
- **motion (10):** growth rate and contact time of the lead and of the nearest VRU; VRU lateral closing
  speed; shortest contact time in and near the path; fastest growth near the path; road users that
  appeared near the path in the last second.
- **attention (13):** gaze spread, peak, distance of the centroid from the centre, active area, number
  of regions, 0.2 s and 1 s shifts; attention on the lead, on the nearest VRU and on the most looked-at
  road user; `unattended_threat`; share of threat that is looked at.
- **GPS (4):** speed, last acceleration, 2 s speed trend, GPS valid.

## 5. Verification

`verify_fusion.py` checks the following and prints ALL CHECKS PASSED:

- the Phase 2 and Phase 4 inputs are unchanged, the outputs are intact, and the fusion hash still
  matches the settings;
- coverage: one frame row per Phase 2 row, and one object row per tracked box with the same track ids;
- labels: braking events recomputed with separate code, and consistency of `y`, `lead_sec` and
  `in_scope`;
- **no look-ahead** (15 sequences per split, cut at a random time);
- no time column among the features;
- value ranges;
- split separation.

The tests check that the verifier does fail on a tampered output, on a changed input and on a
deliberately leaking feature.

## 6. How to run

```powershell
python -m src.fusion.run_fusion            # about 80 s, skips splits that are up to date
python -m src.fusion.verify_fusion         # ALL CHECKS PASSED
python -m src.fusion.train_risk            # validation report + models
python -m src.fusion.train_risk --final-test   # ONCE, when everything is frozen
python -m pytest tests -q
```

Or `powershell -ExecutionPolicy Bypass -File scripts\phase5_run.ps1`, which runs fusion, verification
and training, without the test step.

## 7. Results on validation (95 clips, 2,803 samples, 25.3 % positive)

Numbers from the project environment (`requirements.txt`, scikit-learn 1.7.2).

| model | same-time AUC [95 % CI] | AP | ROC-AUC |
|---|---|---|---|
| **boosting, all features (selected: lr 0.03, 31 leaves, l2 1.0)** | **0.744** [0.664, 0.817] | 0.519 | 0.790 |
| logistic, all features | 0.695 | 0.406 | 0.745 |
| boosting, vision only (demo app) | 0.628 [0.557, 0.700] | 0.369 | 0.629 |
| rule baseline (not fitted) | 0.578 [0.478, 0.682] | 0.362 | 0.644 |
| speed-only (reference) | 0.730 | 0.377 | 0.723 |
| time-only (reference) | 0.500 | 0.523 | 0.812 |

Boosting minus rule, same-time AUC: [+0.056, +0.279] (bootstrap over clips, 1,000 resamples).

What this says:

- **The fused model beats the rule** at equal clip time.
- **Most of the signal is in the GPS.** The car is often already slowing before the hard braking. GPS
  alone reaches 0.741, and removing the GPS drops the model to 0.635.
- **Above 25 km/h**, speed alone is at chance (0.495), while the fused model keeps 0.614 and vision
  alone 0.588. The model therefore uses more than speed.
- **Vision alone is weak but above chance** (0.628, CI above 0.5). It does not differ significantly
  from the rule.
- **Attention adds little** on top of vision (0.628 → 0.635) or on top of everything else
  (0.738 → 0.744, within the noise). This is expected with a centroid-based proxy and an in-lab
  attention reference.
- **Clip-level alarms at the F1 threshold mostly come early** (median 3.9 s before the onset), and
  79 % of the clips without hard braking also raise an alarm. As an imminent-braking alarm the score
  is not usable on its own; it is a ranking of risk.

**Reproducibility.** The tracks and the fusion output (same values on two machines, compared column by column; parquet file hashes differ only through library metadata), the labels, the rule, the logistic model and the references are
identical across machines. Gradient boosting changes slightly with the scikit-learn version: with
1.9.1 the validation gave 0.748 (another grid point), and vision 0.607. Keep the pinned versions to
reproduce the numbers above.

## 8. Limitations

- The clips are braking events, so the base rates are not those of normal driving.
- The 1 Hz GPS places onsets to about ±0.5 s.
- The gaze map is an in-lab reference (average of observers), not the gaze of the person driving; the
  attention on an object comes from the centroid and spread only.
- The ego path is a fixed corridor; sizes and contact times are image proxies.
- Validation has only 95 clips, so the CIs are wide. The test split (266 clips) is the unbiased estimate.
- A possible improvement: re-decode the full gaze maps (needs the BDDA gaze videos) to measure the
  attention mass inside each box, instead of the centroid kernel.
