"""
Rule-based risk score (baseline). Not fitted to any data: every constant below is fixed
a priori from simple driving intuition, so the score is a fair reference for the learned model.

    risk = speed_factor * (0.40 * proximity + 0.30 * approach) + 0.15 * attention + 0.15 * slowing

  speed_factor  speed / 50 km/h, clipped to [0, 1.5]           (missing GPS -> 0.5)
  proximity     max of
                  - in-path road user:  sqrt(area / 10% of the image), clipped to [0, 1.5]
                  - nearest VRU:        sqrt(area / 2% of the image) x centrality, clipped to [0, 1.5]
  approach      relative growth of the largest in-path object over 1 s / 50 %, clipped to [0, 1.5]
  attention     share of the attention map on VRUs and in-path objects, clipped to [0, 1]
  slowing       past GPS deceleration / 2 m/s^2, clipped to [0, 1]

Only the ranking matters for AUC / AP; the alarm threshold is chosen on the validation split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _col(df: pd.DataFrame, name: str, fill: float) -> np.ndarray:
    return df[name].astype(float).fillna(fill).to_numpy()


def rule_score(df: pd.DataFrame) -> np.ndarray:
    speed_f = np.clip(_col(df, "speed_kmh", 25.0) / 50.0, 0, 1.5)
    prox_lead = np.clip(np.sqrt(_col(df, "lead_area", 0) / 0.10), 0, 1.5)
    centrality = 1 - np.clip(_col(df, "vru_min_dx", 0.5) / 0.5, 0, 1)
    prox_vru = np.clip(np.sqrt(_col(df, "vru_max_area", 0) / 0.02) * centrality, 0, 1.5)
    proximity = np.maximum(prox_lead, prox_vru)
    approach = np.clip(_col(df, "lead_area_growth_l", 0) / 0.5, 0, 1.5)
    attention = np.clip(_col(df, "gaze_on_vru", 0) + _col(df, "gaze_on_inpath", 0), 0, 1)
    slowing = np.clip(-_col(df, "accel_past", 0) / 2.0, 0, 1)
    return speed_f * (0.40 * proximity + 0.30 * approach) + 0.15 * attention + 0.15 * slowing
