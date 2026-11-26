"""
Rule-based risk score on the Phase 5 frame features (baseline, NOT fitted: every constant is fixed a
priori), so it is a fair reference for the learned models.

    risk = speed_f * (0.45 * max_threat + 0.30 * contact) + 0.15 * slowing + 0.10 * unattended

  speed_f     speed / 50 km/h clipped to [0, 1.5]; no GPS -> 0.5
  max_threat  largest a-priori object threat (fusion_core.threat_score), clipped to [0, 1]
  contact     (4 s - shortest time-to-contact proxy of a road user near the ego path) / 4 s, in [0, 1]
  slowing     past GPS deceleration / 2 m/s^2, clipped to [0, 1]
  unattended  largest threat x (1 - attention on it), clipped to [0, 1]

Only the ranking matters for AUC / AP; the alarm threshold is chosen on validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _col(df: pd.DataFrame, name: str, fill: float) -> np.ndarray:
    if name not in df.columns:
        return np.full(len(df), fill, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").astype(float).fillna(fill).to_numpy()


def rule_score(df: pd.DataFrame, speed_kmh: float | None = None) -> np.ndarray:
    speed = _col(df, "speed_kmh", 25.0) if speed_kmh is None else np.full(len(df), float(speed_kmh))
    speed_f = np.clip(speed / 50.0, 0, 1.5)
    threat = np.clip(_col(df, "max_threat", 0.0), 0, 1)
    contact = np.clip((4.0 - _col(df, "min_ttc_near", 10.0)) / 4.0, 0, 1)
    slowing = np.clip(-_col(df, "accel_past", 0.0) / 2.0, 0, 1)
    unatt = np.clip(_col(df, "unattended_threat", 0.0), 0, 1)
    return speed_f * (0.45 * threat + 0.30 * contact) + 0.15 * slowing + 0.10 * unatt
