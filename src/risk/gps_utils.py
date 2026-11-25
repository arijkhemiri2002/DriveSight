"""
GPS helpers for the risk stage: load a BDDA GPS JSON into a clean, time-sorted
table, and compute causal (past-only) kinematics at any video timestamp.

BDDA GPS JSON: {"startTime": ms, "endTime": ms, "locations": [{"timestamp": ms,
"speed": m/s, "course": deg, "latitude", "longitude", "accuracy"}, ...]}.
Invalid speeds (negative, NaN) are dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_gps(json_path: Path) -> pd.DataFrame:
    """Return columns t (s from video start), speed (m/s), course (deg), sorted by t."""
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    locs = meta.get("locations", []) or []
    start = meta.get("startTime")
    if not locs:
        return pd.DataFrame(columns=["t", "speed", "course"])
    df = pd.DataFrame(locs)
    if start is None:
        start = float(df["timestamp"].min())
    df = df[["timestamp", "speed"] + (["course"] if "course" in df.columns else [])].copy()
    if "course" not in df.columns:
        df["course"] = np.nan
    df["t"] = (df["timestamp"].astype(float) - float(start)) / 1000.0
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    df = df[np.isfinite(df["speed"]) & (df["speed"] >= 0)]
    df = df.sort_values("t").drop_duplicates("t").reset_index(drop=True)
    return df[["t", "speed", "course"]]


def interval_accelerations(gps: pd.DataFrame, max_interval: float) -> pd.DataFrame:
    """One row per consecutive GPS pair: t_start, t_end, v_start, v_end, accel (m/s^2).
    Pairs further apart than max_interval get accel = NaN."""
    if len(gps) < 2:
        return pd.DataFrame(columns=["t_start", "t_end", "v_start", "v_end", "accel"])
    t, v = gps["t"].to_numpy(), gps["speed"].to_numpy()
    dt = np.diff(t)
    acc = np.where((dt > 0) & (dt <= max_interval), np.diff(v) / np.where(dt > 0, dt, 1), np.nan)
    return pd.DataFrame({"t_start": t[:-1], "t_end": t[1:], "v_start": v[:-1], "v_end": v[1:], "accel": acc})


def causal_kinematics(gps: pd.DataFrame, times: np.ndarray, max_interval: float,
                      trend_window: float = 2.0) -> pd.DataFrame:
    """Kinematics at each time using only GPS records with t_gps <= t (no look-ahead).

    speed_mps      last known speed
    accel_past     acceleration over the last GPS interval ending at or before t
    speed_trend    (v_last - v at ~t-trend_window) / elapsed, past only
    gps_age        t - t_last_record
    gps_valid      a record exists at or before t and is not older than max_interval
    """
    n = len(times)
    out = {k: np.full(n, np.nan) for k in ("speed_mps", "accel_past", "speed_trend", "gps_age")}
    valid = np.zeros(n, dtype=bool)
    if len(gps):
        tg, vg = gps["t"].to_numpy(), gps["speed"].to_numpy()
        idx = np.searchsorted(tg, times, side="right") - 1          # last record <= t
        for i, j in enumerate(idx):
            if j < 0:
                continue
            age = times[i] - tg[j]
            out["gps_age"][i] = age
            out["speed_mps"][i] = vg[j]
            valid[i] = age <= max_interval
            if j >= 1 and 0 < tg[j] - tg[j - 1] <= max_interval:
                out["accel_past"][i] = (vg[j] - vg[j - 1]) / (tg[j] - tg[j - 1])
            k = np.searchsorted(tg, tg[j] - trend_window, side="left")
            if k < j and tg[j] - tg[k] > 0:
                out["speed_trend"][i] = (vg[j] - vg[k]) / (tg[j] - tg[k])
    df = pd.DataFrame(out)
    df["gps_valid"] = valid
    return df
