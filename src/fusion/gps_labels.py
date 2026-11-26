"""
GPS records, causal kinematics and braking labels, all from the Phase 2 rows (no raw BDDA needed).

Phase 2 stores, for each 200 ms sample, the *nearest* 1 Hz GPS record (`gps_rec_t_sec`,
`gps_speed_m_s`). The nearest record can be up to 0.5 s in the FUTURE of the sample, so these columns
are never used as features. Instead the 1 Hz records are rebuilt (one row per distinct
`gps_rec_t_sec`) and the kinematics at time t use only records with t_rec <= t.

Braking label (same rule as the earlier src/risk/braking_labels.py):
  1. acceleration of each consecutive pair of valid records (dt <= max_gps_interval),
  2. hard intervals: accel <= -hard_brake_decel and start speed >= min_speed_before_brake,
  3. consecutive hard intervals form an event; the strongest event (peak deceleration) is kept,
  4. onset = start time of the first interval of that event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.fusion.fusion_config import FusionConfig

EVENT_COLUMNS = ["split", "video_id", "has_event", "onset_sec", "end_sec", "peak_decel", "speed_at_onset",
                 "speed_drop", "n_events", "min_accel", "gps_records", "duration_sec"]


def gps_records(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows of ONE sequence -> distinct valid GPS records: t (s from clip start), speed (m/s)."""
    m = rows["gps_matched"].astype(bool) & rows["gps_speed_valid"].astype(bool)
    g = rows.loc[m, ["gps_rec_t_sec", "gps_speed_m_s"]].rename(columns={"gps_rec_t_sec": "t", "gps_speed_m_s": "speed"})
    g = g.dropna().drop_duplicates("t").sort_values("t").reset_index(drop=True)
    return g


def interval_accelerations(gps: pd.DataFrame, max_interval: float) -> pd.DataFrame:
    if len(gps) < 2:
        return pd.DataFrame(columns=["t_start", "t_end", "v_start", "v_end", "accel"])
    t, v = gps["t"].to_numpy(float), gps["speed"].to_numpy(float)
    dt = np.diff(t)
    ok = (dt > 0) & (dt <= max_interval)
    acc = np.where(ok, np.diff(v) / np.where(dt > 0, dt, 1.0), np.nan)
    return pd.DataFrame({"t_start": t[:-1], "t_end": t[1:], "v_start": v[:-1], "v_end": v[1:], "accel": acc})


def find_braking_event(gps: pd.DataFrame, cfg: FusionConfig) -> dict:
    iv = interval_accelerations(gps, cfg.max_gps_interval)
    has_acc = len(iv) and iv["accel"].notna().any()
    out = {"has_event": False, "onset_sec": np.nan, "end_sec": np.nan, "peak_decel": np.nan,
           "speed_at_onset": np.nan, "speed_drop": np.nan, "n_events": 0,
           "min_accel": float(np.nanmin(iv["accel"])) if has_acc else np.nan, "gps_records": int(len(gps))}
    if not has_acc:
        return out
    hard = ((iv["accel"] <= -cfg.hard_brake_decel) & (iv["v_start"] >= cfg.min_speed_before_brake)).to_numpy()
    if not hard.any():
        return out
    events, start = [], None
    for i, h in enumerate(np.append(hard, False)):
        if h and start is None:
            start = i
        elif not h and start is not None:
            g = iv.iloc[start:i]
            events.append({"onset_sec": float(g["t_start"].iloc[0]), "end_sec": float(g["t_end"].iloc[-1]),
                           "peak_decel": float(-g["accel"].min()), "speed_at_onset": float(g["v_start"].iloc[0]),
                           "speed_drop": float(g["v_start"].iloc[0] - g["v_end"].iloc[-1])})
            start = None
    best = max(events, key=lambda e: e["peak_decel"])
    out.update(best)
    out["has_event"] = True
    out["n_events"] = len(events)
    return out


def causal_kinematics(gps: pd.DataFrame, times: np.ndarray, cfg: FusionConfig, trend_window: float = 2.0) -> pd.DataFrame:
    """Kinematics at each time from records with t_rec <= t only (no look-ahead)."""
    n = len(times)
    out = {k: np.full(n, np.nan) for k in ("speed_kmh", "accel_past", "speed_trend", "gps_age_sec")}
    valid = np.zeros(n, dtype=np.int8)
    if len(gps):
        tg, vg = gps["t"].to_numpy(float), gps["speed"].to_numpy(float)
        idx = np.searchsorted(tg, np.asarray(times, float) + 1e-9, side="right") - 1
        for i, j in enumerate(idx):
            if j < 0:
                continue
            age = times[i] - tg[j]
            out["gps_age_sec"][i] = age
            out["speed_kmh"][i] = vg[j] * 3.6
            valid[i] = int(age <= cfg.max_gps_interval)
            if j >= 1 and 0 < tg[j] - tg[j - 1] <= cfg.max_gps_interval:
                out["accel_past"][i] = (vg[j] - vg[j - 1]) / (tg[j] - tg[j - 1])
            k = np.searchsorted(tg, tg[j] - trend_window - 1e-9, side="left")
            if k < j and tg[j] - tg[k] > 0:
                out["speed_trend"][i] = (vg[j] - vg[k]) / (tg[j] - tg[k])
    df = pd.DataFrame(out)
    df["gps_valid"] = valid
    return df


def sequence_event(split: str, video_id: str, rows: pd.DataFrame, cfg: FusionConfig) -> dict:
    ev = find_braking_event(gps_records(rows), cfg)
    ev.update({"split": split, "video_id": str(video_id), "duration_sec": float(rows["t_sec"].max())})
    return ev


def label_rows(t_sec: np.ndarray, ev: dict, cfg: FusionConfig) -> pd.DataFrame:
    """lead_sec = onset - t; y = 1 if 0 <= lead <= horizon; frames after the onset are out of scope."""
    n = len(t_sec)
    if ev["has_event"]:
        lead = ev["onset_sec"] - np.asarray(t_sec, float)
        y = ((lead >= -1e-6) & (lead <= cfg.horizon_sec + 1e-6)).astype(np.int8)
        scope = lead >= -1e-6
    else:
        lead = np.full(n, np.nan)
        y = np.zeros(n, dtype=np.int8)
        scope = np.full(n, cfg.no_event_policy == "negative")
    return pd.DataFrame({"has_event": bool(ev["has_event"]), "onset_sec": ev["onset_sec"], "lead_sec": lead,
                         "y": y, "in_scope": scope})
