"""
GPS parsing, validity filtering, derived fields and timestamp alignment.

BDDA GPS JSON: {"startTime": ms, "endTime": ms, "locations": [{timestamp, speed,
course, latitude, longitude, accuracy}, ...]} recorded at ~1 Hz by a phone.
Time base: time_sec = (timestamp - startTime) / 1000, i.e. seconds from video start
(verified by the Phase 1 audit: first/last records sit ~0.5 s inside the clip).

Validity rules (a record can be partially valid):
  speed_valid    : finite, 0 <= speed <= gps_max_speed_m_s
                   (iOS reports -1 when speed is unknown; seen in validation/1744)
  course_valid   : finite, 0 <= course < 360 (iOS reports -1 when unknown)
  position_valid : finite lat/lon in range, not (0, 0), accuracy >= 0 when present
Invalid fields become NaN. They are never passed on as real values.

Derived fields:
  acceleration   : backward difference (v_i - v_{i-1}) / dt between two consecutive
                   records that both have valid speed and 0 < dt <= gps_accel_max_dt_sec.
                   It is a 1-second finite difference of phone GPS speed: noisy, and
                   only a coarse proxy of longitudinal acceleration.

Alignment to a target time t (5 Hz rows vs 1 Hz GPS):
  * nearest record within gps_assoc_max_gap_sec -> gps_* columns (step function)
  * gps_speed_interp_m_s: linear interpolation between the two bracketing records
    with valid speed if they are <= gps_interp_max_span_sec apart; NaN outside the
    covered interval (no extrapolation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.preprocess_config import PreprocessConfig

GPS_COLUMNS = [
    "gps_matched", "gps_rec_t_sec", "gps_gap_sec",
    "gps_speed_m_s", "gps_speed_kmh", "gps_speed_valid", "gps_speed_interp_m_s",
    "gps_accel_m_s2",
    "gps_course_deg", "gps_heading_sin", "gps_heading_cos", "gps_course_valid",
    "gps_lat", "gps_lon", "gps_accuracy_m", "gps_position_valid",
]


@dataclass
class GpsTrack:
    t: np.ndarray                 # seconds from video start, sorted, unique
    speed: np.ndarray             # m/s, NaN where invalid
    course: np.ndarray            # degrees, NaN where invalid
    lat: np.ndarray
    lon: np.ndarray
    accuracy: np.ndarray
    accel: np.ndarray             # m/s^2, NaN where not computable
    speed_valid: np.ndarray
    course_valid: np.ndarray
    position_valid: np.ndarray
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.t)


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    if name not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)


def load_gps_track(json_path: Path, config: PreprocessConfig) -> GpsTrack:
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    start = meta.get("startTime")
    locs = meta.get("locations") or []
    df = pd.DataFrame(locs)
    stats = {"gps_records_raw": len(df), "gps_dup_timestamps": 0, "gps_start_time_missing": start is None}
    if df.empty or start is None or "timestamp" not in df.columns:
        empty = np.array([], dtype=float)
        stats.update(gps_records=0, gps_speed_invalid=0, gps_course_invalid=0, gps_position_invalid=0)
        return GpsTrack(empty, empty, empty, empty, empty, empty, empty,
                        empty.astype(bool), empty.astype(bool), empty.astype(bool), stats)

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    n_before = len(df)
    df = df.drop_duplicates("timestamp", keep="first").reset_index(drop=True)
    stats["gps_dup_timestamps"] = n_before - len(df)

    t = (df["timestamp"].to_numpy(dtype=float) - float(start)) / 1000.0
    speed, course = _col(df, "speed"), _col(df, "course")
    lat, lon, acc = _col(df, "latitude"), _col(df, "longitude"), _col(df, "accuracy")

    speed_valid = np.isfinite(speed) & (speed >= 0) & (speed <= config.gps_max_speed_m_s)
    course_valid = np.isfinite(course) & (course >= 0) & (course < 360)
    position_valid = (np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
                      & ~((lat == 0) & (lon == 0)) & ~(np.isfinite(acc) & (acc < 0)))

    speed = np.where(speed_valid, speed, np.nan)
    course = np.where(course_valid, course, np.nan)
    lat = np.where(position_valid, lat, np.nan)
    lon = np.where(position_valid, lon, np.nan)
    acc = np.where(position_valid, acc, np.nan)

    accel = np.full(len(t), np.nan)
    if len(t) > 1:
        dt = np.diff(t)
        ok = speed_valid[1:] & speed_valid[:-1] & (dt > 0) & (dt <= config.gps_accel_max_dt_sec)
        accel[1:][ok] = (speed[1:][ok] - speed[:-1][ok]) / dt[ok]

    stats.update(
        gps_records=len(t),
        gps_speed_invalid=int((~speed_valid).sum()),
        gps_course_invalid=int((~course_valid).sum()),
        gps_position_invalid=int((~position_valid).sum()),
    )
    return GpsTrack(t, speed, course, lat, lon, acc, accel,
                    speed_valid, course_valid, position_valid, stats)


def _nearest(t_ref: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Index of the nearest element of sorted t_ref for each target."""
    i = np.searchsorted(t_ref, targets)
    i = np.clip(i, 1, len(t_ref) - 1) if len(t_ref) > 1 else np.zeros_like(i)
    if len(t_ref) > 1:
        left_closer = np.abs(targets - t_ref[i - 1]) <= np.abs(t_ref[i] - targets)
        i = np.where(left_closer, i - 1, i)
    return i


def interpolate_speed(track: GpsTrack, targets: np.ndarray, max_span: float) -> np.ndarray:
    out = np.full(len(targets), np.nan)
    tv, vv = track.t[track.speed_valid], track.speed[track.speed_valid]
    if len(tv) == 0:
        return out
    if len(tv) == 1:
        out[np.isclose(targets, tv[0])] = vv[0]
        return out
    j = np.searchsorted(tv, targets, side="right")          # tv[j-1] <= t < tv[j]
    inside = (targets >= tv[0]) & (targets <= tv[-1])
    j = np.clip(j, 1, len(tv) - 1)
    t0, t1, v0, v1 = tv[j - 1], tv[j], vv[j - 1], vv[j]
    span = t1 - t0
    w = np.where(span > 0, (targets - t0) / np.where(span > 0, span, 1), 0.0)
    val = v0 + w * (v1 - v0)
    ok = inside & (span <= max_span)
    out[ok] = val[ok]
    # exact hit on the last valid record
    out[np.isclose(targets, tv[-1])] = vv[-1]
    return out


def align_gps(track: GpsTrack, targets: np.ndarray, config: PreprocessConfig) -> pd.DataFrame:
    """One row per target time with the GPS_COLUMNS."""
    n = len(targets)
    df = pd.DataFrame(index=range(n), columns=GPS_COLUMNS, dtype=float)
    if len(track) == 0:
        df["gps_matched"] = False
        for c in ("gps_speed_valid", "gps_course_valid", "gps_position_valid"):
            df[c] = False
        return df
    i = _nearest(track.t, targets)
    gap = np.abs(track.t[i] - targets)
    m = gap <= config.gps_assoc_max_gap_sec

    def pick(a):
        return np.where(m, a[i], np.nan)

    df["gps_matched"] = m
    df["gps_rec_t_sec"] = np.round(pick(track.t), 4)
    df["gps_gap_sec"] = np.round(np.where(m, gap, np.nan), 4)
    sp = pick(track.speed)
    df["gps_speed_m_s"] = sp
    df["gps_speed_kmh"] = np.round(sp * 3.6, 4)
    df["gps_speed_valid"] = m & track.speed_valid[i]
    df["gps_speed_interp_m_s"] = np.round(interpolate_speed(track, targets, config.gps_interp_max_span_sec), 4)
    df["gps_accel_m_s2"] = np.round(pick(track.accel), 4)
    co = pick(track.course)
    df["gps_course_deg"] = co
    df["gps_heading_sin"] = np.round(np.sin(np.deg2rad(co)), 6)
    df["gps_heading_cos"] = np.round(np.cos(np.deg2rad(co)), 6)
    df["gps_course_valid"] = m & track.course_valid[i]
    df["gps_lat"] = pick(track.lat)
    df["gps_lon"] = pick(track.lon)
    df["gps_accuracy_m"] = pick(track.accuracy)
    df["gps_position_valid"] = m & track.position_valid[i]
    return df
