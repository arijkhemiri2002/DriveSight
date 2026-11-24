"""
GPS parsing, derived fields, and timestamp-based alignment to video frames.

GPS timestamps are in milliseconds. Derived fields:
  time_seconds = (timestamp - startTime) / 1000
  speed_kmh = speed * 3.6
  heading_sin / heading_cos = sin/cos(course)
  acceleration = Δspeed / Δt (using actual timestamp differences)

For each selected video timestamp t, the nearest GPS record is matched if
|t_gps - t| <= GPS_ASSOC_MAX_GAP_SEC; otherwise gps_missing=True.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import json

import numpy as np
import pandas as pd

from preprocess_config import PreprocessConfig


@dataclass
class GpsRecord:
    time_seconds: float
    speed_m_s: float
    speed_kmh: float
    course: float
    heading_sin: float
    heading_cos: float
    latitude: float
    longitude: float
    accuracy: float
    timestamp_ms: int
    acceleration: Optional[float] = None


@dataclass
class GpsAlignment:
    video_id: str
    split: str
    gps_records: int
    gps_duration_sec: float
    gps_start_offset_sec: float
    gps_end_gap_sec: float
    median_interval_sec: float
    max_interval_sec: float
    records: list[GpsRecord]


def parse_gps(json_path: Path) -> dict:
    """Load a GPS JSON file and return raw metadata + locations list."""
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    locations = meta.get("locations", [])
    return {
        "startTime": meta.get("startTime"),
        "endTime": meta.get("endTime"),
        "fileName": meta.get("fileName"),
        "locations": locations,
    }


def build_gps_records(gps_path: Path) -> GpsAlignment:
    """Parse GPS JSON, compute derived fields, return alignment info."""
    raw = parse_gps(gps_path)
    start_time = raw["startTime"]
    end_time = raw["endTime"]
    locations = raw["locations"]

    df = pd.DataFrame(locations)
    if df.empty:
        return GpsAlignment(
            video_id=gps_path.stem, split="",
            gps_records=0, gps_duration_sec=0.0,
            gps_start_offset_sec=float("nan"), gps_end_gap_sec=float("nan"),
            median_interval_sec=float("nan"), max_interval_sec=float("nan"),
            records=[],
        )

    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"].astype(float).values

    # time relative to video start (seconds)
    time_seconds = (ts - start_time) / 1000.0

    speed = df["speed"].astype(float).values
    speed_kmh = speed * 3.6
    course = df["course"].astype(float).values
    heading_sin = np.sin(np.deg2rad(course))
    heading_cos = np.cos(np.deg2rad(course))
    latitude = df["latitude"].astype(float).values
    longitude = df["longitude"].astype(float).values
    accuracy = df["accuracy"].astype(float).values

    # acceleration using actual timestamp differences
    dt = np.diff(ts) / 1000.0  # seconds
    d_speed = np.diff(speed)
    acceleration = np.full(len(speed), np.nan)
    valid = dt > 0
    acceleration[1:][valid] = d_speed[valid] / dt[valid]

    intervals = np.diff(ts) / 1000.0
    median_interval = float(np.median(intervals)) if len(intervals) else float("nan")
    max_interval = float(np.max(intervals)) if len(intervals) else float("nan")

    gps_duration = (end_time - start_time) / 1000.0 if (start_time and end_time) else 0.0
    first_offset = float((ts[0] - start_time) / 1000.0) if start_time else float("nan")
    last_gap = float((end_time - ts[-1]) / 1000.0) if end_time else float("nan")

    records = []
    for i in range(len(df)):
        records.append(GpsRecord(
            time_seconds=float(time_seconds[i]),
            speed_m_s=float(speed[i]),
            speed_kmh=float(speed_kmh[i]),
            course=float(course[i]),
            heading_sin=float(heading_sin[i]),
            heading_cos=float(heading_cos[i]),
            latitude=float(latitude[i]),
            longitude=float(longitude[i]),
            accuracy=float(accuracy[i]),
            timestamp_ms=int(ts[i]),
            acceleration=float(acceleration[i]) if not np.isnan(acceleration[i]) else None,
        ))

    return GpsAlignment(
        video_id=gps_path.stem, split="",
        gps_records=len(df), gps_duration_sec=gps_duration,
        gps_start_offset_sec=first_offset, gps_end_gap_sec=last_gap,
        median_interval_sec=median_interval, max_interval_sec=max_interval,
        records=records,
    )


def nearest_gps(gps_records: list[GpsRecord], target_t: float,
                max_gap_sec: float) -> Optional[GpsRecord]:
    """Return the GPS record nearest to target_t within max_gap_sec, else None."""
    if not gps_records:
        return None
    best = None
    best_gap = float("inf")
    for rec in gps_records:
        gap = abs(rec.time_seconds - target_t)
        if gap < best_gap:
            best_gap = gap
            best = rec
    if best_gap <= max_gap_sec:
        return best
    return None
