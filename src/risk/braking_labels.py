"""
Step 1 - Braking-event labels from GPS (needs only data/BDDA/*/gps_jsons).

For every sequence in the clean manifest:
  1. compute the acceleration of each consecutive GPS pair (~1 Hz),
  2. mark "hard braking" intervals: accel <= -HARD_BRAKE_DECEL with speed >= MIN_SPEED_BEFORE_BRAKE,
  3. group consecutive hard-braking intervals into events and keep the strongest one,
  4. braking onset = start time of the first interval of that event.

Outputs
  data/risk/braking_events.csv           one row per sequence (has_event, onset_sec, peak_decel, ...)
  reports/risk/braking_labels_report.md  onset distribution, events found per split
  reports/risk/figures/onset_hist.png

Sanity check: BDDA clips were cut ~6.5 s before the braking event, so onsets should
cluster around 5-8 s. A flat or very different distribution means the threshold
or the GPS data need a second look before training anything.

Run from the repository root:
    python src/risk/braking_labels.py [--decel 2.0]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import risk_config as C
from gps_utils import interval_accelerations, load_gps


def find_braking_event(gps: pd.DataFrame, decel: float, min_speed: float, max_interval: float) -> dict:
    iv = interval_accelerations(gps, max_interval)
    base = {"has_event": False, "onset_sec": np.nan, "end_sec": np.nan, "peak_decel": np.nan,
            "speed_at_onset": np.nan, "speed_drop": np.nan, "n_events": 0,
            "gps_points": len(gps), "min_accel": float(np.nanmin(iv["accel"])) if len(iv) and iv["accel"].notna().any() else np.nan}
    if iv.empty:
        return base
    hard = (iv["accel"] <= -decel) & (iv["v_start"] >= min_speed)
    if not hard.any():
        return base
    # group consecutive hard intervals
    grp = (hard != hard.shift(fill_value=False)).cumsum()
    events = []
    for _, g in iv[hard].groupby(grp[hard]):
        events.append({
            "onset_sec": float(g["t_start"].iloc[0]),
            "end_sec": float(g["t_end"].iloc[-1]),
            "peak_decel": float(-g["accel"].min()),
            "speed_at_onset": float(g["v_start"].iloc[0]),
            "speed_drop": float(g["v_start"].iloc[0] - g["v_end"].iloc[-1]),
        })
    best = max(events, key=lambda e: e["peak_decel"])
    base.update(best)
    base["has_event"] = True
    base["n_events"] = len(events)
    return base


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decel", type=float, default=C.HARD_BRAKE_DECEL, help="hard-braking threshold, m/s^2")
    args = ap.parse_args()

    manifest = pd.read_csv(C.MANIFEST_CSV)
    rows = []
    for _, r in manifest.iterrows():
        split, vid = r["split"], str(r["video_id"])
        gps_path = C.BDDA_ROOT / split / "gps_jsons" / f"{vid}.json"
        if not gps_path.exists():
            rows.append({"split": split, "video_id": vid, "has_event": False, "problem": "gps file missing"})
            continue
        ev = find_braking_event(load_gps(gps_path), args.decel, C.MIN_SPEED_BEFORE_BRAKE, C.MAX_GPS_INTERVAL)
        ev.update({"split": split, "video_id": vid,
                   "duration_sec": float(min(r["cam_duration"], r["gaze_duration"])), "problem": ""})
        rows.append(ev)

    df = pd.DataFrame(rows)
    cols = ["split", "video_id", "has_event", "onset_sec", "end_sec", "peak_decel", "speed_at_onset",
            "speed_drop", "n_events", "min_accel", "gps_points", "duration_sec", "problem"]
    df = df.reindex(columns=cols)
    C.RISK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.LABELS_SEQ_CSV, index=False)
    write_report(df, args.decel)
    print(f"Wrote {C.LABELS_SEQ_CSV} ({len(df)} sequences, {int(df['has_event'].sum())} with a braking event)")


def write_report(df: pd.DataFrame, decel: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = C.RISK_REPORT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ev = df[df["has_event"] == True]  # noqa: E712

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].hist(ev["onset_sec"], bins=np.arange(0, 16.5, 0.5), color="#3b6ea5")
    ax[0].axvline(6.5, color="k", ls="--", lw=1, label="expected ~6.5 s")
    ax[0].set(xlabel="braking onset (s from clip start)", ylabel="sequences", title="Onset distribution")
    ax[0].legend()
    ax[1].hist(-df["min_accel"].dropna(), bins=40, color="#a55a3b")
    ax[1].axvline(decel, color="k", ls="--", lw=1, label=f"threshold {decel} m/s²")
    ax[1].set(xlabel="strongest deceleration in clip (m/s²)", ylabel="sequences", title="Peak deceleration")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "onset_hist.png", dpi=120)
    plt.close(fig)

    lines = ["# Braking-event labels (GPS)", "",
             f"- Hard-braking threshold: {decel} m/s² (speed before braking >= {C.MIN_SPEED_BEFORE_BRAKE} m/s, "
             f"GPS gaps > {C.MAX_GPS_INTERVAL} s ignored)",
             f"- Positive horizon used later: {C.HORIZON_SEC} s; no-event policy: `{C.NO_EVENT_POLICY}`", "",
             "| split | sequences | with event | % | onset median (s) | onset IQR (s) | peak decel median (m/s²) |",
             "|---|---|---|---|---|---|---|"]
    for split in C.SPLITS:
        s = df[df["split"] == split]
        e = s[s["has_event"] == True]  # noqa: E712
        if len(s) == 0:
            continue
        q = e["onset_sec"].quantile([0.25, 0.5, 0.75]) if len(e) else pd.Series([np.nan] * 3, index=[0.25, 0.5, 0.75])
        lines.append(f"| {split} | {len(s)} | {len(e)} | {100 * len(e) / len(s):.1f} | {q[0.5]:.2f} | "
                     f"{q[0.25]:.2f}-{q[0.75]:.2f} | {e['peak_decel'].median():.2f} |")
    in_window = ev["onset_sec"].between(4.5, 8.5).mean() if len(ev) else float("nan")
    lines += ["", f"- Share of onsets between 4.5 and 8.5 s: {100 * in_window:.1f}% "
              "(BDDA clips were cut ~6.5 s before the braking event; a high share validates the label).",
              f"- Clips with several hard-braking events: {int((df['n_events'] > 1).sum())} (the strongest one is kept).",
              "", "![onsets](figures/onset_hist.png)", ""]
    (C.RISK_REPORT_DIR / "braking_labels_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
