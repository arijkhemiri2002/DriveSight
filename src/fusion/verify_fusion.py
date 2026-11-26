"""
Independent checks of the Phase 5 fusion output. Prints ALL CHECKS PASSED or the list of failures and
writes reports/video_dataset/phase5_verification.md.

Checks, per split:
  1. inputs unchanged      SHA-256 of aligned_records.csv and tracks.parquet = the ones recorded at fusion time
  2. outputs intact        SHA-256 of the three outputs = _schema.json; fusion hash recomputed from settings
  3. coverage              one frame row per Phase 2 row, same (video_id, t_ms) keys, no duplicate
  4. link to Phase 4       one object row per tracked box, same (video_id, t_ms, det_idx, track_id)
  5. labels                braking events recomputed here from the Phase 2 GPS columns (own code), and
                           y / lead_sec / in_scope consistent with them, the horizon and the warm-up
  6. no look-ahead         for sampled sequences, the fusion is re-run on inputs cut at a random time T;
                           every feature at t <= T must be identical (labels excluded)
  7. no clip time          no time column, nor gps_age_sec, is a model feature
  8. value ranges          attention in [0, 1], contact time in (0, cap], counts >= 0, no infinity
  9. split separation      a video id belongs to one split only
Run from the repository root:
    python -m src.fusion.verify_fusion [--n-causal 15]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.fusion.fusion_config import SPLITS, FusionConfig, file_sha256
from src.fusion.fusion_core import ALL_FEATURES, STORED_ONLY, fuse_sequence
from src.fusion.run_fusion import add_common_args, build_config, upstream_info

TIME_COLUMNS = {"t_sec", "t_ms", "sample_idx", "lead_sec", "onset_sec", "warmup"} | set(STORED_ONLY)


class Report:
    def __init__(self):
        self.lines, self.failures = [], []

    def check(self, ok: bool, name: str, detail: str = "") -> None:
        mark = "PASS" if ok else "FAIL"
        self.lines.append(f"- {mark} {name}" + (f" ({detail})" if detail else ""))
        if not ok:
            self.failures.append(f"{name} {detail}".strip())


def own_braking_onset(rows: pd.DataFrame, cfg: FusionConfig):
    """Independent re-implementation of the label rule (strongest run of hard intervals)."""
    g = rows[rows["gps_matched"].astype(bool) & rows["gps_speed_valid"].astype(bool)]
    g = g[["gps_rec_t_sec", "gps_speed_m_s"]].dropna().drop_duplicates("gps_rec_t_sec").sort_values("gps_rec_t_sec")
    t, v = g["gps_rec_t_sec"].to_numpy(float), g["gps_speed_m_s"].to_numpy(float)
    best = None
    i = 0
    while i < len(t) - 1:
        runs = []
        j = i
        while j < len(t) - 1:
            dt = t[j + 1] - t[j]
            a = (v[j + 1] - v[j]) / dt if 0 < dt <= cfg.max_gps_interval else np.nan
            if np.isfinite(a) and a <= -cfg.hard_brake_decel and v[j] >= cfg.min_speed_before_brake:
                runs.append(-a)
                j += 1
            else:
                break
        if runs:
            peak = max(runs)
            if best is None or peak > best[1]:
                best = (t[i], peak)
            i = j
        else:
            i += 1
    return (best[0] if best else None), (best[1] if best else None)


def verify_split(split: str, cfg: FusionConfig, rep: Report, n_causal: int, rng) -> dict:
    sp = cfg.fusion_schema_path(split)
    if not sp.exists():
        rep.check(False, f"[{split}] fusion output exists", str(sp))
        return {}
    schema = json.loads(sp.read_text(encoding="utf-8"))
    up_now = upstream_info(split, cfg)
    rep.check(up_now == schema["upstream"], f"[{split}] 1. Phase 2 / Phase 4 inputs unchanged since fusion")
    for key, path in (("frame_features_sha256", cfg.frame_features_path(split)),
                      ("object_features_sha256", cfg.object_features_path(split)),
                      ("events_sha256", cfg.events_path(split))):
        rep.check(Path(path).exists() and file_sha256(path) == schema[key], f"[{split}] 2. {Path(path).name} intact")
    rep.check(cfg.fusion_hash(up_now) == schema["fusion_hash"], f"[{split}] 2. fusion hash matches the current settings",
              schema["fusion_hash"])

    rows = pd.read_csv(cfg.records_path(split), dtype={"video_id": str, "split": str})
    tracks = pd.read_parquet(cfg.tracks_path(split))
    tracks["video_id"] = tracks["video_id"].astype(str)
    fr = pd.read_parquet(cfg.frame_features_path(split))
    ob = pd.read_parquet(cfg.object_features_path(split))
    ev = pd.read_csv(cfg.events_path(split), dtype={"video_id": str})
    fr["video_id"], ob["video_id"] = fr["video_id"].astype(str), ob["video_id"].astype(str)

    # 3. coverage
    k_rows = set(zip(rows["video_id"], rows["t_ms"].astype(int)))
    k_fr = list(zip(fr["video_id"], fr["t_ms"].astype(int)))
    rep.check(len(k_fr) == len(set(k_fr)), f"[{split}] 3. frame keys unique", f"{len(k_fr)} rows")
    rep.check(set(k_fr) == k_rows and len(fr) == len(rows), f"[{split}] 3. one frame row per Phase 2 row",
              f"{len(fr)} vs {len(rows)}")
    rep.check((fr["split"] == split).all(), f"[{split}] 3. split column")

    # 4. link to Phase 4
    kt = ["video_id", "t_ms", "det_idx", "track_id"]
    a = tracks[kt].astype({"t_ms": int, "det_idx": int, "track_id": int})
    b = ob[kt].astype({"t_ms": int, "det_idx": int, "track_id": int})
    same = len(a) == len(b) and a.sort_values(kt).reset_index(drop=True).equals(b.sort_values(kt).reset_index(drop=True))
    rep.check(same, f"[{split}] 4. one object row per tracked box, same track ids", f"{len(b)} vs {len(a)}")

    # 5. labels
    bad_onset = 0
    for vid, g in rows.groupby("video_id"):
        onset, _ = own_braking_onset(g, cfg)
        e = ev[ev["video_id"] == vid]
        if e.empty:
            bad_onset += 1
            continue
        e = e.iloc[0]
        if (onset is None) != (not bool(e["has_event"])) or (onset is not None and abs(onset - e["onset_sec"]) > 1e-6):
            bad_onset += 1
    rep.check(bad_onset == 0 and len(ev) == rows["video_id"].nunique(),
              f"[{split}] 5. braking events recomputed independently", f"{bad_onset} mismatches")
    m = fr.merge(ev[["video_id", "has_event", "onset_sec"]], on="video_id", suffixes=("", "_ev"))
    lead = m["onset_sec_ev"] - m["t_sec"]
    has = m["has_event_ev"].astype(bool)
    y_exp = (has & (lead >= -1e-6) & (lead <= cfg.horizon_sec + 1e-6)).astype(int)
    warm = m["t_sec"] < cfg.motion_lag_sec + cfg.motion_lag_tol_sec - 1e-9
    scope_exp = ~warm & np.where(has, lead >= -1e-6, cfg.no_event_policy == "negative")
    rep.check((m["y"].astype(int) == y_exp).all(), f"[{split}] 5. y consistent with onset and horizon")
    rep.check((m["in_scope"].astype(bool) == scope_exp).all(), f"[{split}] 5. in_scope excludes warm-up and post-onset")
    rep.check(bool((m.loc[has, "lead_sec"] - lead[has]).abs().max() < 1e-6) if has.any() else True,
              f"[{split}] 5. lead_sec = onset - t")

    # 6. no look-ahead
    vids = rng.choice(sorted(fr["video_id"].unique()), size=min(n_causal, fr["video_id"].nunique()), replace=False)
    diffs = []
    for vid in vids:
        r = rows[rows["video_id"] == vid]
        tk = tracks[tracks["video_id"] == vid]
        T = int(rng.choice(r["t_ms"].to_numpy()[5:])) if len(r) > 6 else int(r["t_ms"].max())
        cut_fr, _, _ = fuse_sequence(split, vid, r[r["t_ms"] <= T], tk[tk["t_ms"] <= T], cfg)
        full = fr[(fr["video_id"] == vid) & (fr["t_ms"] <= T)].sort_values("t_ms").reset_index(drop=True)
        cut_fr = cut_fr.sort_values("t_ms").reset_index(drop=True)
        x, z = full[ALL_FEATURES].astype(float).to_numpy(), cut_fr[ALL_FEATURES].astype(float).to_numpy()
        ok = x.shape == z.shape and np.allclose(x, z, equal_nan=True, atol=1e-9)
        if not ok:
            bad = [f for f in ALL_FEATURES if not np.allclose(full[f].astype(float), cut_fr[f].astype(float),
                                                               equal_nan=True, atol=1e-9)] if x.shape == z.shape else ["shape"]
            diffs.append(f"{vid}@{T}:{bad[:4]}")
    rep.check(not diffs, f"[{split}] 6. features unchanged when the future is removed ({len(vids)} sequences)",
              "; ".join(diffs[:3]))

    # 8. ranges
    att = pd.concat([fr[c] for c in ("lead_attention", "vru_attention", "gaze_on_road_max")]).dropna()
    ttc = pd.concat([fr[c] for c in ("lead_ttc_sec", "vru_ttc_sec", "min_ttc_inpath", "min_ttc_near")]).dropna()
    counts = fr[["n_road_users", "n_vru", "n_vehicle", "n_inpath", "n_vru_near_path", "n_new_near_1s"]]
    num = fr[ALL_FEATURES].astype(float).to_numpy()
    rep.check(bool(att.between(0, 1).all()), f"[{split}] 8. attention in [0, 1]")
    rep.check(bool(((ttc > 0) & (ttc <= cfg.ttc_cap_sec + 1e-9)).all()), f"[{split}] 8. contact time in (0, cap]")
    rep.check(bool((counts >= 0).all().all()), f"[{split}] 8. counts >= 0")
    rep.check(not np.isinf(num).any(), f"[{split}] 8. no infinite feature value")
    s = fr[fr["in_scope"]]
    return {"split": split, "sequences": int(fr["video_id"].nunique()), "frames": int(len(fr)),
            "in_scope": int(len(s)), "positives": int(s["y"].sum()), "objects": int(len(ob)),
            "with_event": int(ev["has_event"].sum()), "fusion_hash": schema["fusion_hash"]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-causal", type=int, default=15, help="sequences per split for the look-ahead test")
    ap.add_argument("--split", choices=SPLITS, action="append")
    ap.add_argument("--out", default=None)
    add_common_args(ap)
    args = ap.parse_args(argv)
    cfg = build_config(args)
    rep, rng = Report(), np.random.default_rng(0)
    leaked = sorted(TIME_COLUMNS & set(ALL_FEATURES))
    rep.check(not leaked, "7. no clip-time column among the model features", ",".join(leaked))
    stats, ids = [], {}
    for split in args.split or SPLITS:
        st = verify_split(split, cfg, rep, args.n_causal, rng)
        if st:
            stats.append(st)
            ids[split] = set(pd.read_parquet(cfg.frame_features_path(split), columns=["video_id"])["video_id"].astype(str))
    names = list(ids)
    overlap = [f"{a}/{b}" for i, a in enumerate(names) for b in names[i + 1:] if ids[a] & ids[b]]
    rep.check(not overlap, "9. splits are separate", ",".join(overlap))

    verdict = "ALL CHECKS PASSED" if not rep.failures else f"{len(rep.failures)} CHECK(S) FAILED"
    L = ["# Phase 5 - fusion verification", "", f"**{verdict}**", "",
         "| split | sequences | frames | in scope | positives | object rows | hard-braking clips | fusion hash |",
         "|---|---|---|---|---|---|---|---|"]
    for s in stats:
        L.append(f"| {s['split']} | {s['sequences']} | {s['frames']} | {s['in_scope']} | {s['positives']} | "
                 f"{s['objects']} | {s['with_event']} | `{s['fusion_hash']}` |")
    L += ["", "## Checks", ""] + rep.lines
    out = Path(args.out) if args.out else Path(cfg.reports_dir) / "phase5_verification.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    for f in rep.failures:
        print("FAIL", f)
    print(verdict)
    print(f"report -> {out}")
    if rep.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
