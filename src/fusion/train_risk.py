"""
DriveSight Phase 5, step 2: train and evaluate the risk model ("hard braking within horizon_sec").

Protocol (same discipline as the detector and the tracker):
  - fit on `training`; choose model family, hyper-parameters and alarm threshold on `validation`;
  - `test` is scored ONCE with --final-test, with everything frozen; a second test run is refused
    unless --allow-retest is given (and the report then says so).

The BDDA shortcut. Clips are cut at a fixed offset from the braking event: on this data almost every
GPS braking onset is at 5.5 s of clip time. Clip time alone therefore predicts the label well (reference
`time-only`). Clip time is never given to the models, and the main, shortcut-free metric is the
**same-time AUC**: ROC-AUC computed among frames taken at the SAME clip time (same 200 ms sample) in
different clips, averaged over sample times (weighted by frames). `time-only` scores 0.5 on it by
construction. Model selection uses this metric.

Compared on the same frames (in scope: after the 1.4 s warm-up, up to the onset):
  rule baseline   fixed formula (src/fusion/rule_baseline.py), not fitted
  logistic        standardised, class-balanced logistic regression
  boosting        HistGradientBoosting, class-balanced
  time-only       reference only (sees clip time): measures the shortcut
Feature sets: all (objects, motion, attention, GPS) and vision (objects + motion: what a plain dash-cam
video gives; used by the demo app). Ablation by feature group on validation.

Metrics: AP, ROC-AUC, same-time AUC, F1 at the validation threshold, clip-level alarms (first alarm in a
hard-braking clip: timely = within the horizon, early = before it, miss), false-alarm rate on clips
without hard braking, 95 % CIs by sequence-level bootstrap.

Outputs: data/risk/risk_model_fusion.joblib, data/risk/risk_model_vision_v5.joblib,
         reports/video_dataset/phase5_model_{validation,test}.md / .json, reports/video_dataset/figures/
Run from the repository root:
    python -m src.fusion.train_risk
    python -m src.fusion.train_risk --final-test        # once, at the very end
"""

from __future__ import annotations

import argparse
import itertools
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.fusion.fusion_config import FusionConfig, file_sha256
from src.fusion.fusion_core import (ALL_FEATURES, GAZE_FEATURES, KINEMATIC_FEATURES,
                                    OBJECT_FEATURES, VISION_FEATURES)
from src.fusion.rule_baseline import rule_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ------------------------------------------------------------------------------------------ metrics
def auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC-AUC by the rank (Mann-Whitney) formula, ties counted half."""
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def same_time_auc(y, s, t_ms) -> float:
    y, s, t_ms = np.asarray(y), np.asarray(s), np.asarray(t_ms)
    num = den = 0.0
    order = np.argsort(t_ms, kind="stable")
    t_sorted = t_ms[order]
    cuts = np.flatnonzero(np.diff(t_sorted)) + 1
    for idx in np.split(order, cuts):
        a = auc(y[idx], s[idx])
        if not np.isnan(a):
            num += a * len(idx)
            den += len(idx)
    return num / den if den else float("nan")


def ap(y, s) -> float:
    return float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def best_f1_threshold(y, s) -> float:
    if len(np.unique(y)) < 2:
        return float(np.median(s))
    p, r, th = precision_recall_curve(y, s)
    f = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(th[int(np.nanargmax(f[:-1]))]) if len(th) else float(np.median(s))


def clip_metrics(df: pd.DataFrame, s: np.ndarray, thr: float, horizon: float) -> dict:
    d = df[["video_id", "t_sec", "has_event", "lead_sec"]].assign(_s=s, _a=s >= thr)
    timely = early = miss = 0
    leads = []
    for _, g in d[d["has_event"]].groupby("video_id"):
        hit = g[g["_a"]].sort_values("t_sec")
        if hit.empty:
            miss += 1
            continue
        lead = float(hit["lead_sec"].iloc[0])
        leads.append(lead)
        if lead <= horizon + 1e-6:
            timely += 1
        else:
            early += 1
    n = timely + early + miss
    ne = d[~d["has_event"]].groupby("video_id")["_a"].any()
    return {"event_clips": n, "timely_alarm_rate": timely / n if n else float("nan"),
            "early_alarm_rate": early / n if n else float("nan"), "miss_rate": miss / n if n else float("nan"),
            "median_lead_sec": float(np.median(leads)) if leads else float("nan"),
            "no_event_clips": int(len(ne)), "false_alarm_clip_rate": float(ne.mean()) if len(ne) else float("nan")}


def all_metrics(df: pd.DataFrame, s: np.ndarray, thr: float, horizon: float) -> dict:
    y = df["y"].to_numpy()
    return {"AP": ap(y, s), "ROC_AUC": auc(y, s), "same_time_AUC": same_time_auc(y, s, df["t_ms"].to_numpy()),
            "F1_at_thr": float(f1_score(y, (s >= thr).astype(int), zero_division=0)), "threshold": float(thr),
            "frames": int(len(y)), "positive_rate": float(y.mean()) if len(y) else float("nan"),
            **clip_metrics(df, s, thr, horizon)}


def bootstrap(df: pd.DataFrame, scores: dict, pairs: list, n: int, seed: int) -> dict:
    """Sequence-level bootstrap: 95 % CIs of AP and same-time AUC, and of paired differences."""
    rng = np.random.default_rng(seed)
    vids = df["video_id"].to_numpy()
    uniq = np.unique(vids)
    idx_of = {v: np.flatnonzero(vids == v) for v in uniq}
    y, t = df["y"].to_numpy(), df["t_ms"].to_numpy()
    samp = {(m, k): [] for m in scores for k in ("AP", "stAUC")}
    for _ in range(n):
        ii = np.concatenate([idx_of[v] for v in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[ii])) < 2:
            continue
        for m, s in scores.items():
            samp[(m, "AP")].append(ap(y[ii], s[ii]))
            samp[(m, "stAUC")].append(same_time_auc(y[ii], s[ii], t[ii]))
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    out = {f"{m}|{k}": ci(v) for (m, k), v in samp.items() if v}
    for a, b in pairs:
        for k in ("AP", "stAUC"):
            out[f"{a} - {b}|{k}"] = ci(np.array(samp[(a, k)]) - np.array(samp[(b, k)]))
    return out


# ------------------------------------------------------------------------------------------- models
def logistic(C_reg: float, seed: int):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=C_reg, class_weight="balanced", max_iter=3000))


def boosting(lr: float, leaves: int, l2: float, seed: int):
    return HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=leaves, l2_regularization=l2,
                                          max_iter=300, min_samples_leaf=40, early_stopping=False,
                                          class_weight="balanced", random_state=seed)


GRIDS = {"logistic": [dict(C_reg=c) for c in (0.01, 0.1, 1.0)],
         "boosting": [dict(lr=lr, leaves=lv, l2=l2)
                      for lr, lv, l2 in itertools.product((0.03, 0.1), (15, 31), (0.0, 1.0))]}
FACTORY = {"logistic": logistic, "boosting": boosting}


def fit_select(kind: str, tr: pd.DataFrame, va: pd.DataFrame, feats: list, seed: int):
    """Best hyper-parameters on validation same-time AUC (ties: AP)."""
    best, table = None, []
    for params in GRIDS[kind]:
        m = FACTORY[kind](**params, seed=seed).fit(tr[feats], tr["y"])
        s = m.predict_proba(va[feats])[:, 1]
        st, a = same_time_auc(va["y"], s, va["t_ms"]), ap(va["y"].to_numpy(), s)
        table.append({"kind": kind, **params, "val_same_time_AUC": st, "val_AP": a})
        if best is None or (st, a) > (best[1], best[2]):
            best = (m, st, a, params)
    return best, table


# ------------------------------------------------------------------------------------------ figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_pr(df, scores, path, title):
    plt = _plt()
    y = df["y"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, ls="--" if name == "time-only" else "-", label=f"{name} (AP {ap(y, s):.3f})")
    ax.axhline(y.mean(), color="grey", lw=0.8, ls=":", label="chance")
    ax.set(xlabel="recall", ylabel="precision", title=title, xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_importance(imp: pd.Series, path):
    plt = _plt()
    top = imp.sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.barh(top.index, top.values, color="#3b6ea5")
    ax.set(xlabel="drop in validation same-time AUC when shuffled", title="Permutation importance (top 15)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_timelines(df, scores, thr, path, horizon, n=6):
    plt = _plt()
    vids = df[df["has_event"]]["video_id"].drop_duplicates().head(n).tolist()
    if not vids:
        return
    fig, axes = plt.subplots(len(vids), 1, figsize=(8, 1.7 * len(vids)), sharex=True, squeeze=False)
    for ax, vid in zip(axes[:, 0], vids):
        m = (df["video_id"] == vid).to_numpy()
        g = df[m]
        for name, s in scores.items():
            ax.plot(g["t_sec"], s[m], label=name)
        onset = g["onset_sec"].iloc[0]
        ax.axvspan(onset - horizon, onset, color="orange", alpha=0.2)
        ax.axvline(onset, color="red", lw=1)
        ax.axhline(thr, color="grey", lw=0.8, ls=":")
        ax.set_ylabel(vid, fontsize=8)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    axes[-1, 0].set_xlabel("clip time (s) - red: braking onset, orange: positive window")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- main
def load_split(cfg: FusionConfig, split: str) -> pd.DataFrame:
    p = cfg.frame_features_path(split)
    if not p.exists():
        raise SystemExit(f"{p} not found: run python -m src.fusion.run_fusion first.")
    df = pd.read_parquet(p)
    df["video_id"] = df["video_id"].astype(str)
    return df[df["in_scope"]].reset_index(drop=True)


SPEED_STRATA = (0.0, 15.0, 25.0)


def speed_strata(df: pd.DataFrame, scores: dict) -> list:
    """Same-time AUC on samples with GPS speed >= a floor. A hard braking (>= 2 m/s^2) is physically
    unlikely at very low speed, so speed alone separates the classes; the strata show what is left."""
    out = []
    for lo in SPEED_STRATA:
        m = (df["speed_kmh"] >= lo).to_numpy() if lo > 0 else np.ones(len(df), bool)
        d = df[m]
        out.append({"speed_min_kmh": lo, "samples": int(m.sum()), "positive_rate": float(d["y"].mean()),
                    **{k: same_time_auc(d["y"], v[m], d["t_ms"]) for k, v in scores.items() if k != "time-only"}})
    return out


def permutation_importance_st(model, df, feats, seed, repeats=3) -> pd.Series:
    rng = np.random.default_rng(seed)
    base = same_time_auc(df["y"], model.predict_proba(df[feats])[:, 1], df["t_ms"])
    out = {}
    for f in feats:
        drops = []
        for _ in range(repeats):
            x = df[feats].copy()
            x[f] = rng.permutation(x[f].to_numpy())
            drops.append(base - same_time_auc(df["y"], model.predict_proba(x)[:, 1], df["t_ms"]))
        out[f] = float(np.mean(drops))
    return pd.Series(out)


def main(argv=None) -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--final-test", action="store_true", help="score the protected test split (once)")
    ap_.add_argument("--allow-retest", action="store_true", help="accept a second test run (reported)")
    ap_.add_argument("--n-bootstrap", type=int, default=None)
    ap_.add_argument("--processed-root", default=None)
    ap_.add_argument("--reports-dir", default=None)
    ap_.add_argument("--models-dir", default=None)
    args = ap_.parse_args(argv)
    kw = {k: v for k, v in (("processed_root", args.processed_root), ("reports_dir", args.reports_dir),
                            ("models_dir", args.models_dir)) if v}
    if args.n_bootstrap is not None:
        kw["n_bootstrap"] = args.n_bootstrap
    cfg = FusionConfig(**kw)
    rep, fig_dir, mdir = Path(cfg.reports_dir), Path(cfg.reports_dir) / "figures", Path(cfg.models_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)
    marker = rep / "phase5_test_runs.json"
    if args.final_test and marker.exists() and not args.allow_retest:
        raise SystemExit(f"The test split was already scored ({marker}). Use --allow-retest only if you "
                         "accept that the test estimate is no longer untouched.")

    tr, va = load_split(cfg, "training"), load_split(cfg, "validation")
    for name, part in (("training", tr), ("validation", va)):
        if part.empty or part["y"].nunique() < 2:
            raise SystemExit(f"{name}: no in-scope frames or a single class - cannot train / evaluate.")
    print(f"in-scope frames: training {len(tr)} ({tr['video_id'].nunique()} clips), "
          f"validation {len(va)} ({va['video_id'].nunique()} clips)")

    # ---- candidates (all features), selection on validation same-time AUC
    sel_table, cands = [], {}
    for kind in ("logistic", "boosting"):
        (m, st, a, params), table = fit_select(kind, tr, va, ALL_FEATURES, cfg.seed)
        sel_table += table
        cands[kind] = (m, params)
        print(f"  {kind:9s} best val same-time AUC {st:.3f} (AP {a:.3f}) {params}")
    sel_st = {k: same_time_auc(va["y"], cands[k][0].predict_proba(va[ALL_FEATURES])[:, 1], va["t_ms"]) for k in cands}
    best = max(sel_st, key=sel_st.get)
    best_model, best_params = cands[best]
    print(f"  selected: {best} {best_params}")
    time_ref = boosting(0.1, 15, 0.0, cfg.seed).fit(tr[["t_sec"]], tr["y"])
    vision = FACTORY[best](**best_params, seed=cfg.seed).fit(tr[VISION_FEATURES], tr["y"])

    def scores_for(part):
        return {"rule baseline": rule_score(part),
                "logistic": cands["logistic"][0].predict_proba(part[ALL_FEATURES])[:, 1],
                "boosting": cands["boosting"][0].predict_proba(part[ALL_FEATURES])[:, 1],
                "vision only": vision.predict_proba(part[VISION_FEATURES])[:, 1],
                "speed-only": part["speed_kmh"].fillna(0.0).to_numpy(float),
                "time-only": time_ref.predict_proba(part[["t_sec"]])[:, 1]}

    s_va = scores_for(va)
    thr = {k: best_f1_threshold(va["y"].to_numpy(), v) for k, v in s_va.items()}
    res = {"validation": {k: all_metrics(va, v, thr[k], cfg.horizon_sec) for k, v in s_va.items()}}
    boot = {"validation": bootstrap(va, {k: s_va[k] for k in ("rule baseline", best, "vision only")},
                                    [(best, "rule baseline"), ("vision only", "rule baseline")],
                                    cfg.n_bootstrap, cfg.seed)}

    # ---- ablation (validation)
    groups = {"objects": OBJECT_FEATURES, "objects + motion (vision)": VISION_FEATURES,
              "vision + attention": VISION_FEATURES + GAZE_FEATURES, "GPS kinematics only": KINEMATIC_FEATURES,
              "all - attention": [f for f in ALL_FEATURES if f not in GAZE_FEATURES],
              "all - GPS": [f for f in ALL_FEATURES if f not in KINEMATIC_FEATURES], "all": ALL_FEATURES}
    ablation = {}
    for g, feats in groups.items():
        m = FACTORY[best](**best_params, seed=cfg.seed).fit(tr[feats], tr["y"])
        s = m.predict_proba(va[feats])[:, 1]
        ablation[g] = {"n_features": len(feats), "AP": ap(va["y"].to_numpy(), s),
                       "same_time_AUC": same_time_auc(va["y"], s, va["t_ms"])}
    imp = permutation_importance_st(best_model, va, ALL_FEATURES, cfg.seed)
    strata = {"validation": speed_strata(va, s_va)}

    plot_pr(va, s_va, fig_dir / "phase5_pr_validation.png", "Validation - hard braking within 2 s")
    plot_importance(imp, fig_dir / "phase5_importance_validation.png")
    plot_timelines(va, {k: s_va[k] for k in ("rule baseline", best, "vision only")}, thr[best],
                   fig_dir / "phase5_timelines_validation.png", cfg.horizon_sec)

    # ---- test, once
    if args.final_test:
        te = load_split(cfg, "test")
        s_te = scores_for(te)
        res["test"] = {k: all_metrics(te, v, thr[k], cfg.horizon_sec) for k, v in s_te.items()}
        strata["test"] = speed_strata(te, s_te)
        boot["test"] = bootstrap(te, {k: s_te[k] for k in ("rule baseline", best, "vision only")},
                                 [(best, "rule baseline"), ("vision only", "rule baseline")],
                                 cfg.n_bootstrap, cfg.seed)
        plot_pr(te, s_te, fig_dir / "phase5_pr_test.png", "Test - hard braking within 2 s")

    # ---- save models
    import joblib
    up = {s: json.loads(cfg.fusion_schema_path(s).read_text(encoding="utf-8"))["fusion_hash"]
          for s in ("training", "validation")}
    common = {"kind": best, "params": best_params, "horizon_sec": cfg.horizon_sec, "fusion_hashes": up,
              "trained_at": datetime.now().isoformat(timespec="seconds"),
              "task": f"hard braking (GPS decel >= {cfg.hard_brake_decel} m/s^2) within {cfg.horizon_sec} s"}
    fusion_path, vision_path = mdir / "risk_model_fusion.joblib", mdir / "risk_model_vision_v5.joblib"
    joblib.dump({**common, "model": best_model, "features": ALL_FEATURES, "threshold": thr[best],
                 "val_same_time_AUC": res["validation"][best]["same_time_AUC"],
                 "val_AP": res["validation"][best]["AP"]}, fusion_path)
    joblib.dump({**common, "model": vision, "features": VISION_FEATURES, "threshold": thr["vision only"],
                 "val_same_time_AUC": res["validation"]["vision only"]["same_time_AUC"],
                 "val_AP": res["validation"]["vision only"]["AP"]}, vision_path)
    models = {"fusion": {"path": str(fusion_path), "sha256": file_sha256(fusion_path)},
              "vision": {"path": str(vision_path), "sha256": file_sha256(vision_path)}}

    # ---- reports
    meta = {"selected": best, "params": best_params, "selection_table": sel_table, "fusion_hashes": up,
            "models": models, "config": cfg.fusion_params(), "n_bootstrap": cfg.n_bootstrap}
    payload = {"metrics": res["validation"], "bootstrap": boot["validation"], "ablation": ablation,
               "speed_strata": strata["validation"],
               "permutation_importance": imp.sort_values(ascending=False).to_dict(), **meta}
    (rep / "phase5_model_validation.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    write_report(rep / "phase5_model_validation.md", "validation", res["validation"], boot["validation"],
                 best, best_params, ablation, imp, cfg, tr, va, sel_table, strata["validation"])
    if args.final_test:
        runs = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else []
        runs.append({"at": datetime.now().isoformat(timespec="seconds"), "models": models})
        marker.write_text(json.dumps(runs, indent=2), encoding="utf-8")
        (rep / "phase5_model_test.json").write_text(json.dumps({"metrics": res["test"], "bootstrap": boot["test"],
                                                                "speed_strata": strata["test"],
                                                                "test_runs": len(runs), **meta}, indent=2,
                                                               default=float), encoding="utf-8")
        write_report(rep / "phase5_model_test.md", "test", res["test"], boot["test"], best, best_params,
                     None, None, cfg, tr, va, None, strata["test"], test_runs=len(runs))
    for split, r in res.items():
        b, v = r[best], r["vision only"]
        print(f"[{split}] {best}: same-time AUC {b['same_time_AUC']:.3f}, AP {b['AP']:.3f} | vision only: "
              f"{v['same_time_AUC']:.3f} / {v['AP']:.3f} | rule: {r['rule baseline']['same_time_AUC']:.3f} / "
              f"{r['rule baseline']['AP']:.3f} | speed-only {r['speed-only']['same_time_AUC']:.3f} | "
              f"time-only AP {r['time-only']['AP']:.3f}")
    print(f"reports -> {rep}/phase5_model_*.md, models -> {mdir}")


def fmt(x, p=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f} %"


def write_report(path, split, res, boot, best, params, ablation, imp, cfg, tr, va, sel_table, strata,
                 test_runs=None):
    L = [f"# Phase 5 - risk model, {split} split", "",
         f"Task: for every 200 ms sample, predict a hard braking (GPS deceleration >= {cfg.hard_brake_decel} m/s², "
         f"speed before >= {cfg.min_speed_before_brake} m/s) within the next {cfg.horizon_sec} s. "
         f"In scope: samples after the {cfg.motion_lag_sec + cfg.motion_lag_tol_sec:.1f} s warm-up and up to the onset; "
         f"clips without a hard braking are negatives (`no_event_policy = {cfg.no_event_policy}`).",
         f"Training: {len(tr)} samples / {tr['video_id'].nunique()} clips. Validation: {len(va)} samples / "
         f"{va['video_id'].nunique()} clips.",
         f"Selected model: **{best}** {params} (highest validation same-time AUC). Thresholds: best F1 on validation.", ""]
    if test_runs:
        L += [f"**Test split, scored with frozen models and thresholds (test run #{test_runs}).**", ""]
    L += ["| model | same-time AUC | AP | ROC-AUC | F1 @thr | timely alarm | early alarm | miss | false-alarm clips | median lead (s) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, m in res.items():
        L.append(f"| {k} | {fmt(m['same_time_AUC'])} | {fmt(m['AP'])} | {fmt(m['ROC_AUC'])} | {fmt(m['F1_at_thr'])} | "
                 f"{pct(m['timely_alarm_rate'])} | {pct(m['early_alarm_rate'])} | {pct(m['miss_rate'])} | "
                 f"{pct(m['false_alarm_clip_rate'])} | {fmt(m['median_lead_sec'], 2)} |")
    L += ["", f"Positive rate: {pct(res[best]['positive_rate'])} ({res[best]['frames']} samples; "
          f"{res[best]['event_clips']} hard-braking clips, {res[best]['no_event_clips']} clips without).", "",
          f"95 % CIs, sequence-level bootstrap ({cfg.n_bootstrap} resamples):", ""]
    for k, v in boot.items():
        name, metric = k.split("|")
        L.append(f"- {name}, {'same-time AUC' if metric == 'stAUC' else 'AP'}: [{fmt(v[0])}, {fmt(v[1])}]")
    L += ["", "**How to read this.** `time-only` sees nothing but the clip time; its AP shows how much the BDDA "
          "cut (braking onset at about 5.5 s in almost every clip) gives away. It scores 0.5 on the same-time AUC "
          "by construction. A model is only useful if it beats 0.5 there, i.e. if it ranks a sample before a hard "
          "braking above a sample of another clip at the same clip time.",
          "", "`speed-only` ranks samples by GPS speed. A hard braking (>= 2 m/s² over ~1 s) is physically unlikely at "
          "very low speed, so speed alone separates many samples. The table below keeps only samples above a speed floor "
          "to show what each model adds beyond speed.", "",
          "| speed >= (km/h) | samples | positive rate | " + " | ".join(k for k in strata[0] if k not in
                                                                    ("speed_min_kmh", "samples", "positive_rate")) + " |",
          "|---|---|---|" + "---|" * (len(strata[0]) - 3)]
    for r in strata:
        L.append(f"| {r['speed_min_kmh']:.0f} | {r['samples']} | {pct(r['positive_rate'])} | " + " | ".join(
            fmt(v) for k, v in r.items() if k not in ("speed_min_kmh", "samples", "positive_rate")) + " |")
    L += ["", "(same-time AUC in each stratum)", ""]
    if ablation:
        L += ["## Ablation (validation)", "", "| features | n | same-time AUC | AP |", "|---|---|---|---|"]
        for g, m in ablation.items():
            L.append(f"| {g} | {m['n_features']} | {fmt(m['same_time_AUC'])} | {fmt(m['AP'])} |")
        L += ["", "## Most useful features (permutation, validation same-time AUC)", ""]
        for f, v in imp.sort_values(ascending=False).head(12).items():
            L.append(f"- `{f}`: {v:+.4f}")
        L += ["", "## Hyper-parameter selection (validation)", "", "| kind | params | same-time AUC | AP |",
              "|---|---|---|---|"]
        for r in sel_table:
            p = {k: v for k, v in r.items() if k not in ("kind", "val_same_time_AUC", "val_AP")}
            L.append(f"| {r['kind']} | {p} | {fmt(r['val_same_time_AUC'])} | {fmt(r['val_AP'])} |")
        L += ["", "![PR](figures/phase5_pr_validation.png)", "", "![importance](figures/phase5_importance_validation.png)",
              "", "![timelines](figures/phase5_timelines_validation.png)", ""]
    L += ["## Caveats", "",
          "- BDDA clips are braking events cut at a fixed offset: base rates are not those of normal driving.",
          "- Labels come from ~1 Hz phone GPS: onsets are known to about +/- 0.5 s.",
          "- The gaze map is an in-lab attention reference (average of observers), not the recording driver's gaze; "
          "attention on an object is approximated from the map's centroid and spread.",
          "- The ego path is a fixed image-space corridor (no lane detection); sizes are image proxies, not metres.",
          "- Validation (95 clips) chose the model, its settings and the thresholds; only the test split is unbiased."]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
