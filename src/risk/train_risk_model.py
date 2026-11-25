"""
Step 4 - Train and evaluate the risk model ("imminent braking within HORIZON_SEC").

Protocol (same discipline as the V2 detector):
  - train on `training`, choose the model / hyper-parameters / alarm threshold on `validation`,
  - `test` stays closed; it is scored once, only with --final-test.

Compared on the same frames:
  rule baseline      fixed formula (rule_baseline.py), no fitting
  logistic           standardised logistic regression (class-balanced)
  boosting           HistGradientBoosting (class-balanced)
  time-only (ref.)   a model that sees only the clip time. BDDA clips are cut ~6.5 s before the brake,
                     so clip time alone is predictive. It is NOT a candidate: it measures this shortcut.
                     Clip time is never given to the other models.

Metrics
  AP (PR-AUC, main metric), ROC-AUC,
  time-matched AUC   ROC-AUC computed inside 1-s bins of clip time, then averaged: can the model tell
                     positive from negative frames at the same moment of the clip? (shortcut-free)
  event metrics      per clip, first frame whose score >= threshold: timely alarm (within the horizon),
                     early alarm (before it), or miss; median lead time
  95% CIs            sequence-level bootstrap (whole clips are resampled)
Ablation (validation): which feature groups help (objects, motion, attention, kinematics).

Outputs: reports/risk/risk_model_report.md, metrics_validation.json (and metrics_test.json),
         reports/risk/figures/*.png, data/risk/risk_model.joblib (all features) and
         data/risk/risk_model_vision.joblib (objects + motion only, used by the demo app's video page)
Run from the repository root:
    python src/risk/train_risk_model.py [--final-test]
"""

from __future__ import annotations

import argparse
import itertools
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import risk_config as C
from build_features import ALL_FEATURES, GAZE_FEATURES, KINEMATIC_FEATURES, MOTION_FEATURES, OBJECT_FEATURES
from rule_baseline import rule_score

warnings.filterwarnings("ignore", category=UserWarning)
FIG = C.RISK_REPORT_DIR / "figures"


# ----------------------------------------------------------------------------- metrics
def safe(fn, y, s):
    return float(fn(y, s)) if len(np.unique(y)) == 2 else float("nan")


def time_matched_auc(y, s, t):
    bins = np.floor(np.asarray(t) / C.TIME_BIN_SEC)
    num = den = 0.0
    for b in np.unique(bins):
        m = bins == b
        if len(np.unique(y[m])) == 2:
            num += roc_auc_score(y[m], s[m]) * m.sum()
            den += m.sum()
    return num / den if den else float("nan")


def best_f1_threshold(y, s):
    if len(np.unique(y)) < 2:
        return float(np.median(s))
    p, r, th = precision_recall_curve(y, s)
    f = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(th[int(np.nanargmax(f[:-1]))]) if len(th) else float(np.median(s))


def event_metrics(df, s, thr):
    out = {"timely": 0, "early": 0, "miss": 0, "leads": []}
    d = df.assign(_s=s)
    for _, g in d[d["has_event"]].groupby(["split", "video_id"]):
        g = g.sort_values("time_seconds")
        hit = g[g["_s"] >= thr]
        if hit.empty:
            out["miss"] += 1
            continue
        lead = float(hit["lead_sec"].iloc[0])
        out["leads"].append(lead)
        out["timely" if lead <= C.HORIZON_SEC + 1e-6 else "early"] += 1
    n = out["timely"] + out["early"] + out["miss"]
    return {"clips": n,
            "timely_alarm_rate": out["timely"] / n if n else float("nan"),
            "early_alarm_rate": out["early"] / n if n else float("nan"),
            "miss_rate": out["miss"] / n if n else float("nan"),
            "median_lead_sec": float(np.median(out["leads"])) if out["leads"] else float("nan")}


def all_metrics(df, s, thr):
    y = df["y"].to_numpy()
    pred = (s >= thr).astype(int)
    return {"AP": safe(average_precision_score, y, s), "ROC_AUC": safe(roc_auc_score, y, s),
            "time_matched_AUC": time_matched_auc(y, s, df["time_seconds"].to_numpy()),
            "F1@thr": float(f1_score(y, pred, zero_division=0)), "threshold": float(thr),
            "positive_rate": float(y.mean()) if len(y) else float("nan"), "frames": int(len(y)),
            **event_metrics(df, s, thr)}


def bootstrap(df, scores: dict, pairs, n=C.N_BOOTSTRAP, seed=C.SEED):
    """Sequence-level bootstrap CIs for AP and time-matched AUC, plus paired differences."""
    rng = np.random.default_rng(seed)
    keys = df[["split", "video_id"]].astype(str).agg("/".join, axis=1).to_numpy()
    uniq = np.unique(keys)
    idx_of = {k: np.flatnonzero(keys == k) for k in uniq}
    y, t = df["y"].to_numpy(), df["time_seconds"].to_numpy()
    samples = {(m, k): [] for m in scores for k in ("AP", "tmAUC")}
    for _ in range(n):
        ii = np.concatenate([idx_of[k] for k in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[ii])) < 2:
            continue
        for m, s in scores.items():
            samples[(m, "AP")].append(average_precision_score(y[ii], s[ii]))
            samples[(m, "tmAUC")].append(time_matched_auc(y[ii], s[ii], t[ii]))
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))] if len(a) else [np.nan, np.nan]
    res = {f"{m}_{k}_CI95": ci(v) for (m, k), v in samples.items()}
    for a, b in pairs:
        for k in ("AP", "tmAUC"):
            d = np.array(samples[(a, k)]) - np.array(samples[(b, k)])
            res[f"{a}_minus_{b}_{k}_CI95"] = ci(d)
    return res


# ----------------------------------------------------------------------------- models
def logistic(C_reg):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C_reg, class_weight="balanced", max_iter=2000))


def boosting(lr, leaves, l2):
    return HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=leaves, l2_regularization=l2,
                                          max_iter=300, early_stopping=False, class_weight="balanced",
                                          random_state=C.SEED)


GRIDS = {
    "logistic": [dict(C_reg=c) for c in (0.01, 0.1, 1.0)],
    "boosting": [dict(lr=lr, leaves=lv, l2=l2) for lr, lv, l2 in itertools.product((0.03, 0.1), (15, 31), (0.0, 1.0))],
}
FACTORY = {"logistic": logistic, "boosting": boosting}


def fit_select(kind, tr, va, feats):
    best = None
    for params in GRIDS[kind]:
        m = FACTORY[kind](**params).fit(tr[feats], tr["y"])
        ap = safe(average_precision_score, va["y"].to_numpy(), m.predict_proba(va[feats])[:, 1])
        if best is None or (np.nan_to_num(ap, nan=-1) > np.nan_to_num(best[1], nan=-1)):
            best = (m, ap, params)
    return best


# ----------------------------------------------------------------------------- figures
def plot_pr(df, scores, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    y = df["y"].to_numpy()
    for name, s in scores.items():
        if len(np.unique(y)) < 2:
            break
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=f"{name} (AP {average_precision_score(y, s):.3f})", ls="--" if "time" in name else "-")
    ax.axhline(y.mean() if len(y) else 0, color="grey", lw=0.8, ls=":", label="chance")
    ax.set(xlabel="recall", ylabel="precision", title=title, xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_importance(imp: pd.Series, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = imp.sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top.index, top.values, color="#3b6ea5")
    ax.set(xlabel="drop in validation AP when shuffled", title="Permutation importance (top 15)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_timelines(df, scores, thr, path, n=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seqs = df[df["has_event"]][["split", "video_id"]].drop_duplicates().head(n).itertuples(index=False)
    seqs = list(seqs)
    if not seqs:
        return
    fig, axes = plt.subplots(len(seqs), 1, figsize=(8, 1.7 * len(seqs)), sharex=True, squeeze=False)
    for ax, (split, vid) in zip(axes[:, 0], seqs):
        m = ((df["split"] == split) & (df["video_id"] == vid)).to_numpy()
        g = df[m]
        for name, s in scores.items():
            if "time" in name:
                continue
            ax.plot(g["time_seconds"], s[m], label=name)
        onset = g["onset_sec"].iloc[0]
        ax.axvspan(onset - C.HORIZON_SEC, onset, color="orange", alpha=0.2)
        ax.axvline(onset, color="red", lw=1)
        ax.axhline(thr, color="grey", lw=0.8, ls=":")
        ax.set_ylabel(f"{vid}", fontsize=8)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    axes[-1, 0].set_xlabel("clip time (s)  -  red: braking onset, orange: positive window")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final-test", action="store_true", help="score the protected test split once")
    args = ap.parse_args()

    df = pd.read_csv(C.FEATURES_CSV, dtype={"video_id": str})
    df = df[df["in_scope"]].reset_index(drop=True)
    tr, va, te = (df[df["split"] == s].reset_index(drop=True) for s in C.SPLITS)
    for name, part in (("training", tr), ("validation", va)):
        if part.empty or part["y"].nunique() < 2:
            raise SystemExit(f"{name} split has no frames or only one class: cannot train/evaluate.")
    n_seq = df.groupby("split")["video_id"].nunique().to_dict()
    print(f"In-scope frames: train {len(tr)}, val {len(va)}, test {len(te)} | sequences {n_seq}")
    if n_seq.get("training", 0) < 50:
        print("WARNING: very few training sequences (pilot?) - results only check that the pipeline runs.")

    # --- candidates
    results, models = {}, {}
    for kind in ("logistic", "boosting"):
        m, ap_va, params = fit_select(kind, tr, va, ALL_FEATURES)
        models[kind] = (m, ALL_FEATURES, params)
        print(f"  {kind:9s} val AP {ap_va:.3f}  params {params}")
    time_ref = boosting(0.1, 15, 0.0).fit(tr[["time_seconds"]], tr["y"])

    def scores_for(part):
        s = {"rule baseline": rule_score(part)}
        for kind, (m, feats, _) in models.items():
            s[kind] = m.predict_proba(part[feats])[:, 1]
        s["time-only (ref.)"] = time_ref.predict_proba(part[["time_seconds"]])[:, 1]
        return s

    s_va = scores_for(va)
    thresholds = {k: best_f1_threshold(va["y"].to_numpy(), v) for k, v in s_va.items()}
    results["validation"] = {k: all_metrics(va, v, thresholds[k]) for k, v in s_va.items()}
    best_kind = max(models, key=lambda k: np.nan_to_num(results["validation"][k]["AP"], nan=-1))
    print(f"  selected: {best_kind}")
    boot_va = bootstrap(va, {k: s_va[k] for k in ("rule baseline", best_kind)}, [(best_kind, "rule baseline")])

    # --- ablation on validation (boosting with the selected params, or logistic if selected)
    groups = {"objects": OBJECT_FEATURES, "objects+motion": OBJECT_FEATURES + MOTION_FEATURES,
              "objects+motion+attention": OBJECT_FEATURES + MOTION_FEATURES + GAZE_FEATURES,
              "kinematics only": KINEMATIC_FEATURES, "all": ALL_FEATURES,
              "all - attention": [f for f in ALL_FEATURES if f not in GAZE_FEATURES]}
    kind_p = models[best_kind][2]
    ablation = {}
    for gname, feats in groups.items():
        m = FACTORY[best_kind](**kind_p).fit(tr[feats], tr["y"])
        s = m.predict_proba(va[feats])[:, 1]
        y = va["y"].to_numpy()
        ablation[gname] = {"AP": safe(average_precision_score, y, s),
                           "time_matched_AUC": time_matched_auc(y, s, va["time_seconds"].to_numpy())}

    # --- importance, figures
    FIG.mkdir(parents=True, exist_ok=True)
    best_model = models[best_kind][0]
    pi = permutation_importance(best_model, va[ALL_FEATURES], va["y"], scoring="average_precision",
                                n_repeats=5, random_state=C.SEED)
    imp = pd.Series(pi.importances_mean, index=ALL_FEATURES)
    plot_importance(imp, FIG / "feature_importance_val.png")
    plot_pr(va, s_va, FIG / "pr_curve_val.png", "Validation - imminent braking")
    plot_timelines(va, {k: s_va[k] for k in ("rule baseline", best_kind)}, thresholds[best_kind],
                   FIG / "timelines_val.png")

    # --- test (one shot, opt-in)
    boot_te = None
    if args.final_test:
        if te.empty:
            raise SystemExit("test split is empty")
        s_te = scores_for(te)
        results["test"] = {k: all_metrics(te, v, thresholds[k]) for k, v in s_te.items()}
        boot_te = bootstrap(te, {k: s_te[k] for k in ("rule baseline", best_kind)}, [(best_kind, "rule baseline")])
        plot_pr(te, s_te, FIG / "pr_curve_test.png", "Test - imminent braking")

    # --- vision-only model for the demo app (an arbitrary video has no attention map and no GPS)
    vision_feats = OBJECT_FEATURES + MOTION_FEATURES
    vision = FACTORY[best_kind](**kind_p).fit(tr[vision_feats], tr["y"])
    s_vis = vision.predict_proba(va[vision_feats])[:, 1]
    vis_thr = best_f1_threshold(va["y"].to_numpy(), s_vis)
    vis_ap = safe(average_precision_score, va["y"].to_numpy(), s_vis)
    print(f"  vision-only model (objects + motion): val AP {vis_ap:.3f}")

    # --- save
    import joblib
    C.RISK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": vision, "kind": best_kind, "features": vision_feats, "params": kind_p, "threshold": vis_thr,
                 "horizon_sec": C.HORIZON_SEC, "val_AP": vis_ap}, C.RISK_DATA_DIR / "risk_model_vision.joblib")
    joblib.dump({"model": best_model, "kind": best_kind, "features": ALL_FEATURES, "params": models[best_kind][2],
                 "threshold": thresholds[best_kind], "horizon_sec": C.HORIZON_SEC}, C.RISK_DATA_DIR / "risk_model.joblib")
    C.RISK_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for split, res in results.items():
        payload = {"metrics": res, "selected": best_kind,
                   "bootstrap": boot_va if split == "validation" else boot_te}
        if split == "validation":
            payload["ablation"] = ablation
            payload["permutation_importance"] = imp.sort_values(ascending=False).to_dict()
        (C.RISK_REPORT_DIR / f"metrics_{split}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(results, best_kind, boot_va, boot_te, ablation, imp, n_seq, models)
    print(f"Report -> {C.RISK_REPORT_DIR / 'risk_model_report.md'}")


def fmt(x, p=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"


def write_report(results, best_kind, boot_va, boot_te, ablation, imp, n_seq, models):
    L = ["# DriveSight - risk model (imminent braking)", "",
         f"Task: for each frame (5 fps), predict a hard braking within the next {C.HORIZON_SEC} s "
         f"(GPS deceleration >= {C.HARD_BRAKE_DECEL} m/s²). Frames after the onset are excluded.",
         f"Sequences in scope: {n_seq}. Selected model: **{best_kind}** {models[best_kind][2]} "
         "(highest validation AP).", ""]
    for split, res in results.items():
        L += [f"## {split.capitalize()}", "",
              "| model | AP | ROC-AUC | time-matched AUC | F1 @thr | timely alarm | early alarm | miss | median lead (s) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k, m in res.items():
            L.append(f"| {k} | {fmt(m['AP'])} | {fmt(m['ROC_AUC'])} | {fmt(m['time_matched_AUC'])} | {fmt(m['F1@thr'])} | "
                     f"{fmt(100 * m['timely_alarm_rate'], 1)}% | {fmt(100 * m['early_alarm_rate'], 1)}% | "
                     f"{fmt(100 * m['miss_rate'], 1)}% | {fmt(m['median_lead_sec'], 2)} |")
        b = boot_va if split == "validation" else boot_te
        if b:
            d_ap = b.get(f"{best_kind}_minus_rule baseline_AP_CI95")
            d_tm = b.get(f"{best_kind}_minus_rule baseline_tmAUC_CI95")
            L += ["", f"Sequence-level bootstrap ({C.N_BOOTSTRAP}x), {best_kind} minus rule baseline: "
                  f"AP [{fmt(d_ap[0])}, {fmt(d_ap[1])}], time-matched AUC [{fmt(d_tm[0])}, {fmt(d_tm[1])}]."]
        L.append("")
    L += ["`time-only (ref.)` is not a candidate: it only sees clip time and measures how much of the task the "
          "BDDA clip cut (~6.5 s before the brake) gives away. A useful model must beat it on **time-matched AUC**, "
          "which compares frames at the same clip time.", "",
          "## Ablation (validation)", "", "| features | AP | time-matched AUC |", "|---|---|---|"]
    for g, m in ablation.items():
        L.append(f"| {g} | {fmt(m['AP'])} | {fmt(m['time_matched_AUC'])} |")
    L += ["", "## Most useful features (permutation importance, validation AP)", ""]
    for f, v in imp.sort_values(ascending=False).head(10).items():
        L.append(f"- `{f}`: {v:+.4f}")
    L += ["", "![PR validation](figures/pr_curve_val.png)", "",
          "![importance](figures/feature_importance_val.png)", "", "![timelines](figures/timelines_val.png)", "",
          "## Caveats", "",
          "- The BDDA attention map is the average of in-lab observers (where an attentive driver would look), "
          "not the recording driver's gaze.",
          "- Labels come from ~1 Hz GPS: onset times are accurate to about +/- 0.5-1 s.",
          "- The ego-path corridor is a fixed image-space approximation (no lane detection).",
          "- The validation split was used for model choice, threshold and comparison; the test split is the only "
          "unbiased estimate (`--final-test`)."]
    (C.RISK_REPORT_DIR / "risk_model_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
