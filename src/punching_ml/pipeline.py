"""End-to-end experiment: `python -m punching_ml.pipeline` (needs data/raw/*.xlsx, see data/README.md)."""
from __future__ import annotations

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import plots
from .data import FEATURES, GROUP, ROOT, TARGET, core_set, read_sheet
from .ec2 import capacity_kN
from .evaluate import conformal_study, metrics, nested_tuned_oof, oof_predictions, summarize
from .models import SEED, EC2Residual, Features, PowerLaw, build_models, xgb

RESULTS = ROOT / "results"
CONF_FILES = {"EC2 (C refitted)": "ec2_refitted", "Power law (log-linear)": "power_law",
              "EC2 + XGBoost correction": "ec2_xgboost"}
FIG = RESULTS / "figures"


def bootstrap_exponents(X, y, groups, n=500, seed=SEED) -> pd.DataFrame:
    """Resample whole test programmes, refit the power law, collect exponents."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    rows = []
    for _ in range(n):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        coef = PowerLaw().fit(X.iloc[idx], y.iloc[idx]).lr_.coef_
        rows.append(dict(zip(["d", "fc", "rho_l", "u1", "r_over_d"], coef, strict=True)))
    return pd.DataFrame(rows)


def run(fast: bool = False) -> dict:
    t0 = time.time()
    plots.style()
    FIG.mkdir(parents=True, exist_ok=True)
    reps = 2 if fast else 5

    full = read_sheet()
    df = core_set(full)
    X, y, g = df[FEATURES + ["h"]], df[TARGET], df[GROUP].to_numpy()
    info = {"tests_in_sheet": len(full), "sources_in_sheet": int(full[GROUP].nunique()),
            "core_tests": len(df), "core_sources": int(df[GROUP].nunique()),
            "core_unknown_mode": int(df["mode_unknown"].sum()),
            "square": int((df.circular == 0).sum()), "circular": int(df.circular.sum())}
    print("[data]", info)
    plots.eda(df, FIG / "01_distributions.png")
    plots.per_source(df, FIG / "02_tests_per_programme.png")

    # 1. Random vs grouped CV for every model ---------------------------------
    rows, oof = [], {}
    for name, model in build_models().items():
        for kind in ("random", "grouped"):
            P = oof_predictions(model, X, y, g, kind, 5, reps)
            if kind == "grouped":
                oof[name] = P[:, 0]
            rows.append({"model": name, "cv": kind, **summarize(y, P)})
    cv = pd.DataFrame(rows)
    cv.to_csv(RESULTS / "cv_random_vs_grouped.csv", index=False)
    print(cv[["model", "cv", "R2", "ratio_mean", "ratio_CoV", "log_RMSE"]].round(3).to_string(index=False))
    plots.cv_bars(cv, FIG / "03_random_vs_grouped.png")
    plots.ratio_vs_d(df, {k: oof[k] for k in ["EC2 (mean values)", "Power law (log-linear)", "XGBoost",
                                              "EC2 + XGBoost correction"]},
                     FIG / "04_ratio_vs_depth.png")

    # 2. Nested-tuned EC2 + XGBoost (tuning only inside outer training folds) ----
    Pn, chosen = nested_tuned_oof(X, y, g, n_repeats=1 if fast else 3, n_iter=8 if fast else 25)
    nested = summarize(y, Pn)
    print("[nested]", {k: round(v, 3) for k, v in nested.items()})

    # 3. Sensitivity: drop tests whose failure mode is not reported ------------
    sub = core_set(full, drop_unknown_mode=True)
    Xs, ys, gs = sub[FEATURES + ["h"]], sub[TARGET], sub[GROUP].to_numpy()
    sens = {name: summarize(ys, oof_predictions(build_models()[name], Xs, ys, gs, "grouped", 5, reps))
            for name in ["EC2 (mean values)", "Power law (log-linear)", "EC2 + XGBoost correction"]}
    sens = {k: {m: round(v[m], 3) for m in ("ratio_mean", "ratio_CoV", "log_RMSE")} for k, v in sens.items()}
    sens["n_tests"], sens["n_sources"] = len(sub), int(sub[GROUP].nunique())

    # 4. Power-law exponents with programme bootstrap ---------------------------
    boot = bootstrap_exponents(X, y, g, n=100 if fast else 500)
    boot.to_csv(RESULTS / "powerlaw_bootstrap.csv", index=False)
    plots.exponents(boot, FIG / "05_powerlaw_exponents.png")
    pl_full = PowerLaw().fit(X, y)

    # 5. SHAP: where does EC2 deviate from the tests? ----------------------------
    res = EC2Residual(xgb(n_estimators=300, max_depth=2)).fit(X, y)
    Z = Features().transform(X)
    sv = shap.TreeExplainer(res.est_)(Z)
    sv.feature_names = [{"r_over_d": "r / d (slenderness)", "r_load": "r (load radius)", "rho_l": "ρ",
                         "fc": "f_c", "b_over_d": "b / d", "u1_over_d": "u₁ / d", "log_d": "ln d",
                         "rho_fc_cbrt": "(ρ f_c)^(1/3)", "circular": "circular column"}.get(c, c)
                        for c in Z.columns]
    plt.figure()
    shap.plots.beeswarm(sv, max_display=11, show=False)
    plt.title("SHAP of the EC2 correction: which inputs make EC2 under- or over-predict",
              loc="left", fontweight="bold", fontsize=10)
    plt.gcf().set_size_inches(7.6, 5)
    plt.savefig(FIG / "06_shap_ec2_correction.png")
    plt.close()
    shap_imp = pd.Series(np.abs(sv.values).mean(0), index=Z.columns).sort_values(ascending=False)
    shap_imp.rename("mean_abs_shap_log").to_csv(RESULTS / "shap_ec2_correction.csv")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    r_ec2 = (y / capacity_kN(X)).to_numpy()
    for a, xv, lab, cap, cap_lab in [(ax[0], X["d"], "d (mm)", 200, "k = 2.0 cap below d = 200 mm"),
                                      (ax[1], X["rho_l"], "ρ (%)", 2.0, "ρ ≤ 2 % cap")]:
        a.scatter(xv, r_ec2, s=12, color=plots.BLUE, alpha=0.7, edgecolor=plots.SURFACE, linewidth=0.4)
        a.axhline(1, color=plots.INK2, linestyle="--", linewidth=1)
        a.axvline(cap, color=plots.ORANGE, linewidth=1.5)
        a.text(cap, 0.43, cap_lab + " ", color=plots.INK, fontsize=8.5, va="bottom", ha="right")
        a.set(xlabel=lab, ylabel="V_test / V_EC2 (mean values)", ylim=(0.4, 2.0))
    plots.logx(ax[0], [50, 100, 200, 400])
    ax[0].set_title("EC2 is most conservative for thin slabs")
    ax[1].set_title("…and beyond its reinforcement-ratio cap")
    plots.save(fig, FIG / "07_ec2_residuals.png")

    # 6. Conformal 90% intervals on unseen programmes ---------------------------
    conf = {}
    for name in ["EC2 (C refitted)", "Power law (log-linear)", "EC2 + XGBoost correction"]:
        c = conformal_study(build_models()[name], X, y, g, n_repeats=10 if fast else 30)
        conf[name] = {"coverage_mean": float(c.coverage.mean()), "coverage_sd": float(c.coverage.std()),
                      "factor_mean": float(c.factor.mean())}
        c.to_csv(RESULTS / f"conformal_{CONF_FILES[name]}.csv", index=False)
    print("[conformal]", conf)

    best = "Power law (log-linear)"
    p = oof[best]
    q = conformal_study(build_models()[best], X, y, g, n_repeats=30 if not fast else 10).q_log.mean()
    plots.parity(y.to_numpy(), p, p * np.exp(-q), p * np.exp(q), FIG / "08_parity_powerlaw.png",
                 f"Power law, grouped CV predictions with ±{(np.exp(q) - 1) * 100:.0f}% band (mean conformal q)")

    report = {
        "dataset": info,
        "cv_grouped": cv[cv.cv == "grouped"].set_index("model")[
            ["R2", "ratio_mean", "ratio_CoV", "unsafe_%", "log_RMSE", "log_RMSE_sd"]].round(4).to_dict("index"),
        "cv_random": cv[cv.cv == "random"].set_index("model")[
            ["R2", "ratio_CoV", "log_RMSE"]].round(4).to_dict("index"),
        "nested_tuned_ec2_xgb": {k: round(v, 4) for k, v in nested.items()},
        "sensitivity_known_mode_only": sens,
        "powerlaw_full_fit": {"intercept": float(pl_full.lr_.intercept_),
                              **dict(zip(["d", "fc", "rho_l", "u1", "r_over_d"],
                                         map(float, pl_full.lr_.coef_), strict=True))},
        "powerlaw_exponent_90CI": boot.quantile([0.05, 0.95]).round(3).to_dict(),
        "ec2_full_data": {k: round(v, 4) for k, v in metrics(y, capacity_kN(X)).items()},
        "conformal_90": conf,
        "shap_top": shap_imp.head(5).round(4).to_dict(),
        "runtime_s": round(time.time() - t0, 1),
    }
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[done] {report['runtime_s']} s")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    run(fast=ap.parse_args().fast)
