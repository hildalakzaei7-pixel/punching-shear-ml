"""Grouped evaluation: random vs source-grouped CV, nested tuning, ratio statistics, conformal intervals."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold

from .models import SEARCH_SPACE, SEED, EC2Residual, xgb


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    r = y / p
    return {
        "R2": r2_score(y, p),
        "RMSE_kN": float(np.sqrt(mean_squared_error(y, p))),
        "MAE_kN": mean_absolute_error(y, p),
        "ratio_mean": float(r.mean()),            # V_test / V_pred
        "ratio_CoV": float(r.std(ddof=1) / r.mean()),
        "unsafe_%": float(np.mean(r < 1.0) * 100),  # prediction above test value
        "log_RMSE": float(np.sqrt(np.mean(np.log(r) ** 2))),
    }


def splitter(kind, n_splits, seed):
    if kind == "random":
        return KFold(n_splits, shuffle=True, random_state=seed)
    return GroupKFold(n_splits, shuffle=True, random_state=seed)


def oof_predictions(model, X, y, groups, kind="grouped", n_splits=5, n_repeats=5) -> np.ndarray:
    """Out-of-fold predictions, one column per repeat."""
    P = np.zeros((len(y), n_repeats))
    for r in range(n_repeats):
        cv = splitter(kind, n_splits, SEED + r)
        split = cv.split(X, y, groups) if kind == "grouped" else cv.split(X, y)
        for tr, te in split:
            m = clone(model).fit(X.iloc[tr], y.iloc[tr])
            P[te, r] = m.predict(X.iloc[te])
    return P


def summarize(y, P) -> dict:
    """Metrics per repeat -> mean and sd across repeats."""
    rows = pd.DataFrame([metrics(y, P[:, r]) for r in range(P.shape[1])])
    out = {f"{k}": rows[k].mean() for k in rows}
    out.update({f"{k}_sd": rows[k].std() for k in ("R2", "log_RMSE", "ratio_CoV")})
    return out


def tune_residual(X, y, groups, n_iter=25, seed=SEED) -> dict:
    """Random search of the EC2-correction XGBoost with grouped inner CV on log ratio error."""
    rng = np.random.default_rng(seed)
    grid = list(itertools.product(*SEARCH_SPACE.values()))
    best, best_score = None, np.inf
    for idx in rng.choice(len(grid), size=min(n_iter, len(grid)), replace=False):
        params = dict(zip(SEARCH_SPACE, grid[idx], strict=True))
        P = oof_predictions(EC2Residual(xgb(**params)), X, y, groups, "grouped", 5, 1)
        s = float(np.sqrt(np.mean(np.log(y.to_numpy() / P[:, 0]) ** 2)))
        if s < best_score:
            best, best_score = params, s
    return {"params": best, "inner_log_RMSE": best_score}


def nested_tuned_oof(X, y, groups, n_repeats=3, n_iter=25) -> tuple[np.ndarray, list]:
    """Outer grouped CV; tuning happens inside each outer training fold only."""
    P = np.zeros((len(y), n_repeats))
    chosen = []
    for r in range(n_repeats):
        for tr, te in GroupKFold(5, shuffle=True, random_state=SEED + r).split(X, y, groups):
            t = tune_residual(X.iloc[tr], y.iloc[tr], groups[tr], n_iter=n_iter, seed=SEED + r)
            chosen.append(t["params"])
            m = EC2Residual(xgb(**t["params"])).fit(X.iloc[tr], y.iloc[tr])
            P[te, r] = m.predict(X.iloc[te])
    return P, chosen


def conformal_study(model, X, y, groups, alpha=0.1, n_repeats=30, cal_frac=0.25) -> pd.DataFrame:
    """Grouped split-conformal on log scale: interval = V_hat * exp(+-q).
    Train / calibration / test are disjoint in *sources*, repeated over random group splits."""
    rows = []
    for r in range(n_repeats):
        rest, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=1000 + r).split(X, y, groups))
        tr_i, cal_i = next(GroupShuffleSplit(1, test_size=cal_frac, random_state=2000 + r)
                           .split(X.iloc[rest], y.iloc[rest], groups[rest]))
        tr, cal = rest[tr_i], rest[cal_i]
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        s = np.abs(np.log(y.iloc[cal].to_numpy() / m.predict(X.iloc[cal])))
        n = len(s)
        q = float(np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))
        p = m.predict(X.iloc[te])
        yt = y.iloc[te].to_numpy()
        lo, hi = p * np.exp(-q), p * np.exp(q)
        rows.append({"repeat": r, "q_log": q, "factor": float(np.exp(q)),
                     "coverage": float(np.mean((yt >= lo) & (yt <= hi))), "n_test": len(te),
                     "n_cal": n})
    return pd.DataFrame(rows)
