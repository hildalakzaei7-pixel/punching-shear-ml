"""Code baseline, calibrated mechanics-based regression, and ML models.

All learners fit in log space (log V); errors in punching tests scale with capacity.
The 'EC2 + XGBoost' model is physics-informed: it learns only a multiplicative correction
to the EC2 prediction, V = V_EC2 * exp(f(x)), so it falls back to the code when f ~ 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from .data import FEATURES
from .ec2 import capacity_kN, control_perimeter

SEED = 42


class Features(BaseEstimator, TransformerMixin):
    """Raw inputs plus mechanics-motivated, dimensionless groups."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X[FEATURES].astype(float).copy()
        u1 = control_perimeter(X["b"], X["d"], X["circular"])
        X["u1_over_d"] = u1 / X["d"]
        X["b_over_d"] = X["b"] / X["d"]
        X["r_over_d"] = X["r_load"] / X["d"]
        X["log_d"] = np.log(X["d"])
        X["rho_fc_cbrt"] = np.cbrt(X["rho_l"] * X["fc"])
        return X


class EC2Mean(BaseEstimator, RegressorMixin):
    """EN 1992-1-1 6.4.4 with mean values; nothing is fitted."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return capacity_kN(X)


class EC2Calibrated(BaseEstimator, RegressorMixin):
    """EC2 form with C_Rd,c refitted on training data (geometric-mean ratio)."""

    def fit(self, X, y):
        self.c_ = 0.18 * float(np.exp(np.mean(np.log(np.asarray(y) / capacity_kN(X)))))
        return self

    def predict(self, X):
        return capacity_kN(X, c_rdc=self.c_)


class PowerLaw(BaseEstimator, RegressorMixin):
    """V = A * d^a * u1^b * f_c^c * rho^e * (r/d)^g, linear in log space."""

    cols = ["d", "fc", "rho_l"]

    def _design(self, X):
        u1 = control_perimeter(X["b"], X["d"], X["circular"])
        Z = np.log(X[self.cols].astype(float).to_numpy())
        return np.column_stack([Z, np.log(u1), np.log(X["r_load"] / X["d"])])

    def fit(self, X, y):
        self.lr_ = LinearRegression().fit(self._design(X), np.log(y))
        return self

    def predict(self, X):
        return np.exp(self.lr_.predict(self._design(X)))


class LogTarget(BaseEstimator, RegressorMixin):
    """Tree model on engineered features, trained on log V."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.fx_ = Features()
        self.est_ = self.estimator.fit(self.fx_.transform(X), np.log(y))
        return self

    def predict(self, X):
        return np.exp(self.est_.predict(self.fx_.transform(X)))


class EC2Residual(BaseEstimator, RegressorMixin):
    """Physics-informed: XGBoost learns log(V_test / V_EC2)."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.fx_ = Features()
        self.est_ = self.estimator.fit(self.fx_.transform(X), np.log(np.asarray(y) / capacity_kN(X)))
        return self

    def predict(self, X):
        return capacity_kN(X) * np.exp(self.est_.predict(self.fx_.transform(X)))


class StressTarget(BaseEstimator, RegressorMixin):
    """Tree model on the nominal shear stress V / (u1 d): removes the size of the specimen from
    the target so the trees need not extrapolate in force."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    @staticmethod
    def _area(X):
        return control_perimeter(X["b"], X["d"], X["circular"]) * X["d"].to_numpy(float) / 1000.0

    def fit(self, X, y):
        self.fx_ = Features()
        self.est_ = self.estimator.fit(self.fx_.transform(X), np.log(np.asarray(y) / self._area(X)))
        return self

    def predict(self, X):
        return self._area(X) * np.exp(self.est_.predict(self.fx_.transform(X)))


def xgb(**kw) -> XGBRegressor:
    p = {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 3, "subsample": 0.8,
         "colsample_bytree": 0.8, "min_child_weight": 3, "reg_lambda": 1.0,
         "random_state": SEED, "n_jobs": 4, "tree_method": "hist"}
    p.update(kw)
    return XGBRegressor(**p)


def build_models() -> dict:
    return {
        "EC2 (mean values)": EC2Mean(),
        "EC2 (C refitted)": EC2Calibrated(),
        "Power law (log-linear)": PowerLaw(),
        "Random forest": LogTarget(RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                                         n_jobs=4, random_state=SEED)),
        "XGBoost": LogTarget(xgb()),
        "XGBoost (stress target)": StressTarget(xgb()),
        "EC2 + XGBoost correction": EC2Residual(xgb(n_estimators=300, max_depth=2)),
    }


SEARCH_SPACE = {
    "n_estimators": [150, 300, 500],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.02, 0.05, 0.1],
    "min_child_weight": [1, 3, 6],
    "subsample": [0.7, 0.9],
    "reg_lambda": [0.5, 1.0, 5.0],
}


def as_frame(X) -> pd.DataFrame:
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=FEATURES)
