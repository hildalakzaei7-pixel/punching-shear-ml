"""Punching resistance of slabs without shear reinforcement, EN 1992-1-1:2004, clause 6.4.4.

    v_Rd,c = C_Rd,c * k * (100 * rho_l * f_ck)^(1/3) >= v_min
    k      = 1 + sqrt(200 / d) <= 2.0          (d in mm)
    rho_l  <= 0.02
    v_min  = 0.035 * k^(3/2) * f_ck^(1/2)
    V      = v * u1 * d,   u1 = basic control perimeter at 2d from the column face

For comparison with tests, *mean* values are used: C_Rd,c = 0.18 (gamma_c = 1.0) and f_ck -> f_cm.
This is a common way to assess a design model against test data; it is not a design check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def control_perimeter(b, d, circular):
    """u1 at 2d: square column of side b -> 4b + 4*pi*d; circular of diameter b -> pi*(b + 4d)."""
    b, d, circular = np.asarray(b, float), np.asarray(d, float), np.asarray(circular, bool)
    return np.where(circular, np.pi * (b + 4 * d), 4 * b + 4 * np.pi * d)


def v_rdc(fc, rho_l_pct, d, c_rdc=0.18, cap_k=True, cap_rho=True):
    """Shear stress resistance in MPa. rho_l given in percent, as in the database."""
    fc, d = np.asarray(fc, float), np.asarray(d, float)
    rho = np.asarray(rho_l_pct, float) / 100.0
    k = 1 + np.sqrt(200.0 / d)
    if cap_k:
        k = np.minimum(k, 2.0)
    if cap_rho:
        rho = np.minimum(rho, 0.02)
    v = c_rdc * k * np.cbrt(100 * rho * fc)
    v_min = 0.035 * k**1.5 * np.sqrt(fc)
    return np.maximum(v, v_min)


def capacity_kN(df: pd.DataFrame, c_rdc: float = 0.18) -> np.ndarray:
    u1 = control_perimeter(df["b"], df["d"], df["circular"])
    return v_rdc(df["fc"], df["rho_l"], df["d"], c_rdc=c_rdc) * u1 * df["d"].to_numpy(float) / 1000.0
