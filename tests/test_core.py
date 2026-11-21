"""Unit tests. They use a small synthetic sheet, so CI runs without the copyrighted raw file."""
import numpy as np
import pandas as pd
import pytest

from punching_ml.data import FEATURES, GROUP, RAW_XLSX, core_set, parse
from punching_ml.ec2 import capacity_kN, control_perimeter, v_rdc
from punching_ml.evaluate import conformal_study, metrics, oof_predictions
from punching_ml.models import build_models


def synthetic_sheet(n_sources=12, per_source=5, seed=0) -> pd.DataFrame:
    """Mimics the workbook: 3 header rows, reference only on the first row of each series."""
    rng = np.random.default_rng(seed)
    rows = [[None] * 22 for _ in range(3)]
    for s in range(n_sources):
        for i in range(per_source):
            d = rng.uniform(80, 300)
            row = [None] * 22
            row[0] = f"Author {s} (19{50 + s}). Tests." if i == 0 else None
            row[1], row[2], row[3] = f"S{s}-{i}", d + 30, d
            row[4], row[6] = rng.uniform(150, 400), rng.choice(["s", "c"])
            row[5] = row[4] * 1.128
            row[7], row[8], row[10] = rng.uniform(20, 60), rng.uniform(0.5, 2.0), 500
            row[19] = d * rng.uniform(5, 9)
            df_row = pd.DataFrame([{"b": row[4], "d": d, "circular": int(row[6] == "c"),
                                    "fc": row[7], "rho_l": row[8]}])
            row[20] = float(capacity_kN(df_row)[0]) * rng.lognormal(0.15, 0.15)
            row[21] = "S"
            rows.append(row)
    # one shear-reinforced, one fibre, one flexural failure: all must be excluded
    for k, (col, val) in enumerate([(15, 0.5), (17, 1.0), (21, "F")]):
        extra = list(rows[3 + k])
        extra[0] = f"Excluded {k}"
        extra[col] = val
        rows.append(extra)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def data():
    return core_set(parse(synthetic_sheet()))


def test_parse_forward_fills_reference_and_groups():
    df = parse(synthetic_sheet())
    assert df["reference"].notna().all()
    assert df[GROUP].nunique() == 12 + 3


def test_core_set_excludes_shear_fibre_flexural(data):
    assert len(data) == 60
    assert not data["reference"].str.startswith("Excluded").any()


def test_control_perimeter():
    assert control_perimeter(300, 200, False) == pytest.approx(4 * 300 + 4 * np.pi * 200)
    assert control_perimeter(300, 200, True) == pytest.approx(np.pi * (300 + 800))


def test_v_rdc_caps_and_minimum():
    # d = 100 mm -> k = 1 + sqrt(2) > 2, so capped at 2
    v = v_rdc(fc=30, rho_l_pct=1.0, d=100)
    assert v == pytest.approx(0.18 * 2.0 * (100 * 0.01 * 30) ** (1 / 3))
    # rho capped at 2 %
    assert v_rdc(30, 5.0, 300) == pytest.approx(v_rdc(30, 2.0, 300))
    # v_min governs at very low reinforcement
    k = 1 + np.sqrt(200 / 300)
    assert v_rdc(30, 0.01, 300) == pytest.approx(0.035 * k**1.5 * np.sqrt(30))


def test_capacity_units_kN():
    one = pd.DataFrame([{"b": 300, "d": 200, "circular": 0, "fc": 30, "rho_l": 1.0}])
    expected = v_rdc(30, 1.0, 200) * control_perimeter(300, 200, False) * 200 / 1000
    assert capacity_kN(one)[0] == pytest.approx(expected)


def test_grouped_cv_never_shares_sources(data):
    from sklearn.model_selection import GroupKFold
    g = data[GROUP].to_numpy()
    for tr, te in GroupKFold(5, shuffle=True, random_state=0).split(data, groups=g):
        assert set(g[tr]).isdisjoint(g[te])


def test_all_models_run_and_stay_positive(data):
    X, y, g = data[FEATURES + ["h"]], data["V_test"], data[GROUP].to_numpy()
    for name, m in build_models().items():
        P = oof_predictions(m, X, y, g, "grouped", n_splits=4, n_repeats=1)
        assert np.all(P > 0), name
        assert metrics(y, P[:, 0])["ratio_mean"] > 0


def test_conformal_runs(data):
    X, y, g = data[FEATURES + ["h"]], data["V_test"], data[GROUP].to_numpy()
    out = conformal_study(build_models()["Power law (log-linear)"], X, y, g, n_repeats=3)
    assert ((out.coverage >= 0) & (out.coverage <= 1)).all() and (out.factor > 1).all()


@pytest.mark.skipif(not RAW_XLSX.exists(), reason="raw workbook not present (not redistributed)")
def test_real_file_profile():
    from punching_ml.data import read_sheet
    full = read_sheet()
    assert len(full) == 543 and full[GROUP].nunique() == 54
    assert len(core_set(full)) == 228
