"""Loading and cleaning the TU Dortmund punching shear database.

Dataset: Spyridis, P., Mellios, N., Uz, O., & Walter, L. (2023). Database of experimental results from
punching shear tests with various types of reinforcement [Dataset]. https://doi.org/10.17877/DE290R-23085
The raw file is protected by original copyright and is NOT redistributed; see data/README.md.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_XLSX = ROOT / "data" / "raw" / "20230131_PunchingShearDataRev1-Spyridis.xlsx"
SHA256 = "6bf3c6f6b380b8fc560de50c07bbb3f6db25389f85571d5129b478b1d10f41bb"
DOWNLOAD_URL = "https://eldorado.tu-dortmund.de/bitstreams/0f08ec27-4ab9-407c-84be-da50323aeb32/download"

SHEET_COLUMNS = [
    "reference", "specimen", "h", "d", "b", "b_eq", "column_type", "fc", "rho_l", "rho_2", "fy",
    "s0", "s1", "ns", "beta", "rho_w", "fyw", "fibre_vol", "fibre_kg", "r_load", "V_test", "failure_mode",
]
NUMERIC = [c for c in SHEET_COLUMNS if c not in ("reference", "specimen", "column_type", "failure_mode")]
FEATURES = ["d", "b", "circular", "fc", "rho_l", "r_load"]
TARGET = "V_test"
GROUP = "source_id"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_sheet(path: Path | str = RAW_XLSX, check_hash: bool = True) -> pd.DataFrame:
    """Parse the workbook into one row per test with canonical column names and a source_id."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the file from https://doi.org/10.17877/DE290R-23085 "
            "and place it in data/raw/ (see data/README.md).")
    if check_hash and sha256(path) != SHA256:
        raise ValueError("Workbook checksum differs from the version analysed (Rev. 1, Nov 2022).")
    raw = pd.read_excel(path, header=None)
    return parse(raw)


def parse(raw: pd.DataFrame) -> pd.DataFrame:
    """Rows 0-2 are header/units/blank; the reference is given once per test series."""
    df = raw.iloc[3:, : len(SHEET_COLUMNS)].copy()
    df.columns = SHEET_COLUMNS
    df = df.dropna(how="all").reset_index(drop=True)
    df["reference"] = df["reference"].ffill().astype(str).str.strip()
    df[GROUP] = pd.factorize(df["reference"])[0]
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["column_type"] = df["column_type"].astype(str).str.strip().str.lower()
    df["failure_mode"] = df["failure_mode"].astype(str).str.strip().str.upper().replace({"NAN": np.nan})
    df["circular"] = (df["column_type"] == "c").astype(int)
    df["has_shear_reinf"] = df["rho_w"].fillna(0) > 0
    df["has_fibres"] = (df["fibre_vol"].fillna(0) > 0) | (df["fibre_kg"].fillna(0) > 0)
    df["mode_unknown"] = df["failure_mode"].isna()
    return df


def core_set(df: pd.DataFrame, drop_unknown_mode: bool = False) -> pd.DataFrame:
    """Interior slab-column tests without shear reinforcement or fibres, not flexural failures,
    with all model inputs present. See data/DATA_DICTIONARY.md for the rationale."""
    m = ~df["has_shear_reinf"] & ~df["has_fibres"] & (df["failure_mode"] != "F")
    if drop_unknown_mode:
        m &= ~df["mode_unknown"]
    m &= df["column_type"].isin(["c", "s"])
    out = df[m].dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    validate(out)
    return out


def validate(df: pd.DataFrame) -> None:
    if (df[["d", "b", "fc", "rho_l", "r_load", TARGET]] <= 0).any().any():
        raise ValueError("Non-positive geometry, material or load values")
    if (df["d"] > df["h"].fillna(np.inf)).any():
        raise ValueError("Effective depth exceeds slab thickness")
