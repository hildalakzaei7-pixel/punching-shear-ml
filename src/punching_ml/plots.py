"""Figure styling shared by the pipeline (same validated palette as the concrete-strength project)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Categorical order fixed; validated for CVD separation (see README).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"



def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def logx(ax, ticks):
    from matplotlib.ticker import NullFormatter
    ax.set_xscale("log")
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def eda(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(2, 3, figsize=(11, 6))
    spec = [("d", "Effective depth d (mm)"), ("b", "Column size b (mm)"), ("fc", "f_c (MPa)"),
            ("rho_l", "Flexural ratio ρ (%)"), ("r_load", "Load radius r (mm)"), ("V_test", "Failure load (kN)")]
    for a, (c, lab) in zip(ax.ravel(), spec, strict=True):
        a.hist(df[c], bins=25, color=BLUE, edgecolor=SURFACE, linewidth=1.2)
        a.set(xlabel=lab, ylabel="Tests")
    fig.suptitle(f"Core dataset: {len(df)} tests from {df['source_id'].nunique()} test programmes",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def per_source(df: pd.DataFrame, path: Path):
    n = df.groupby("source_id").size().sort_values(ascending=False).to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 3))
    ax.bar(np.arange(len(n)), n, color=BLUE, width=0.8)
    ax.set(xlabel="Test programme (sorted)", ylabel="Tests",
           title="Tests come in series: a random split mixes each series into train and test")
    ax.set_xticks([])
    save(fig, path)


def cv_bars(table: pd.DataFrame, path: Path, metric="log_RMSE"):
    t = table.pivot(index="model", columns="cv", values=metric).sort_values("grouped")
    y = np.arange(len(t))
    h = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.barh(y - h / 2 - 0.01, t["random"], h, color=BLUE, label="Random K-fold (tests)",
            edgecolor=SURFACE)
    ax.barh(y + h / 2 + 0.01, t["grouped"], h, color=ORANGE, label="Grouped K-fold (unseen programmes)",
            edgecolor=SURFACE)
    for i, (r, g) in enumerate(zip(t["random"], t["grouped"], strict=True)):
        ax.text(r + 0.004, i - h / 2, f"{r:.3f}", va="center", fontsize=8, color=INK2)
        ax.text(g + 0.004, i + h / 2, f"{g:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, t.index)
    ax.invert_yaxis()
    ax.set_xlim(0, t.max().max() * 1.18)
    ax.set(xlabel="RMS of ln(V_test / V_pred), lower is better (≈ CoV)")
    ax.set_title("Random splits flatter data-driven models; grouped splits change the ranking", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=2, fontsize=9)
    ax.grid(axis="y", visible=False)
    save(fig, path)


def ratio_vs_d(df: pd.DataFrame, preds: dict, path: Path):
    fig, axes = plt.subplots(1, len(preds), figsize=(3.2 * len(preds), 3.4), sharey=True)
    for a, (name, p) in zip(axes, preds.items(), strict=True):
        r = df["V_test"].to_numpy() / p
        a.scatter(df["d"], r, s=12, color=BLUE, alpha=0.7, edgecolor=SURFACE, linewidth=0.4)
        a.axhline(1, color=INK2, linewidth=1, linestyle="--")
        logx(a, [50, 100, 200, 400])
        cov = r.std(ddof=1) / r.mean()
        a.set_title(f"{name}\nmean {r.mean():.2f} · CoV {cov:.2f}", fontsize=9.5)
        a.set_xlabel("d (mm)")
    axes[0].set_ylabel("V_test / V_pred (grouped CV)")
    fig.suptitle("Grouped CV, one repeat: tree models trained on force under-predict the few thick slabs",
                 x=0.01, ha="left", fontweight="bold", y=1.08)
    save(fig, path)


def parity(y, p, lo, hi, path, title):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    inside = (y >= lo) & (y <= hi)
    ax.vlines(y, lo, hi, color=GRID, linewidth=1.1, zorder=1)
    ax.scatter(y[inside], p[inside], s=18, color=BLUE, edgecolor=SURFACE, linewidth=0.4, zorder=2,
               label="Inside 90% interval")
    ax.scatter(y[~inside], p[~inside], s=24, color=ORANGE, marker="D", edgecolor=SURFACE,
               linewidth=0.4, zorder=3, label="Outside 90% interval")
    lim = [40, max(y.max(), p.max()) * 1.3]
    ax.plot(lim, lim, color=INK2, linewidth=1, linestyle="--")
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="Measured failure load (kN)",
           ylabel="Predicted (kN)")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9.5)
    ax.legend(loc="upper left")
    save(fig, path)


def exponents(boot: pd.DataFrame, path: Path):
    names = {"d": "d", "fc": "f_c", "rho_l": "ρ", "u1": "u₁", "r_over_d": "r/d"}
    q = boot.quantile([0.05, 0.5, 0.95]).T.loc[list(names)]
    ec2 = {"d": None, "fc": 1 / 3, "rho_l": 1 / 3, "u1": 1.0, "r_over_d": 0.0}
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(q))
    ax.hlines(y, q[0.05], q[0.95], color=BLUE, linewidth=3)
    ax.scatter(q[0.5], y, color=BLUE, s=40, zorder=3, label="Fitted exponent, 90% CI (bootstrap by programme)")
    for i, k in enumerate(q.index):
        if ec2[k] is not None:
            ax.scatter(ec2[k], i, marker="|", s=250, color=ORANGE, zorder=4,
                       label="EC2 exponent" if i == 1 else None)
    ax.set_yticks(y, [names[k] for k in q.index])
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set(xlabel="Exponent in V ∝ ∏ xᵢ^aᵢ")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=2, fontsize=8.5)
    ax.set_title("Fitted power law vs EC2 (6.4.4)", pad=24)
    ax.grid(axis="y", visible=False)
    save(fig, path)
