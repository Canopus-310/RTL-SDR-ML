"""
One plotting style for every saved figure (model comparison and spectrum plots).

    apply_style()                rcParams: white background, >= 10 pt text, light grid
    MODEL_ORDER / MODEL_COLORS / MODEL_LABELS   same order and colours in every figure
    CLASS_ORDER                  class order in every per-class figure/table
    save_figure(fig, path)       300 dpi PNG + vector PDF; returns both paths

Figures are drawn at their final size: SINGLE_COL (3.5 in) or DOUBLE_COL
(7.2 in) wide, so 10 pt text stays 10 pt in the document. Uses the
matplotlib Figure API (no pyplot), so it works headless and in background threads.
"""

import os

import matplotlib

SINGLE_COL = 3.5   # inches
DOUBLE_COL = 7.2

MODEL_ORDER = ["rf", "xgb", "lgbm", "cat", "ensemble", "ensemble_uncal", "rf_deployed"]
MODEL_LABELS = {
    "rf": "RF (tuned)", "xgb": "XGBoost", "lgbm": "LightGBM", "cat": "CatBoost",
    "ensemble": "Ensemble (calibrated)", "ensemble_uncal": "Ensemble (uncalibrated)",
    "rf_deployed": "RF (deployed)",
}
# Okabe-Ito colour-blind-safe palette
MODEL_COLORS = {
    "rf": "#0072B2", "xgb": "#D55E00", "lgbm": "#009E73", "cat": "#CC79A7",
    "ensemble": "#E69F00", "ensemble_uncal": "#56B4E9", "rf_deployed": "#7F7F7F",
}
CLASS_ORDER = ["FM_BROADCAST", "DRONE_TELEMETRY", "DRONE_VIDEO", "INTERFERENCE", "NOISE", "LORA"]

STYLE = {
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.6,
    "legend.framealpha": 0.9,
    "pdf.fonttype": 42,      # embed TrueType so text stays editable/searchable in the PDF
}


def apply_style():
    matplotlib.rcParams.update(STYLE)


def save_figure(fig, path_without_ext):
    """Save fig as <path>.png (300 dpi) and <path>.pdf (vector). Creates folders; returns the paths."""
    os.makedirs(os.path.dirname(path_without_ext) or ".", exist_ok=True)
    paths = [path_without_ext + ".png", path_without_ext + ".pdf"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    return paths
