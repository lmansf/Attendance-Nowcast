"""Shared Matplotlib styling so every notebook's charts read the same way.

Colours come from one fixed categorical order and are assigned by *role*
(estimate, measurement, truth...), never cycled, so "the estimate" is the same
blue in every chart of the series.
"""

import matplotlib.pyplot as plt

COLORS = {
    "estimate": "#2a78d6",     # blue   - filter estimate and its band
    "measurement": "#eb6834",  # orange - raw measurements / implied totals
    "forecast": "#1baf7a",     # aqua   - pre-open forecast (the prior)
    "alt": "#4a3aa7",          # violet - a second variant being compared
    "truth": "#0b0b0b",        # ink    - the true value (dashed)
    "muted": "#52514e",        # secondary ink for annotations
}


def apply_style() -> None:
    """Recessive grid and axes, thin lines, readable default size."""
    plt.rcParams.update({
        "figure.figsize": (9, 4.5),
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8a8984",
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.8,
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "legend.frameon": False,
        "font.size": 10,
    })
