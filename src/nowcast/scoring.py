"""Error metrics by hour of day for the nowcast."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_by_hour(results: pd.DataFrame) -> pd.DataFrame:
    """Per hour: MAE of forecast-only vs filtered, and ±2σ band coverage.

    ``results`` is the output of ``runner.nowcast_days``.
    coverage_2sd is the share of days whose true total falls inside x ± 2√P;
    for a well-calibrated filter it should be near 95%.
    """
    df = results.assign(
        fc_abs_err=(results["forecast"] - results["truth"]).abs(),
        kf_abs_err=(results["x"] - results["truth"]).abs(),
        inside=(results["x"] - results["truth"]).abs() <= 2 * np.sqrt(results["P"]),
    )
    g = df.groupby("hour")
    table = pd.DataFrame({
        "mae_forecast": g["fc_abs_err"].mean(),
        "mae_filtered": g["kf_abs_err"].mean(),
        "coverage_2sd": g["inside"].mean(),
    })
    table["improvement_pct"] = 100 * (1 - table["mae_filtered"] / table["mae_forecast"])
    return table
