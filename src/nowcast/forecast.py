"""The pre-open forecast: the filter's prior.

A deliberately simple, transparent baseline:

    forecast(day) = mean attendance on the same weekday over the previous N weeks

Its uncertainty is estimated from its own past mistakes. Errors are measured as
*relative* errors (actual / forecast − 1) because a busy Saturday misses by more
guests than a quiet Tuesday. The prior variance for a day is then

    P0 = (relative_sd × forecast)²

The Attendance-Projections model (weather, calendar, lags) can later replace
this function; the filter only needs a forecast and its variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def same_weekday_forecast(daily: pd.DataFrame, n_weeks: int = 4) -> pd.Series:
    """Average of the last ``n_weeks`` same-weekday totals, using only earlier days.

    ``daily`` needs columns visit_date, attendance. Returns a Series aligned with
    ``daily`` (NaN until enough history exists).
    """
    by_date = daily.set_index("visit_date")["attendance"].asfreq("D")
    forecast = pd.Series(0.0, index=by_date.index)
    for k in range(1, n_weeks + 1):
        forecast += by_date.shift(7 * k)       # the same weekday, k weeks earlier
    forecast /= n_weeks
    return forecast.reindex(daily["visit_date"]).to_numpy()


def relative_error_sd(actual, forecast) -> float:
    """Standard deviation of (actual / forecast − 1) over days that have both."""
    rel = np.asarray(actual, float) / np.asarray(forecast, float) - 1.0
    return float(np.nanstd(rel))
