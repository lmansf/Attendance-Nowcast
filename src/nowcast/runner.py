"""Run the 1D attendance nowcast over one day or many days.

Each hour:
    predict   x⁻ = x,  P⁻ = P + Q             (Q = (q × x)², a relative shock size)
    measure   z_t = C_t / f_t                 (skipped while f_t < min_share)
    update    K = P⁻ / (P⁻ + R_t),  x = x⁻ + K (z_t − x⁻),  P = (1 − K) P⁻
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nowcast.kalman import predict, update


def nowcast_day(hours, counts, f: pd.Series, x0: float, P0: float, R,
                q: float = 0.0, min_share: float = 0.10) -> pd.DataFrame:
    """Filter one day's cumulative counts. Returns one row per hour.

    hours, counts : reading hours and cumulative gate counts C_t
    f             : arrival curve, indexed by hour
    x0, P0        : prior (pre-open forecast and its variance)
    R             : a function R(hour, x_prior) -> variance (see noise.py)
    q             : process noise as a fraction of the estimate per hour
    min_share     : skip the update while f_t is below this (z_t too unstable)
    """
    x, P = float(x0), float(P0)
    rows = []
    for hour, C in zip(hours, counts):
        x, P = predict(x, P, Q=(q * x) ** 2)
        f_t = float(f[hour])
        if f_t < min_share:
            z, R_t, K = np.nan, np.nan, 0.0
        else:
            z = C / f_t
            R_t = R(hour, x)
            x, P, K = update(x, P, z, R_t)
        rows.append({"hour": int(hour), "count": int(C), "f": f_t, "z": z,
                     "R": R_t, "K": K, "x": x, "P": P})
    return pd.DataFrame(rows)


def nowcast_days(hourly: pd.DataFrame, days: pd.DataFrame, f: pd.Series, R,
                 q: float = 0.0, min_share: float = 0.10) -> pd.DataFrame:
    """Run ``nowcast_day`` for every row of ``days``.

    ``days`` needs visit_date, forecast, P0 and attendance (the truth, used only
    for scoring - it is never shown to the filter).
    ``f`` is one arrival curve for every day, or a dict of curves keyed by day
    type ("peak"/"regular"), in which case ``days`` also needs a day_type column.
    """
    grouped = dict(tuple(hourly.groupby("visit_date")))
    out = []
    for day in days.itertuples(index=False):
        h = grouped[day.visit_date]
        f_day = f[day.day_type] if isinstance(f, dict) else f
        res = nowcast_day(h["hour"].to_numpy(), h["cumulative_entries"].to_numpy(),
                          f_day, day.forecast, day.P0, R, q=q, min_share=min_share)
        res.insert(0, "visit_date", day.visit_date)
        res["forecast"] = day.forecast
        res["truth"] = day.attendance
        out.append(res)
    return pd.concat(out, ignore_index=True)
