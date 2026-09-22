"""Measurement noise R_t: how much to trust each hour's implied total.

At hour t we see the cumulative count C_t. If today follows the historical
arrival curve, the final total should be about

    z_t = C_t / f_t          (the "implied final total")

Early in the day f_t is small, so any wobble in C_t is divided by a small number
and blown up: z_t is noisy. By late afternoon f_t is near 1 and z_t is reliable.

We measure that empirically on past days (where the true final total is known)
and turn it into R_t in one of three ways ("schemes"). Each scheme is a small
function ``R(hour, x_prior) -> variance`` that the filter calls every hour:

* ``constant_R``  — one variance for every hour (the naive choice)
* ``absolute_R``  — a variance per hour, in guests²
* ``relative_R``  — a relative sd per hour, scaled by today's estimate:
                    R_t = (s_t × x⁻)²   (busy days get proportionally more noise)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

R_FLOOR = 1.0   # guests²; keeps K strictly below 1 even when history says "no noise"


def implied_totals(hourly: pd.DataFrame, f: pd.Series) -> pd.Series:
    """z_t = C_t / f_t for every row of ``hourly`` (f is indexed by hour)."""
    return hourly["cumulative_entries"] / hourly["hour"].map(f)


def implied_errors(hourly: pd.DataFrame, daily: pd.DataFrame, f) -> pd.DataFrame:
    """Per day and hour: z_t, the true final total, and the errors of z_t.

    ``f`` is one curve, or a dict of curves keyed by day type, in which case
    ``daily`` needs a day_type column.
    """
    cols = ["visit_date", "attendance"] + (["day_type"] if isinstance(f, dict) else [])
    df = hourly.merge(daily[cols], on="visit_date")
    if isinstance(f, dict):
        parts = [implied_totals(g, f[t]) for t, g in df.groupby("day_type")]
        df["z"] = pd.concat(parts)
    else:
        df["z"] = implied_totals(df, f)
    df["abs_error"] = df["z"] - df["attendance"]
    df["rel_error"] = df["z"] / df["attendance"] - 1.0
    return df


def noise_by_hour(errors: pd.DataFrame) -> pd.DataFrame:
    """Summarize implied-total errors by hour: bias, absolute variance, relative sd."""
    g = errors.groupby("hour")
    return pd.DataFrame({
        "rel_bias": g["rel_error"].mean(),
        "rel_sd": g["rel_error"].std(),
        "abs_var": g["abs_error"].var(),
    })


def constant_R(variance: float):
    """Same R for every hour."""
    return lambda hour, x_prior: max(variance, R_FLOOR)


def absolute_R(abs_var: pd.Series):
    """R_t looked up per hour, in guests²."""
    return lambda hour, x_prior: max(float(abs_var[hour]), R_FLOOR)


def relative_R(rel_sd: pd.Series):
    """R_t = (s_t × x⁻)²: the hour's relative sd scaled by the current estimate."""
    return lambda hour, x_prior: max(float(rel_sd[hour] * x_prior) ** 2, R_FLOOR)
