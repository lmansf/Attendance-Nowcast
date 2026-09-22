"""Calendar helpers: which kind of day is it?

Crowds arrive later on weekends and holidays, so the arrival curve f_t is
estimated separately for "peak" days (weekend or holiday) and "regular" days.
The day type is known before opening, so using it is not cheating.
"""

from __future__ import annotations

import holidays as holidays_lib
import pandas as pd


def florida_holidays(years) -> holidays_lib.HolidayBase:
    """US federal holidays plus Florida state holidays."""
    return holidays_lib.country_holidays("US", subdiv="FL", years=list(years))


def day_type(dates) -> pd.Series:
    """'peak' for weekends and Florida holidays, else 'regular'."""
    dates = pd.to_datetime(pd.Series(dates))
    hol = florida_holidays(range(dates.dt.year.min(), dates.dt.year.max() + 1))
    is_peak = (dates.dt.dayofweek >= 5) | dates.apply(lambda d: d in hol)
    return pd.Series(["peak" if p else "regular" for p in is_peak], index=dates.index)
