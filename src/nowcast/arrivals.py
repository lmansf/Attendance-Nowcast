"""Arrival curves and the intraday (hourly) arrival simulator.

An *arrival curve* ``f_t`` is the share of a day's guests who have come through
the gates by the end of clock hour ``t``. It starts near 0 after opening and
reaches exactly 1 at closing.

    f_t = F(t) / F(close)      with   F(t) = 1 − exp(−(hours_open / scale)^shape)

``F`` is a Weibull cumulative curve. A ``shape`` above 1 gives the realistic
pattern: a slow first hour, a morning surge, then a tapering afternoon.
``scale`` says how late the crowd arrives (bigger = later).

SIMULATED DATA: everything produced by ``simulate_day_counts`` is synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ArrivalCurve:
    """A configurable cumulative arrival curve for one kind of day."""

    open_hour: int = 9      # park opens at 09:00 local time
    close_hour: int = 18    # last reading is the count at 18:00 (closing)
    shape: float = 1.8      # >1 means slow start, then a surge
    scale: float = 3.0      # hours after opening by which ~63% have arrived

    @property
    def hours(self) -> np.ndarray:
        """Clock hours at which a cumulative count is read (hour *ending*)."""
        return np.arange(self.open_hour + 1, self.close_hour + 1)

    def cumulative(self, pace: float = 0.0) -> np.ndarray:
        """Cumulative share f_t for each reading hour.

        ``pace`` shifts the whole day: +0.2 means guests arrive about 22% later
        than usual (scale × e^0.2), −0.2 means earlier. The curve still ends at 1.
        """
        hours_open = self.hours - self.open_hour
        scale = self.scale * np.exp(pace)
        F = 1.0 - np.exp(-((hours_open / scale) ** self.shape))
        return F / F[-1]


# Weekends and holidays: families arrive a little later in the morning.
REGULAR_DAY = ArrivalCurve(scale=3.0)
PEAK_DAY = ArrivalCurve(scale=3.4)


def simulate_day_counts(
    total: float,
    curve: ArrivalCurve,
    rng: np.random.Generator,
    noise_sd: float = 0.15,
    pace: float = 0.0,
    storm_hour: int | None = None,
    storm_keep: float = 0.3,
) -> np.ndarray:
    """SIMULATED cumulative gate counts for one day, one value per reading hour.

    1. Split ``total`` across hours using the curve (shifted by ``pace``).
    2. Multiply each hour's arrivals by lognormal noise, so no two days match,
       then rescale so the day still adds up to ``total``.
    3. If ``storm_hour`` is set, an afternoon storm keeps only ``storm_keep`` of
       the arrivals in the hours after it. The day then ends *below* ``total``:
       a genuine surprise the pre-open forecast could not know about.
    4. Accumulate and round to whole guests.

    Increments are never negative, so the counts never decrease. The last value
    is the realized final attendance.
    """
    shares = np.diff(np.concatenate([[0.0], curve.cumulative(pace)]))
    noise = np.exp(rng.normal(0.0, noise_sd, size=shares.size))
    hourly = shares * noise
    hourly = total * hourly / hourly.sum()

    if storm_hour is not None:
        after_storm = curve.hours > storm_hour
        hourly[after_storm] *= storm_keep

    return np.round(np.cumsum(hourly)).astype(int)


def estimate_curve(hourly: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """Estimate f_t from history: the average share of the final total seen by hour t.

    ``hourly`` has columns visit_date, hour, cumulative_entries.
    ``daily`` has columns visit_date, attendance (the final total).
    Returns a Series indexed by hour.
    """
    merged = hourly.merge(daily[["visit_date", "attendance"]], on="visit_date")
    merged["share"] = merged["cumulative_entries"] / merged["attendance"]
    return merged.groupby("hour")["share"].mean()
