"""SIMULATED data for a fictional park in Tampa, Florida.

Nothing here is real attendance. Every table this module returns has
``source = "simulation"`` so it can never be mistaken for production rows.

The park is placed in Florida so the same feature families as the
Attendance-Projections model apply: Florida weather (hot, wet summers with
afternoon thunderstorms), US + Florida holidays, weekday/weekend and season.

How a simulated day is made:

1. **Weather:** Tampa-like temperature, humidity and rain; in the wet season
   (Jun–Sep) some rainy days bring an afternoon thunderstorm.
2. **Planned demand:** base level × season × weekday × holiday × weather × noise.
3. **Hourly arrivals:** spread over the day with an arrival curve (see
   ``arrivals.py``), with a random daily ``pace`` (early/late crowd). A storm cuts
   arrivals after it hits, so the realized total ends below planned demand.
4. **Realized final attendance:** the last cumulative count; this becomes the
   ``daily_attendance`` row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nowcast.arrivals import PEAK_DAY, REGULAR_DAY, simulate_day_counts
from nowcast.daytype import florida_holidays

SOURCE = "simulation"

PARK = {
    "park_id": "SIM_TAMPA_FL",
    "name": "Simulated park (Tampa, FL)",
    "timezone": "America/New_York",
    "open_hour": REGULAR_DAY.open_hour,
    "close_hour": REGULAR_DAY.close_hour,
}

BASE_DEMAND = 3500           # guests on an ordinary mid-season weekday
ANNUAL_GROWTH = 0.03         # 3% more guests per year
DEMAND_NOISE_SD = 0.10       # day-to-day randomness nobody can forecast
PACE_SD = 0.12               # spread of early/late crowds (see ArrivalCurve)

# Florida season: spring break peak, summer vacation, quiet September.
MONTH_FACTOR = {1: 0.85, 2: 0.95, 3: 1.30, 4: 1.20, 5: 0.95, 6: 1.10,
                7: 1.15, 8: 0.85, 9: 0.65, 10: 0.80, 11: 0.90, 12: 1.10}
WEEKDAY_FACTOR = {0: 0.80, 1: 0.75, 2: 0.80, 3: 0.85, 4: 1.00, 5: 1.55, 6: 1.40}
HOLIDAY_FACTOR = 1.40


def simulate_weather(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """SIMULATED daily weather with a Tampa-like climate.

    Also returns ``storm_hour``: the clock hour an afternoon thunderstorm hits,
    or NaN. Storms happen on some wet-season rainy days.
    """
    doy = dates.dayofyear.to_numpy()
    wet_season = dates.month.isin([6, 7, 8, 9])

    # Highs around 21 °C in January and 33 °C in late July.
    temp_max = 27.0 - 6.0 * np.cos(2 * np.pi * (doy - 20) / 365.25) + rng.normal(0, 1.5, len(dates))
    temp_min = temp_max - 9.0 + rng.normal(0, 1.0, len(dates))
    humidity = np.clip(np.where(wet_season, 78, 68) + rng.normal(0, 6, len(dates)), 35, 100)

    rain_prob = np.where(wet_season, 0.60, 0.20)
    rainy = rng.random(len(dates)) < rain_prob
    precip = np.where(rainy, rng.gamma(0.8, 8.0, len(dates)), 0.0)

    storm = rainy & wet_season & (rng.random(len(dates)) < 0.5)
    storm_hour = np.where(storm, rng.integers(13, 17, len(dates)), np.nan)
    precip = np.where(storm, precip + rng.uniform(15, 40, len(dates)), precip)

    return pd.DataFrame({
        "visit_date": dates,
        "temp_max_c": temp_max.round(1),
        "temp_min_c": temp_min.round(1),
        "humidity_pct": humidity.round(0),
        "precip_mm": precip.round(1),
        "storm_hour": storm_hour,
    })


def planned_demand(row, is_holiday: bool, start_year: int) -> float:
    """Expected guests before noise: base × season × weekday × holiday × weather."""
    demand = BASE_DEMAND * (1 + ANNUAL_GROWTH) ** (row.visit_date.year - start_year)
    demand *= MONTH_FACTOR[row.visit_date.month]
    demand *= WEEKDAY_FACTOR[row.visit_date.dayofweek]
    if is_holiday:
        demand *= HOLIDAY_FACTOR
    if row.temp_max_c > 33:                                   # very hot: some stay home
        demand *= 0.92
    if row.precip_mm > 0 and np.isnan(row.storm_hour):        # rain in the forecast
        demand *= 0.85
    return demand


def simulate_park(start: str = "2022-01-01", end: str = "2024-12-31",
                  seed: int = 42) -> dict[str, pd.DataFrame]:
    """Build every SIMULATED table for the Tampa park.

    Returns a dict of DataFrames keyed by table name, matching ``schema.sql``.
    Dates are ISO strings (``YYYY-MM-DD``) so they load into any SQL database.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    hol = florida_holidays(range(dates.year.min(), dates.year.max() + 1))
    weather = simulate_weather(dates, rng)

    daily_rows, hourly_rows, log_rows = [], [], []
    for row in weather.itertuples(index=False):
        is_holiday = row.visit_date in hol
        is_peak = is_holiday or row.visit_date.dayofweek >= 5
        curve = PEAK_DAY if is_peak else REGULAR_DAY

        demand = planned_demand(row, is_holiday, dates.year.min())
        demand *= np.exp(rng.normal(0.0, DEMAND_NOISE_SD))
        pace = rng.normal(0.0, PACE_SD)
        storm_hour = None if np.isnan(row.storm_hour) else int(row.storm_hour)

        counts = simulate_day_counts(demand, curve, rng, pace=pace, storm_hour=storm_hour)

        day = row.visit_date.strftime("%Y-%m-%d")
        daily_rows.append({"visit_date": day, "attendance": int(counts[-1])})
        for hour, c in zip(curve.hours, counts):
            hourly_rows.append({"visit_date": day, "hour": int(hour), "cumulative_entries": int(c)})
        log_rows.append({"visit_date": day, "planned_demand": round(float(demand), 1),
                         "pace": round(float(pace), 4), "storm_hour": storm_hour})

    weather = weather.drop(columns="storm_hour")
    weather["visit_date"] = weather["visit_date"].dt.strftime("%Y-%m-%d")

    tables = {
        "parks": pd.DataFrame([PARK]),
        "daily_attendance": pd.DataFrame(daily_rows),
        "hourly_entries": pd.DataFrame(hourly_rows),
        "daily_weather": weather,
        "simulation_log": pd.DataFrame(log_rows),
    }
    for df in tables.values():
        if "park_id" not in df.columns:
            df.insert(0, "park_id", PARK["park_id"])
        df["source"] = SOURCE
    return tables
