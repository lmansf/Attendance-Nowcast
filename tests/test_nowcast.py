"""End-to-end: on simulated data the filter must beat the forecast by closing time."""

import numpy as np
import pytest

from nowcast import db
from nowcast.arrivals import estimate_curve
from nowcast.forecast import relative_error_sd, same_weekday_forecast
from nowcast.noise import implied_errors, noise_by_hour, relative_R
from nowcast.runner import nowcast_days
from nowcast.scoring import score_by_hour


@pytest.fixture(scope="module")
def scores(tmp_path_factory):
    engine = db.get_engine(f"sqlite:///{tmp_path_factory.mktemp('db') / 'e2e.db'}")
    pid = db.ensure_simulated_data(engine)
    daily = db.load_daily(engine, pid, "simulation")
    hourly = db.load_hourly(engine, pid, "simulation")

    daily["forecast"] = same_weekday_forecast(daily)
    train = daily["visit_date"] < "2024-01-01"
    train_hourly = hourly[hourly["visit_date"] < "2024-01-01"]
    f = estimate_curve(train_hourly, daily[train])
    rel_sd = relative_error_sd(daily.loc[train, "attendance"], daily.loc[train, "forecast"])
    daily["P0"] = (rel_sd * daily["forecast"]) ** 2
    noise = noise_by_hour(implied_errors(train_hourly, daily[train], f))

    test = daily[~train].dropna()
    return score_by_hour(nowcast_days(hourly, test, f, relative_R(noise["rel_sd"])))


def test_final_hour_filtered_beats_forecast(scores):
    last = scores.iloc[-1]
    assert last["mae_filtered"] < last["mae_forecast"]


def test_filtered_beats_forecast_from_late_morning(scores):
    assert (scores.loc[11:, "mae_filtered"] < scores.loc[11:, "mae_forecast"]).all()


def test_error_shrinks_through_the_afternoon(scores):
    assert np.all(np.diff(scores.loc[12:, "mae_filtered"]) < 0)
