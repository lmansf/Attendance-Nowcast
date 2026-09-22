import numpy as np
import pandas as pd
import pytest

from nowcast.arrivals import PEAK_DAY, REGULAR_DAY, ArrivalCurve, estimate_curve, simulate_day_counts


@pytest.mark.parametrize("pace", [-0.3, 0.0, 0.3])
def test_curve_is_increasing_and_ends_at_one(pace):
    f = REGULAR_DAY.cumulative(pace)
    assert np.all(np.diff(f) > 0)
    assert f[0] > 0
    assert f[-1] == pytest.approx(1.0)


def test_late_pace_means_smaller_shares_early():
    assert REGULAR_DAY.cumulative(0.3)[2] < REGULAR_DAY.cumulative(-0.3)[2]
    assert PEAK_DAY.cumulative()[2] < REGULAR_DAY.cumulative()[2]


@pytest.mark.parametrize("seed", range(20))
def test_simulated_counts_non_decreasing_and_end_at_total(seed):
    rng = np.random.default_rng(seed)
    total = rng.integers(500, 10_000)
    counts = simulate_day_counts(total, REGULAR_DAY, rng, noise_sd=0.3, pace=rng.normal(0, 0.2))
    assert np.all(np.diff(counts) >= 0)
    assert counts[-1] == total
    assert len(counts) == len(REGULAR_DAY.hours)


def test_storm_cuts_the_final_total():
    counts = simulate_day_counts(5000, REGULAR_DAY, np.random.default_rng(0), storm_hour=13)
    assert np.all(np.diff(counts) >= 0)
    assert counts[-1] < 5000


def test_no_two_days_identical():
    rng = np.random.default_rng(1)
    a = simulate_day_counts(4000, REGULAR_DAY, rng)
    b = simulate_day_counts(4000, REGULAR_DAY, rng)
    assert not np.array_equal(a, b)


def test_estimate_curve_recovers_shape():
    curve = ArrivalCurve()
    rng = np.random.default_rng(3)
    hourly, daily = [], []
    for d in range(300):
        counts = simulate_day_counts(3000, curve, rng, noise_sd=0.1)
        hourly += [{"visit_date": d, "hour": h, "cumulative_entries": c} for h, c in zip(curve.hours, counts)]
        daily.append({"visit_date": d, "attendance": counts[-1]})
    f = estimate_curve(pd.DataFrame(hourly), pd.DataFrame(daily))
    np.testing.assert_allclose(f.to_numpy(), curve.cumulative(), atol=0.02)
