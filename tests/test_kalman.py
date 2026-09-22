import numpy as np
import pytest

from nowcast.kalman import kalman_gain, predict, run_filter, update


@pytest.mark.parametrize("P_prior", [1e-6, 0.5, 1.0, 1e3, 1e9])
@pytest.mark.parametrize("R", [1e-6, 0.5, 1.0, 1e3, 1e9])
def test_gain_between_0_and_1_and_variance_shrinks(P_prior, R):
    x, P, K = update(10.0, P_prior, 12.0, R)
    assert 0.0 <= K <= 1.0
    assert P <= P_prior


def test_predict_grows_variance_only():
    x, P = predict(5.0, 2.0, Q=0.5)
    assert x == 5.0
    assert P == 2.5


def test_huge_R_keeps_prior():
    x, P, K = update(100.0, 25.0, z=500.0, R=1e12)
    assert x == pytest.approx(100.0, abs=1e-6)
    assert P == pytest.approx(25.0, rel=1e-6)


def test_tiny_R_snaps_to_measurement():
    x, P, K = update(100.0, 25.0, z=500.0, R=1e-12)
    assert x == pytest.approx(500.0, abs=1e-6)
    assert P == pytest.approx(0.0, abs=1e-6)


def test_equal_variances_average():
    x, P, K = update(0.0, 4.0, z=10.0, R=4.0)
    assert K == 0.5
    assert x == 5.0
    assert P == 2.0


def test_run_filter_skips_nan_measurements():
    out = run_filter(0.0, 1.0, zs=[np.nan, 2.0], Rs=1.0, Q=0.0)
    assert out["K"][0] == 0.0
    assert out["x"][0] == 0.0 and out["P"][0] == 1.0
    assert out["x"][1] == pytest.approx(1.0)


def test_run_filter_variance_nonincreasing_without_Q():
    out = run_filter(0.0, 10.0, zs=np.ones(20), Rs=2.0, Q=0.0)
    assert np.all(np.diff(out["P"]) <= 0)
    assert kalman_gain(1.0, 1.0) == 0.5
