"""A one-dimensional Kalman filter, written out step by step.

The state is a single number ``x`` (our best guess of some quantity) and its
variance ``P`` (how unsure we are about that guess). Each cycle has two steps:

* **predict** - time passes, the world might have shifted, so uncertainty grows.
* **update**  - a measurement arrives, and we blend it with our guess.

Everything here is plain Python floats and NumPy; no filter libraries.
"""

from __future__ import annotations

import numpy as np


def predict(x: float, P: float, Q: float) -> tuple[float, float]:
    """Predict step for a state we expect to stay constant.

    x⁻ = x        (no known dynamics: tomorrow's guess equals today's)
    P⁻ = P + Q    (but we are a little less sure, by the process noise Q)
    """
    x_prior = x
    P_prior = P + Q
    return x_prior, P_prior


def kalman_gain(P_prior: float, R: float) -> float:
    """K = P⁻ / (P⁻ + R).

    The share of the "surprise" we accept. If our guess is shaky (big P⁻)
    relative to the sensor (small R), K is near 1 and we trust the sensor.
    """
    return P_prior / (P_prior + R)


def update(x_prior: float, P_prior: float, z: float, R: float) -> tuple[float, float, float]:
    """Update step: blend the predicted state with measurement ``z``.

    K = P⁻ / (P⁻ + R)
    x = x⁻ + K (z − x⁻)     move toward the measurement by a fraction K
    P = (1 − K) P⁻          uncertainty shrinks by that same fraction

    Returns (x, P, K) so callers can inspect the gain.
    """
    K = kalman_gain(P_prior, R)
    innovation = z - x_prior          # how surprised we are by the measurement
    x = x_prior + K * innovation
    P = (1.0 - K) * P_prior
    return x, P, K


def run_filter(x0: float, P0: float, zs, Rs, Q: float = 0.0) -> dict[str, np.ndarray]:
    """Run predict/update over a sequence of measurements.

    Parameters
    ----------
    x0, P0 : starting guess and its variance (the prior).
    zs     : measurements, one per step. ``nan`` means "no measurement this
             step" - we predict but skip the update.
    Rs     : measurement noise variance per step (a scalar is broadcast).
    Q      : process noise added at every predict step.

    Returns a dict of arrays, one entry per step, holding the *posterior*
    estimate ``x``, variance ``P`` and gain ``K`` (K = 0 on skipped steps).
    """
    zs = np.asarray(zs, dtype=float)
    Rs = np.broadcast_to(np.asarray(Rs, dtype=float), zs.shape)

    x, P = float(x0), float(P0)
    xs, Ps, Ks = [], [], []
    for z, R in zip(zs, Rs):
        x, P = predict(x, P, Q)
        if np.isnan(z) or not np.isfinite(R):
            K = 0.0                    # nothing to learn from this step
        else:
            x, P, K = update(x, P, z, R)
        xs.append(x)
        Ps.append(P)
        Ks.append(K)
    return {"x": np.array(xs), "P": np.array(Ps), "K": np.array(Ks)}
