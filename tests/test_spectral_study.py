import importlib.util
import math
import os

import numpy as np

spec = importlib.util.spec_from_file_location(
    "spectral_study", os.path.join(os.path.dirname(__file__), "..", "scripts", "spectral_study.py"))
ss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ss)


def fair(n, pool, rng):
    return ss.fair_history(rng, n, pool)


def test_haar_round_trip_and_energy():
    rng = np.random.default_rng(0)
    x = rng.random((3, 256))
    c = ss.haar_dwt(x, 8)
    assert np.allclose(ss.haar_idwt(c), x)
    assert np.allclose(sum((ci ** 2).sum(1) for ci in c), (x ** 2).sum(1))  # orthonormal


def test_fourier_forecast_extends_a_sinusoid_whose_period_does_not_divide_the_window():
    t = np.arange(257)
    x = 0.5 + 0.3 * np.sin(2 * np.pi * t / 9.7 + 2.5)
    pred = ss.fourier_forecast(x[None, :256], k=1)[0]
    assert abs(pred - x[256]) < 0.02
    assert abs(pred - x[0]) > 0.05  # not just the window wrapped round


def test_morlet_finds_the_period_of_a_burst():
    t = np.arange(1024)
    x = np.random.default_rng(1).normal(size=1024)
    x[400:700] += 3 * np.sin(2 * np.pi * t[400:700] / 20)
    scales = np.geomspace(2, 256, 40)
    P = ss.morlet_power(x, scales)
    i, j = np.unravel_index(P.argmax(), P.shape)
    assert 16 < scales[i] * 1.033 < 25 and 400 < j < 700


def test_planted_periodic_number_is_detected():
    # Rig the machine: ball 7 is forced into every 8th round.
    rng = np.random.default_rng(2)
    sets, drawn, bonus = fair(1024, 59, rng)
    for r in range(0, 1024, 8):
        if 7 not in sets[r]:
            drawn[r, 0] = 7
            sets[r] = np.sort(drawn[r])
    assert all(len(set(row)) == 6 for row in sets.tolist())
    scales = np.geomspace(2, 256, 24)
    obs = ss.history_stats(sets, drawn, bonus, 59, 7, scales)
    null = [ss.history_stats(*fair(1024, 59, rng), 59, 7, scales)["number_period_g"] for _ in range(20)]
    assert obs["number_period_which"] == 7 and obs["number_period_g"] > max(null)


def test_strategies_return_valid_tickets():
    rng = np.random.default_rng(3)
    hist = fair(300, 59, rng)[0]
    for name, fn in ss.STRATEGIES.items():
        t = fn(hist, 5, rng)
        assert t.shape == (5, 6) and all(len(set(r)) == 6 and min(r) >= 1 and max(r) <= 59 for r in t.tolist())
