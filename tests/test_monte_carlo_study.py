import os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import monte_carlo_study as mc  # noqa: E402


def test_line_probabilities_add_up():
    p = mc.line_probs()
    assert abs(sum(p[k] for k in range(7)) - 1) < 1e-12
    assert abs(p["5"] + p["5b"] - p[5]) < 1e-15 and abs(1 / p[6] - 45_057_474) < 1


def test_wheel_season_costs_and_returns():
    rng = np.random.default_rng(0)
    lines = [[1 + (6 * i + k) % 59 for k in range(6)] for i in range(15)]
    s = mc.wheel_season(lines, 2.0, 5_000_000, 50_000, 2_000, rng)
    r = s["per_round"]
    assert r["cost"] == 30 and 7 < r["fixed_mean"] < 11 and 0 < r["return_pct"] < 100
    y = s["seasons"]["1 year"]
    assert y["cost"] == 104 * 30 and y["net_p5"] <= y["net_median"] <= y["net_p95"] < 0


def test_biased_history_favours_the_weighted_numbers():
    rng = np.random.default_rng(1)
    w = np.ones(59)
    w[0] = 3.0
    h = mc.biased_history(5000, w, rng)
    assert h.shape == (5000, 6) and all(len(set(r)) == 6 for r in h[:200])
    counts = np.bincount(h.ravel(), minlength=60)[1:]
    assert counts[0] > 2 * counts[1:].mean()


def test_power_is_low_without_bias_and_high_with_a_large_one():
    rng = np.random.default_rng(2)
    out = mc.power(600, 200, 2.9, 40, [0, 1.0], rng)
    six = out["six numbers"]["rows"]
    assert six[0]["chi_square"] <= 0.2 and six[1]["chi_square"] >= 0.9
