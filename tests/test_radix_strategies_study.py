import importlib.util
import os

import numpy as np

spec = importlib.util.spec_from_file_location(
    "radix_strategies_study", os.path.join(os.path.dirname(__file__), "..", "scripts", "radix_strategies_study.py"))
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)


def test_fair_counts_give_no_lift_in_any_base():
    for b in rs.BASES:
        dc = rs.DigitClasses(b)
        assert dc.M.shape[0] == 59 and np.allclose(dc.M.sum(1), dc.w)  # one digit per position
        lift = dc.lift(np.full(59, 10000 * 6 / 59), 10000 * 6)  # every number drawn exactly as often as fair
        assert np.allclose(lift, 1, atol=1e-6)


def test_tickets_are_valid_for_every_strategy_and_base():
    import pandas as pd
    rng = np.random.default_rng(0)
    T = 300
    sets = np.array([np.sort(rng.choice(np.arange(1, 60), 6, replace=False)) for _ in range(T)])
    drawn = np.array([rng.permutation(s) for s in sets])
    df = pd.DataFrame({**{f"n{i + 1}": sets[:, i] for i in range(6)}, **{f"d{i + 1}": drawn[:, i] for i in range(6)},
                       "bonus": [int(rng.choice(np.setdiff1d(np.arange(1, 60), s))) for s in sets],
                       "machine": rng.choice(["A", "B"], T), "round": np.ones(T, int)})
    h = rs.History(df)
    for b in (2, 5, rs.NUMBERS):
        dc = rs.DigitClasses(b)
        F = rs.digit_features(h, dc)
        ctx = {"cols": rs.position_columns(dc), "F": F, "Fn": lambda t: F[t]}
        for s in rs.STRATEGIES[:-1]:
            tk = rs.make_tickets(h, dc, s, 280, 3, rng, 1, ctx)
            assert all(len(set(t)) == 6 and t.min() >= 1 and t.max() <= 59 for t in tk), s


def test_a_planted_digit_bias_is_found_in_its_own_base():
    res = rs.planted()["hot_weighted_z"]
    assert res["base 5"] > 2.5 and res["base 5"] > res["base 7"]
