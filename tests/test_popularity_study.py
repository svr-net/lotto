import importlib.util
import os

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "popularity_study", os.path.join(os.path.dirname(__file__), "..", "scripts", "popularity_study.py"))
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)


def fake_history(n=1500, birthday_lift=1.3, seed=0):
    """Rounds where each number 1-31 multiplies the expected winners by birthday_lift."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        line = np.sort(rng.choice(np.arange(1, 60), 6, replace=False))
        mu = 0.3 * birthday_lift ** ((line <= 31).sum() - 3)
        wins = rng.poisson(mu)
        rows.append({"draw_no": 2066 + i, "date": f"{2016 + i // 104}-01-01", "round": 1,
                     **{f"n{j + 1}": int(line[j]) for j in range(6)}, "bonus": 0,
                     "jackpot": 2_000_000 / max(wins, 1), "wins": wins})
    return pd.DataFrame(rows)


def test_fit_recovers_a_planted_birthday_effect_and_ignores_per_winner_jackpots():
    res = ps.fit(fake_history())
    f = {x["name"]: x for x in res["features"]}
    assert abs(f["birthday numbers (1-31)"]["multiplier"] - 1.3) < 0.1
    assert f["birthday numbers (1-31)"]["z"] > 5 and res["p"] < 1e-6


def test_no_effect_means_no_signal():
    res = ps.fit(fake_history(birthday_lift=1.0, seed=1))
    assert res["p"] > 0.001


def test_popularity_aware_lines_avoid_popular_numbers_and_patterns():
    pop = ps.Popularity([np.log(1.2), 0, np.log(1.5), 0, 0, 0], lam=0.2)
    assert pop.popularity([3, 7, 12, 19, 25, 31]) > 1 > pop.popularity([33, 38, 44, 49, 53, 58])
    assert pop.value([33, 38, 44, 49, 53, 58]) > 1 > pop.value([3, 7, 12, 19, 25, 31])
    lines = ps.choose_lines(np.ones(59), pop, seed=3)
    assert len(lines) == 5 and all(len(set(l)) == 6 for l in lines)
    assert all(not ps.pattern_line(l) for l in lines)
    assert all(len(set(a) & set(b)) <= 2 for i, a in enumerate(lines) for b in lines[i + 1:])
    assert np.mean([sum(x <= 31 for x in l) for l in lines]) < 2
    assert ps.pattern_line([7, 14, 21, 28, 35, 42]) and ps.pattern_line([4, 5, 6, 20, 30, 40])
