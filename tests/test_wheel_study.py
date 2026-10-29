import os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import wheel_study as ws  # noqa: E402


def fake_era(n=300, seed=0):
    rng = np.random.default_rng(seed)
    sets = np.sort(np.argsort(rng.random((n, 59)), axis=1)[:, :6] + 1, axis=1)
    dates = pd.date_range("2015-10-10", periods=(n + 1) // 2, freq="7D").strftime("%Y-%m-%d")
    rows = [{"draw_no": 2066 + i // 2, "date": dates[i // 2], "round": i % 2 + 1, **{f"n{k + 1}": int(s[k]) for k in range(6)}}
            for i, s in enumerate(sets)]
    return pd.DataFrame(rows)


def test_every_rule_picks_twelve_distinct_numbers():
    e = fake_era()
    sets = e[ws.N_COLS].to_numpy(int)
    x = ws.onehot(sets)
    rng = np.random.default_rng(1)
    for name, fn in ws.RULES.items():
        c = [int(v) for v in fn(x, sets, rng)]
        assert len(set(c)) == 12 and 1 <= min(c) and max(c) <= 59, name


def test_candidate_backtest_and_layouts_run():
    res, nights, rounds = ws.candidate_backtest(fake_era(), "2016-12-31", seeds=1)
    assert set(res) == set(ws.RULES) and rounds == 2 * nights
    s = ws.layout_stats(ws.random_lines(0), n=20_000, chunk=10_000)
    assert 0 < s["p3"] < s["p2"] < 1 and s["money_mean"] > 0


def test_spread_lines_share_no_pair_and_cover_every_number():
    import itertools
    from collections import Counter
    score = [1.0 / (i + 1) for i in range(59)]
    lines = ws.br.spread_lines(score, seed=20261031)
    assert len(lines) == 15 and all(len(set(l)) == 6 and 1 <= min(l) and max(l) <= 59 for l in lines)
    pairs = [p for l in lines for p in itertools.combinations(l, 2)]
    assert len(pairs) == len(set(pairs))
    uses = Counter(v for l in lines for v in l)
    assert len(uses) == 59 and {v for v, c in uses.items() if c == 2} == set(range(1, 32))
