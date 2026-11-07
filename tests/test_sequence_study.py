import os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import sequence_study as sq  # noqa: E402


def fair(n, pool, seed=0):
    return np.argsort(np.random.default_rng(seed).random((n, pool)), axis=1)[:, :6] + 1


def test_hit_chance_for_15_numbers():
    assert abs(sq.p_hit(59) - 0.031702) < 1e-5


def test_gaps_count_rounds_since_last_drawn():
    x = sq.onehot(np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [1, 8, 13, 14, 15, 16]]), 59)
    g = sq.gaps_before(x)
    assert g[1, 0] == 0 and g[2, 0] == 1 and g[2, 7] == 0 and g[0, 0] == 0 and g[2, 58] == 2


def test_memory_finds_nothing_in_fair_draws_and_a_planted_echo():
    rng = np.random.default_rng(1)
    m = sq.memory(fair(400, 59), 59, 100, rng)
    assert m["hazard"]["p"] > 0.01 and m["lag1"]["p"] > 0.01
    # numbers never repeat in the next round: the lag-1 test must flag it
    h = []
    prev = set()
    r = np.random.default_rng(2)
    for _ in range(400):
        pool = [n for n in range(1, 60) if n not in prev]
        d = sorted(r.choice(pool, 6, replace=False))
        h.append(d)
        prev = set(d)
    m = sq.memory(np.array(h), 59, 100, rng)
    assert m["lag1"]["observed"] == 0 and m["lag1"]["p"] < 0.05


def test_rules_pick_15_and_timing_and_waits_run():
    sets = fair(300, 59, 3)
    x = sq.onehot(sets, 59)
    g = sq.gaps_before(x)
    rng = np.random.default_rng(0)
    for fn in sq.RULES.values():
        assert len(set(np.asarray(fn(x[:250], g[250], rng, 59)).tolist())) == 15
    seq = (np.random.default_rng(4).random(2000) < 0.0317).astype(int)
    t = sq.timing(seq)
    assert t["otherwise"][1] > 0
    o = sq.over_time(seq, 0.0317)
    assert 20 < o["waits"]["mean"] < 45 and 0 <= o["waits"]["p"] <= 1
