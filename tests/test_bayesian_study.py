import importlib.util
import math
import os

import numpy as np

spec = importlib.util.spec_from_file_location(
    "bayesian_study", os.path.join(os.path.dirname(__file__), "..", "scripts", "bayesian_study.py"))
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)


def test_dirichlet_multinomial_matches_closed_forms():
    c = np.array([3, 0, 2])
    # a = 1: uniform Dirichlet, P(sequence) = (K-1)! prod(c_i!) / (N+K-1)!
    expect = math.log(math.factorial(2) * math.factorial(3) * math.factorial(2) / math.factorial(7))
    assert math.isclose(bs.log_dm(c, np.array([1.0]))[0], expect, rel_tol=1e-9)
    # a -> infinity: the fair machine
    assert math.isclose(bs.log_dm(c, np.array([1e9]))[0], bs.log_fair(c), rel_tol=1e-6)


def weighted_sets(rng, n, K, w):
    return np.array([np.sort(rng.choice(np.arange(1, K + 1), 6, replace=False, p=w / w.sum())) for _ in range(n)])


def test_bayes_factor_backs_fair_data_and_flags_a_biased_ball():
    rng = np.random.default_rng(0)
    fair = bs.counts_of(bs.fair_sets(rng, 1200, 59), 59)
    assert bs.log10_bf_fair(fair) > 1  # strong support for fair
    w = np.ones(59)
    w[6] = 2.0  # ball 7 twice as likely
    biased = bs.counts_of(weighted_sets(rng, 1200, 59, w), 59)
    assert bs.log10_bf_fair(biased) < -2  # decisive support for bias


def test_group_bayes_factor_detects_machines_that_differ():
    rng = np.random.default_rng(1)
    same = np.array([bs.counts_of(bs.fair_sets(rng, 300, 59), 59) for _ in range(3)])
    assert bs.log10_bf_same(same) > 0  # 300 draws per group: leans towards "same"
    groups = []
    for g in range(3):
        w = np.ones(59)
        w[g * 10:(g * 10) + 5] = 2.5  # each machine favours its own five balls
        groups.append(bs.counts_of(weighted_sets(rng, 300, 59, w), 59))
    assert bs.log10_bf_same(np.array(groups)) < -2


def test_strategies_return_valid_tickets():
    rng = np.random.default_rng(2)
    hist = bs.fair_sets(rng, 300, 59)
    machines = np.where(np.arange(300) % 2, "A", "B")
    for name, t in bs.strategies(hist, machines, "A", 59, 5, rng).items():
        assert t.shape[1] == 6 and all(len(set(r)) == 6 and min(r) >= 1 and max(r) <= 59 for r in t.tolist())
