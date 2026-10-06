import importlib.util
import math
import os

import numpy as np

spec = importlib.util.spec_from_file_location(
    "prime_radix_study", os.path.join(os.path.dirname(__file__), "..", "scripts", "prime_radix_study.py"))
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)


def fair(n, rng):
    return np.argsort(rng.random((n, 59)), axis=1)[:, :6] + 1


def test_digits_round_trip_in_every_prime_base():
    x = np.arange(1, 60)
    for p in pr.FIRST_10_PRIMES:
        d = pr.digits(x, p)
        assert (d < p).all()
        assert ((d * p ** np.arange(d.shape[1])).sum(1) == x).all()
    assert pr.digits(np.array([59]), 7).tolist() == [[3, 1, 1]]  # least significant first


def test_norm_ppf_inverts_norm_cdf():
    z = np.linspace(-4, 4, 41)
    assert np.allclose(pr.norm_ppf(pr.norm_cdf(z)), z, atol=1e-6)
    assert abs(pr.norm_ppf(np.array([0.975]))[0] - 1.959964) < 1e-5


def test_strategies_return_valid_tickets():
    rng = np.random.default_rng(0)
    hist = np.sort(fair(300, rng), axis=1)
    for p in (2, 7, 29):
        for fn in (pr.cond_prob_tickets, pr.copula_tickets):
            t = fn(hist, p, 10, rng)
            assert t.shape == (10, 6)
            assert all(len(set(r)) == 6 and min(r) >= 1 and max(r) <= 59 for r in t.tolist())


def test_structure_tests_catch_a_planted_residue_chain():
    # Rig the machine: whenever possible, the next ball drawn has the same residue mod 5
    # as the one before it. Fair draws should not trigger the test.
    rng = np.random.default_rng(3)
    drawn = []
    for _ in range(800):
        left = list(range(1, 60))
        first = int(rng.choice(left))
        row = [first]
        left.remove(first)
        while len(row) < 6:
            same = [x for x in left if x % 5 == row[-1] % 5]
            nxt = int(rng.choice(same if same and rng.random() < 0.6 else left))
            row.append(nxt)
            left.remove(nxt)
        drawn.append(row)
    drawn = np.array(drawn)
    obs = pr.stats_for(drawn, 5)["within_cond"]
    null = [pr.stats_for(fair(800, rng), 5)["within_cond"] for _ in range(20)]
    assert obs > max(null) * 2
