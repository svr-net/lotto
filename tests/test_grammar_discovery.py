import importlib.util
import itertools
import math
import os

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "grammar_discovery", os.path.join(os.path.dirname(__file__), "..", "scripts", "grammar_discovery.py"))
gd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gd)

POOL = 8
rng = np.random.default_rng(0)
TRAIN = np.argsort(rng.random((40, POOL)), axis=1)[:, :6] + 1
ALL_SEQS = np.array(list(itertools.permutations(range(1, POOL + 1), 6)))
ALL_SETS = np.array(list(itertools.combinations(range(1, POOL + 1), 6)))


def ctx(n):
    return {"prev_sets": np.tile([1, 2, 3, 4, 5, 6], (n, 1)), "machine": np.array(["A"] * n),
            "train_machine": np.array(["A"] * 20 + ["B"] * 20)}


@pytest.mark.parametrize("model", gd.ORDER_MODELS + [gd.Uniform])
def test_ordered_models_are_proper_distributions(model):
    setting = 5 if model in (gd.TransformerOrder,) else 0.3 if model is gd.Echo else 2
    m = model().fit(TRAIN, POOL, setting, ctx(40))
    total = np.sum(2.0 ** -m.bits(ALL_SEQS, ctx(len(ALL_SEQS))))
    assert math.isclose(total, 1.0, rel_tol=1e-6)


@pytest.mark.parametrize("model", gd.SET_MODELS + [gd.SortedUniform])
def test_set_models_are_proper_distributions(model):
    setting = 5 if model is gd.TransformerSorted else 2
    m = model().fit(np.sort(TRAIN, axis=1), POOL, setting)
    total = np.sum(2.0 ** -m.bits(ALL_SETS, ctx(len(ALL_SETS))))
    assert math.isclose(total, 1.0, rel_tol=1e-6)


def test_uniform_bits_match_the_fair_limit():
    m = gd.Uniform().fit(TRAIN, 59)
    row = np.array([[1, 2, 3, 4, 5, 6]])
    assert math.isclose(m.bits(row, None)[0], math.log2(math.perm(59, 6)))


def test_rules_and_lags_on_a_planted_pattern():
    r = np.random.default_rng(1)
    drawn = np.argsort(r.random((600, 59)), axis=1)[:, :6] + 1
    # Plant the rule "7 then 11" in every third draw that doesn't already hold 7 or 11.
    rows = [i for i in range(0, 600, 3) if not np.isin([7, 11], drawn[i]).any()]
    drawn[rows, :2] = [7, 11]
    assert all(len(set(row)) == 6 for row in drawn)
    res = gd.rules_test(drawn, 59, sims=30, rng=r)
    assert res["digram"]["top"][0]["rule"] == "7 then 11"
    assert res["digram"]["p"] < 0.05 and res["pair"]["p"] < 0.05


def test_heavily_smoothed_sorted_grammar_is_the_fair_machine():
    m = gd.SortedGrammar().fit(np.sort(TRAIN, axis=1), POOL, 1e9)
    bits = m.bits(ALL_SETS, ctx(len(ALL_SETS)))
    assert np.allclose(bits, math.log2(math.comb(POOL, 6)), atol=1e-6)
