import json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import consensus_wheels as cw  # noqa: E402

CONS = [11, 12, 31, 34, 36, 37, 38, 39, 40, 42, 45, 48, 52, 54, 59]
DESIGN = os.path.join(os.path.dirname(__file__), "..", "data", "designs", "c15_6_4_141.json")


def test_both_designs_keep_their_guarantees_on_any_15():
    lines3, _ = cw.place(CONS, cw.W34)
    assert len(lines3) == 15 and cw.guarantee(lines3, CONS, 3)
    lines4, _ = cw.place(CONS, json.load(open(DESIGN))["lines"])
    assert len(lines4) == 141 and cw.guarantee(lines4, CONS, 4)


def test_exact_odds_of_the_4_if_4_wheel():
    lines4, _ = cw.place(CONS, json.load(open(DESIGN))["lines"])
    p, money = cw.exact_odds(lines4, CONS)
    assert abs(p["4"] - 0.031702) < 1e-5  # some line has 4+ exactly when 4+ of the 15 are drawn
    assert 0.16 < p["3"] < 0.17 and 80 < money < 85


def test_consensus_has_15_numbers_named_by_the_models():
    rng = np.random.default_rng(0)
    sets = np.argsort(rng.random((300, 59)), axis=1)[:, :6] + 1
    picks, cons, votes = cw.model_picks(sets, combined_score=rng.random(59).tolist())
    assert len(cons) == 15 and len(set(cons)) == 15 and all(1 <= n <= 59 for n in cons)
    assert all(len(v) == 15 for v in picks.values()) and set(votes) == set(cons)
