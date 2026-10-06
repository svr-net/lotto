import importlib.util
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FEED = """<PRE>
No., Day,DD,MMM,YYYY, N1,N2,N3,N4,N5,N6,BN,   Jackpot, Wins,   Machine  ,Set
3212, Sat, 3,Oct,2026, 37,42,53,39,46,15,33,   2000000,    0,   Lotto 6  ,  9 
3212, Sat, 3,Oct,2026, 11,44,54,12,52,05,01,   2000000,    0,   Lotto 4  , 10 
3211, Wed,30,Sep,2026, 48,18,49,51,53,26,33,   2000000,    1,   Lotto 6  ,  9 
</PRE>"""


def test_feed_parser_keeps_drawn_order_and_sorted_numbers():
    rows = _load("fetch_merseyworld").parse(FEED)
    assert [(r["draw_no"], r["round"]) for r in rows] == [(3211, 1), (3212, 1), (3212, 2)]
    r = rows[1]
    assert r["drawn"] == [37, 42, 53, 39, 46, 15]
    assert r["main"] == [15, 37, 39, 42, 46, 53]
    assert r["bonus"] == 33 and r["machine"] == "Lotto 6"


def test_order_markov_counts_drawn_transitions():
    wf = _load("walk_forward_backtest")
    drawn = np.array([[5, 1, 9, 2, 7, 3], [5, 1, 4, 6, 8, 10]])
    start, trans = wf.order_markov(drawn)
    assert start[5 - 1] == 3  # two draws started with 5, plus add-1 smoothing
    assert trans[5, 1 - 1] == 3  # 5 -> 1 seen twice
    assert trans[1, 9 - 1] == 2 and trans[1, 4 - 1] == 2


def test_order_stats_on_fair_draws_match_known_values():
    wf = _load("walk_forward_backtest")
    rng = np.random.default_rng(0)
    drawn = np.argsort(rng.random((20000, 59)), axis=1)[:, :6] + 1
    s = wf.order_stats(drawn)
    assert abs(s["serial_corr"] - (-1 / 58)) < 0.01  # sampling without replacement
    assert abs(s["rising_share"] - 0.5) < 0.01
    assert abs(s["first_ball_carry"] - 6 / 59) < 0.01


def test_draw_order_tests_flag_a_biased_first_ball():
    wf = _load("walk_forward_backtest")
    rng = np.random.default_rng(1)
    drawn = np.argsort(rng.random((1200, 59)), axis=1)[:, :7] + 1
    # Rig the machine: the first ball out is always 1-30. Later balls that would
    # duplicate it are swapped for the seventh ball.
    drawn[:, 0] = np.where(drawn[:, 0] > 30, (drawn[:, 0] % 30) + 1, drawn[:, 0])
    dup = drawn[:, 1:6] == drawn[:, :1]
    drawn[:, 1:6] = np.where(dup, drawn[:, 6:7], drawn[:, 1:6])
    df = pd.DataFrame(drawn[:, :6], columns=wf.DRAWN)
    df[wf.COLS] = np.sort(drawn[:, :6], axis=1)
    df["bonus"] = drawn[:, 6]
    df["machine"] = np.where(np.arange(1200) % 2, "A", "B")
    df["ball_set"] = np.where(np.arange(1200) % 3, "1", "2")
    res = wf.draw_order_tests(df, sims=50, seed=0)
    first = next(t for t in res["tests"] if t["test"].startswith("ball 1 out"))
    assert first["p"] < res["bonferroni_alpha"]
