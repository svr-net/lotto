import importlib.util
import os

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("rl_study", os.path.join(os.path.dirname(__file__), "..", "scripts", "rl_study.py"))
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)


def make_df(drawn):
    n = len(drawn)
    df = pd.DataFrame(drawn[:, :6], columns=rl.DRAWN)
    df[rl.SETS] = np.sort(drawn[:, :6], 1)
    df["bonus"] = drawn[:, 6]
    df["draw_no"] = np.arange(n)
    df["round"] = 1
    df["date"] = pd.date_range("2000-01-01", periods=n, freq="3D").strftime("%Y-%m-%d")
    df["machine"] = np.where(np.arange(n) % 3, "A", "B")
    return df


def sticky(n, pool, rng, boost=4.0):
    rows = [list(rng.choice(np.arange(1, pool + 1), 7, replace=False))]
    for _ in range(n - 1):
        w = np.ones(pool)
        w[np.array(rows[-1][:6]) - 1] = boost
        rows.append(list(rng.choice(np.arange(1, pool + 1), 7, replace=False, p=w / w.sum())))
    return np.array(rows)


def test_agents_learn_a_sticky_machine():
    rng = np.random.default_rng(0)
    df = make_df(sticky(330, 59, rng))
    res = rl.run_era(df, 59, df.date.iloc[200], "sticky", [1], tickets=8, picks=True)
    a = res["agents"]
    assert a["REINFORCE policy gradient over numbers"]["z_mean"] > 3
    assert a["linear Thompson sampling over numbers (contextual)"]["z_mean"] > 3
    hedge = res["shares"]["Hedge over strategies (sees every strategy's result)"]
    assert hedge["repeat last draw"] + hedge["hot top-6 (last 50)"] + hedge["hot-weighted (last 100)"] > 0.5
    assert all(len(set(t)) == 6 and max(t) <= 59 for lines in res["picks"].values() for t in lines)


def test_agents_stay_at_chance_on_a_fair_machine():
    rng = np.random.default_rng(1)
    drawn = np.argsort(rng.random((330, 59)), axis=1)[:, :7] + 1
    res = rl.run_era(make_df(drawn), 59, make_df(drawn).date.iloc[200], "fair", [1], tickets=8, picks=False)
    assert all(abs(r["z_mean"]) < 3.5 for r in res["agents"].values())
