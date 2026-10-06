import importlib.util
import os

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "ml_ensemble_study", os.path.join(os.path.dirname(__file__), "..", "scripts", "ml_ensemble_study.py"))
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)


def make_df(drawn, rng):
    n = len(drawn)
    df = pd.DataFrame(drawn[:, :6], columns=ml.DRAWN)
    df[ml.SETS] = np.sort(drawn[:, :6], 1)
    df["bonus"] = drawn[:, 6]
    df["draw_no"] = np.arange(n)
    df["round"] = 1
    df["date"] = pd.date_range("2000-01-01", periods=n, freq="3D").strftime("%Y-%m-%d")
    df["machine"] = np.where(np.arange(n) % 3, "A", "B")
    return df


def fair(n, pool, rng):
    return np.argsort(rng.random((n, pool)), axis=1)[:, :7] + 1


def sticky(n, pool, rng, boost=4.0):
    """Rigged machine: last round's numbers are `boost` times as likely to come out again."""
    rows = [list(rng.choice(np.arange(1, pool + 1), 7, replace=False))]
    for _ in range(n - 1):
        w = np.ones(pool)
        w[np.array(rows[-1][:6]) - 1] = boost
        rows.append(list(rng.choice(np.arange(1, pool + 1), 7, replace=False, p=w / w.sum())))
    return np.array(rows)


def test_features_never_see_the_round_they_predict():
    rng = np.random.default_rng(0)
    drawn = fair(200, 59, rng)
    X1, _ = ml.build_features(make_df(drawn, rng), 59)
    drawn2 = drawn.copy()
    drawn2[150] = fair(1, 59, rng)[0]
    X2, _ = ml.build_features(make_df(drawn2, rng), 59)
    assert np.array_equal(X1[:151], X2[:151])
    assert not np.array_equal(X1[151], X2[151])


def test_per_round_auc():
    p = np.array([[0.9, 0.8, 0.1, 0.2], [0.1, 0.2, 0.9, 0.8]])
    y = np.array([[1, 1, 0, 0], [1, 1, 0, 0]])
    assert np.allclose(ml.per_round_auc(p, y), [1.0, 0.0])


def test_models_detect_a_sticky_machine_and_not_a_fair_one():
    rng = np.random.default_rng(1)
    df = make_df(sticky(500, 59, rng), rng)
    split = df.date.iloc[350]
    res = ml.era_study(df, 59, split, "sticky", [1], block=50, tickets=3)
    gb = res["models"]["gradient boosting"]
    assert gb["auc"] > 0.56 and gb["auc_z"] > 4 and gb["excess_bits"] < 0
    assert res["importance"][0]["feature"] in ("in_last", "in_last_2", "gap", "freq_10", "gap_ratio")
    df = make_df(fair(500, 59, rng), rng)
    res = ml.era_study(df, 59, split, "fair", [1], block=50, tickets=3)
    gb = res["models"]["gradient boosting"]
    assert abs(gb["auc_z"]) < 3.5 and gb["excess_bits"] > -0.05
