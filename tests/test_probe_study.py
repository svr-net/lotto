"""The probe study's strategies make valid lines and its bias tests return p-values."""
import importlib.util, os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("probe_study", os.path.join(HERE, "..", "scripts", "probe_study.py"))
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)


def fake_history(n=300, seed=0):
    rng = np.random.default_rng(seed)
    sets = ps.fair_sets(n, rng)
    ball = np.array([str(1 + k % 4) for k in range(n)])
    mach = np.array(["A" if k % 3 else "B" for k in range(n)])
    return sets, ball, mach


def test_every_strategy_makes_valid_lines():
    sets, ball, mach = fake_history()
    for name, fn in {**ps.STRATEGIES, **ps.ORACLES}.items():
        c = ps.Ctx(sets, ball, mach, {})
        lines = fn(c, 7, np.random.default_rng(1), {"ball": "2", "mach": "A"})
        assert lines.shape == (7, 6), name
        for l in lines:
            assert len(set(l.tolist())) == 6 and l.min() >= 1 and l.max() <= 59, name


def test_bias_tests_give_p_values_on_fair_draws():
    sets, ball, mach = fake_history(240, 3)
    e = pd.DataFrame(sets, columns=ps.N_COLS)
    e["ball_set"], e["machine"] = ball, mach
    # The first 40 rounds two a night, the rest one a night.
    e["date"] = [f"night{k // 2:04d}" if k < 40 else f"night{k:04d}" for k in range(len(e))]
    e["round"] = [1 + k % 2 if k < 40 else 1 for k in range(len(e))]
    out = ps.bias_tests(e, 40, np.random.default_rng(0))
    assert len(out) >= 5
    for t in out:
        assert 0 < t["p"] <= 1, t
