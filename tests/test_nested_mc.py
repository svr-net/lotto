import json, os, subprocess

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
BIN = os.path.join(ROOT, "sim", "bin", "nested_mc")


@pytest.mark.skipif(not os.path.exists(BIN), reason="sim/bin/nested_mc not built (docker build --target native ... sim)")
@pytest.mark.parametrize("backend", ["cpu", "threads"])
def test_fair_model_matches_the_exact_odds(tmp_path, backend):
    out = tmp_path / "n.json"
    subprocess.run([BIN, "--csv", os.path.join(ROOT, "data", "uk_lotto_full.csv"), "--wheel", os.path.join(ROOT, "results", "exports", "frozen_wheel.csv"),
                    "--first-draw", "2026-11-07", "--start", "2026-01-01", "--paths", "300", "--forward-paths", "20000",
                    "--backend", backend, "--strategies", "none", "--out", str(out)], check=True, capture_output=True)
    r = json.load(open(out))
    assert r["backend"] == backend and len(r["frozen_numbers"]) == 15
    fair = r["forward"]["models"]["fair"]["checkpoints"]["104"]
    # the frozen wheel has a 3+ line in 11.43% of fair rounds: 11.9 of 104, and costs 30 a round
    assert 11 <= fair["prize_rounds"][1] <= 13 and -2400 < fair["net"][1] < -2100
    assert set(r["models"]) == {"fair", "hindsight as reality"}


@pytest.mark.skipif(not os.path.exists(BIN), reason="sim/bin/nested_mc not built (docker build --target native ... sim)")
def test_lock_free_threads_match_the_single_core_kernel(tmp_path):
    """Paths are seeded by index and added into histograms, so the pool's scheduling can't change any number."""
    outs = {}
    for backend in ("cpu", "threads"):
        out = tmp_path / f"{backend}.json"
        subprocess.run([BIN, "--csv", os.path.join(ROOT, "data", "uk_lotto_full.csv"), "--wheel", os.path.join(ROOT, "results", "exports", "frozen_wheel.csv"),
                        "--first-draw", "2026-11-07", "--start", "2025-06-01", "--paths", "500", "--forward-paths", "5000",
                        "--backend", backend, "--strategies", "none", "--out", str(out)], check=True, capture_output=True)
        outs[backend] = json.load(open(out))
        outs[backend].pop("backend")
    assert outs["cpu"] == outs["threads"]


@pytest.mark.skipif(not os.path.exists(BIN), reason="sim/bin/nested_mc not built (docker build --target native ... sim)")
def test_strategies_reset_to_the_real_draws(tmp_path):
    """A strategy with no information scores exactly like fair; one backing the real draws in hindsight beats it."""
    import numpy as np, pandas as pd
    df = pd.read_csv(os.path.join(ROOT, "data", "uk_lotto_full.csv"))
    df = df[df.draw_no >= 2066]
    first = df.drop_duplicates("date")
    nights = first[first.date >= "2025-01-01"]
    rows = []
    for _, r in nights.iterrows():
        cheat = np.full(59, 0.1)
        cheat[r[["n1", "n2", "n3", "n4", "n5", "n6"]].to_numpy(int) - 1] = 1.0  # knows the night's draw: an upper bound
        rows.append([r.date, "uniform", *np.full(59, 1 / 59)])
        rows.append([r.date, "oracle", *(cheat / cheat.sum())])
    rows += [["2099-01-01", name, *np.full(59, 1 / 59)] for name in ("uniform", "oracle")]
    csv = tmp_path / "w.csv"
    pd.DataFrame(rows, columns=["date", "strategy", *[f"w{i}" for i in range(1, 60)]]).to_csv(csv, index=False)
    out = tmp_path / "n.json"
    subprocess.run([BIN, "--csv", os.path.join(ROOT, "data", "uk_lotto_full.csv"), "--wheel", os.path.join(ROOT, "results", "exports", "frozen_wheel.csv"),
                    "--start", "2025-01-01", "--paths", "200", "--forward-paths", "1000", "--strategies", str(csv), "--out", str(out)],
                   check=True, capture_output=True)
    s = json.load(open(out))["strategies"]
    assert s["nights"] == len(nights) and s["next_night"] == "2099-01-01"
    m = s["models"]
    assert set(m) == {"uniform", "oracle", "best so far (walk-forward)", "Bayesian mixture (walk-forward)"}
    assert abs(m["uniform"]["log_score_diff"]) < 1e-9 and m["uniform"]["lambda_final"] == 1
    assert m["oracle"]["log_score_diff"] > 5 and m["oracle"]["matches6"] == 6 and m["oracle"]["lambda_final"] < 0.1
    # the walk-forward choices find the oracle from the real draws alone
    assert m["Bayesian mixture (walk-forward)"]["log_score_diff"] > 4 and s["mixture_weights"]["oracle"] > 0.99
    assert s["best_so_far_chosen"]["oracle"] >= s["nights"] - 2  # from the third night: the first two have nothing to go on
