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
                    "--backend", backend, "--out", str(out)], check=True, capture_output=True)
    r = json.load(open(out))
    assert r["backend"] == backend and len(r["frozen_numbers"]) == 15
    fair = r["forward"]["models"]["fair"]["checkpoints"]["104"]
    # the frozen wheel has a 3+ line in 11.43% of fair rounds: 11.9 of 104, and costs 30 a round
    assert 11 <= fair["prize_rounds"][1] <= 13 and -2400 < fair["net"][1] < -2100
    assert set(r["models"]) == {"fair", "hindsight as reality"}
