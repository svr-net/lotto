import json, os, subprocess

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
BIN = os.path.join(ROOT, "sim", "bin", "assimilate")
CSV = os.path.join(ROOT, "data", "uk_lotto_full.csv")
needs_bin = pytest.mark.skipif(not os.path.exists(BIN), reason="sim/bin/assimilate not built (docker build --target native ... sim)")


def run(tmp_path, *args):
    out = tmp_path / "a.json"
    subprocess.run([BIN, "--csv", CSV, "--out", str(out), "--particles", "256", "--samples", "512", *args], check=True, capture_output=True)
    return json.load(open(out))


@needs_bin
def test_lock_free_threads_match_one_core(tmp_path):
    a = run(tmp_path, "--start", "2026-01-01", "--validate", "", "--backend", "cpu")
    b = run(tmp_path, "--start", "2026-01-01", "--validate", "", "--backend", "threads")
    a.pop("backend"), b.pop("backend")
    assert a == b


@needs_bin
def test_finds_a_planted_bias_and_not_a_fair_one(tmp_path):
    r = run(tmp_path, "--real", "0", "--validate", "0,1")
    fair, biased = r["validation"]
    assert fair["models"]["mixture"]["log_lik_ratio"] < 3 and fair["filter"]["tau"][1] < 0.05
    m = biased["models"]["mixture"]
    assert m["log_lik_ratio"] > 50 and m["p_better"] < 0.01 and m["p4"] < 0.01
    assert biased["filter"]["tau"][1] > 0.08 and biased["mixture_weights"]["fair"] < 0.01
    assert abs(r["chance"]["rate4"] - 0.031702) < 1e-5 and abs(r["chance"]["rate3"] - 0.1143) < 1e-3
