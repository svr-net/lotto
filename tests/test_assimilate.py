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
    a = run(tmp_path, "--start", "2026-01-01", "--validate", "", "--nulls", "8", "--backend", "cpu", "--exports", "none")
    b = run(tmp_path, "--start", "2026-01-01", "--validate", "", "--nulls", "8", "--backend", "threads", "--exports", "none")
    a.pop("backend"), b.pop("backend")
    assert a == b


@needs_bin
def test_finds_a_planted_bias_and_not_a_fair_one(tmp_path):
    r = run(tmp_path, "--real", "0", "--validate", "0,1", "--nulls", "0")
    fair, biased = r["validation"]
    assert fair["models"]["mixture"]["log_lik_ratio"] < 3 and fair["filter"]["tau"][1] < 0.05
    m = biased["models"]["mixture"]
    assert m["log_lik_ratio"] > 50 and m["p_better"] < 0.01 and m["p4"] < 0.01
    assert biased["filter"]["tau"][1] > 0.08 and biased["mixture_weights"]["fair"] < 0.01
    assert abs(r["chance"]["rate4"] - 0.031702) < 1e-5 and abs(r["chance"]["rate3"] - 0.1143) < 1e-3


@needs_bin
def test_one_fifteen_number_wheel(tmp_path):
    exports = tmp_path / "exports"
    exports.mkdir()
    r = run(tmp_path, "--start", "2026-06-01", "--validate", "", "--nulls", "0", "--exports", str(exports))
    nxt = r["real"]["next"]
    assert len(nxt["numbers"]) == 15 and len(nxt["lines"]) == 15 and abs(sum(nxt["weights"]) - 1) < 1e-4
    assert all(len(l) == 6 and set(l) <= set(nxt["numbers"]) for l in nxt["lines"])
    assert set(r["real"]["models"]) == {"fair", "particle filter", "filter + neural net", "radix digits", "reverse-engineered machine", "mixture"}
    assert abs(sum(r["real"]["machine_aware"]["hypotheses"].values()) - 1) < 1e-4 and len(r["real"]["networks"]) == 18
    assert abs(sum(r["real"]["mixture_weights"].values()) - 1) < 1e-4 and abs(sum(r["real"]["radix"]["bases"].values()) - 1) < 1e-4
    csv = (exports / "wheel.csv").read_text().splitlines()
    assert csv[0] == "line,n1,n2,n3,n4,n5,n6" and [list(map(int, l.split(",")[1:])) for l in csv[1:]] == nxt["lines"]
    # any 4 of the 15 share 3 with some line: the "3 if 4" guarantee
    from itertools import combinations
    assert all(any(len(set(q) & set(l)) >= 3 for l in nxt["lines"]) for q in combinations(nxt["numbers"], 4))


@needs_bin
def test_reverse_engineering_finds_a_planted_machine_bias_and_ranks_the_real_draws(tmp_path):
    r = run(tmp_path, "--real", "0", "--validate", "m1.5", "--nulls", "0")
    v = r["validation"][0]
    assert v["machine"] and v["mixture_weights"]["reverse-engineered machine"] > 0.5
    assert v["models"]["reverse-engineered machine"]["log_lik_ratio"] > 2
    rr = run(tmp_path, "--start", "2026-06-01", "--validate", "", "--nulls", "20", "--exports", "none")["reverse"]
    assert rr["nulls"] == 20 and len(rr["hypotheses"]) == 7 and all(0 < h["p"] <= 1 for h in rr["hypotheses"].values())
