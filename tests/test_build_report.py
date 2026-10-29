import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "build_report", os.path.join(os.path.dirname(__file__), "..", "scripts", "build_report.py"))
br = importlib.util.module_from_spec(spec)
spec.loader.exec_module(br)


def test_combined_picks_are_five_distinct_valid_lines_led_by_the_consensus():
    sources = [
        ("A", [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 7]]),
        ("B", [[1, 2, 3, 4, 5, 8]]),
        ("C", [[1, 2, 3, 4, 50, 51], [1, 2, 3, 40, 41, 42], [1, 2, 3, 4, 5, 9]]),
    ]
    res = br.combine(sources, seed=20261007)
    lines = res["lines"]
    assert len(lines) == 5 and len({tuple(l) for l in lines}) == 5
    assert all(len(set(l)) == 6 and min(l) >= 1 and max(l) <= 59 and l == sorted(l) for l in lines)
    assert lines[0][:5] == [1, 2, 3, 4, 5]  # the numbers every investigation backs come first
    assert res["support"][0] == 3 and res["support"][58] == 0
    assert br.combine(sources, seed=20261007)["lines"] == lines  # reproducible


def test_pick_sources_skip_random_baselines_and_backtest_ensembles():
    payload = {"predictions": {"random (baseline)": [[1, 2, 3, 4, 5, 6]], "ENSEMBLE x": [[7, 8, 9, 10, 11, 12]],
                               "hot": [[13, 14, 15, 16, 17, 18]]},
               "ml": {"eras": [{"picks": {"top6": [1, 2, 3, 4, 5, 6], "sampled": [[2, 3, 4, 5, 6, 7]]}}]}}
    src = dict(br.pick_sources(payload))
    assert src["Strategy backtest"] == [[13, 14, 15, 16, 17, 18]]
    assert len(src["Machine learning"]) == 2


def test_lines_from_score_reproduces_combine():
    sources = [("A", [[1, 2, 3, 4, 5, 6]]), ("B", [[1, 2, 3, 40, 41, 42], [7, 8, 9, 10, 11, 12]])]
    res = br.combine(sources, seed=20261014)
    assert br.lines_from_score(res["score"], 20261014) == res["lines"]
    assert br.lines_from_score(res["score"], 1)[0] == res["lines"][0]  # line 1 doesn't depend on the seed


def test_combined_lines_share_no_numbers_and_cover_more():
    score = [1.0 / (i + 1) for i in range(59)]
    lines = br.lines_from_score(score, 7)
    nums = [x for l in lines for x in l]
    assert len(nums) == len(set(nums)) == 30
    assert lines[0] == [1, 2, 3, 4, 5, 6]
    spread, stacked = br.coverage(lines, n=200_000), br.coverage([lines[0]] * 5, n=200_000)
    assert spread["p2"] > stacked["p2"] + 0.2 and spread["p3"] > stacked["p3"] * 4


def test_digit_strategy_picks_vote_within_prime_radix():
    payload = {"prime_radix": {"backtest": {"picks": {"conditional probability, base 2": [[1, 2, 3, 4, 5, 6]]}}},
               "radix_strat": {"backtest": {"picks": {"hot top-6, base 5": [[7, 8, 9, 10, 11, 12]]}}}}
    src = dict(br.pick_sources(payload))
    assert src["Prime radix"] == [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]


def test_played_wheel_spreads_all_59_numbers_without_a_shared_pair():
    import itertools
    from collections import Counter
    from popularity_study import pattern_line
    score = [1.0 / (i + 1) for i in range(59)]
    w = br.wheel(score, seed=20261031)
    lines = w["lines"]
    assert w["candidates"] == list(range(1, 60))
    assert len(lines) == 15 and all(len(set(l)) == 6 and 1 <= min(l) and max(l) <= 59 for l in lines)
    pairs = [p for l in lines for p in itertools.combinations(l, 2)]
    assert len(pairs) == len(set(pairs))
    uses = Counter(v for l in lines for v in l)
    assert len(uses) == 59 and set(w["twice"]) == {v for v, c in uses.items() if c == 2} == set(range(1, 32))
    assert not any(pattern_line(l) for l in lines)
    assert br.wheel_lines(score, 20261031)[1] == lines
    assert w["coverage"]["p3"] > 0.15 and w["coverage"]["p2"] > 0.9
