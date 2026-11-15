import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "score_live", os.path.join(os.path.dirname(__file__), "..", "scripts", "score_live.py"))
sl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sl)

ROUNDS = [{"draw_no": 1, "round": 1, "numbers": [1, 2, 3, 4, 5, 6], "bonus": 7},
          {"draw_no": 1, "round": 2, "numbers": [10, 20, 30, 40, 50, 59], "bonus": 2}]


def test_lines_are_scored_against_every_round_with_bonus():
    s = sl.score_line([7, 1, 2, 3, 10, 20], ROUNDS)
    assert s["line"] == [1, 2, 3, 7, 10, 20]
    assert s["matches"] == [3, 2] and s["bonus"] == [True, True]


def test_entry_scores_both_wheels_and_the_summary_counts_them():
    combined = {"next_draw_date": "2026-11-07", "wheel": {"lines": [[1, 2, 3, 7, 10, 20], [11, 12, 13, 14, 15, 16]]}}
    frozen = {"wheel": "3 if 4", "numbers": list(range(1, 16)), "first_draw": "2026-11-07", "lines": [[1, 2, 3, 4, 5, 6]]}
    entry = sl.score_entry(combined, frozen, ROUNDS, "abc")
    assert entry["frozen_wheel"]["in_numbers"] == [6, 1] and entry["frozen_wheel"]["lines"][0]["matches"] == [6, 0]
    sm = sl.summarise([entry])
    assert sm["wheel59_lines"]["line_rounds"] == 4 and sm["wheel59_lines"]["counts"][:4] == [2, 0, 1, 1]
    assert sm["wheel59_lines"]["prize"] == 1 and sm["frozen_lines"]["line_rounds"] == 2
    assert abs(sum(sm["chance"].values()) - 1) < 1e-12 and abs(sm["chance_mean"] - 36 / 59) < 1e-12
    early = sl.score_entry({**combined, "next_draw_date": "2026-11-04"}, frozen, ROUNDS, "abc")
    assert "frozen_wheel" not in early
