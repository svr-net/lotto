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


def test_summary_counts_line_rounds_and_prizes():
    entry = sl.score_entry({"next_draw_date": "2026-10-10", "lines": [[1, 2, 3, 7, 10, 20], [11, 12, 13, 14, 15, 16]]},
                           {"predictions": {"hot": [[4, 5, 6, 40, 50, 59]]}}, ROUNDS, "abc")
    sm = sl.summarise([entry])
    assert sm["combined"]["line_rounds"] == 4 and sm["combined"]["counts"][:4] == [2, 0, 1, 1]
    assert sm["combined"]["prize"] == 1 and sm["combined"]["bonus"] == 2
    assert sm["strategies"]["counts"][3] == 2 and sm["strategies"]["prize"] == 2
    assert abs(sum(sm["chance"].values()) - 1) < 1e-12 and abs(sm["chance_mean"] - 36 / 59) < 1e-12


def test_committed_59_number_wheel_is_kept_as_is(tmp_path, monkeypatch):
    rounds = [{"draw_no": 1, "round": 1, "numbers": [1, 2, 3, 4, 5, 6], "bonus": 7}]
    lines = [[1 + (6 * i + k) % 59 for k in range(6)] for i in range(15)]  # 90 places over all 59 numbers
    entry = {"date": "2026-11-04", "draw_no": 1, "rounds": rounds, "combined": [], "strategies": {},
             "wheel_lines": [sl.score_line(l, rounds) for l in lines]}
    monkeypatch.setattr(sl, "REPLAY", str(tmp_path / "none.json"))
    record = {"draws": [entry]}
    sl.backfill_wheel59(record, None)
    assert entry["wheel59"]["source"] == "committed" and entry["wheel59"]["lines"] == entry["wheel_lines"]
    assert sl.summarise(record["draws"])["wheel59_lines"]["line_rounds"] == 15
