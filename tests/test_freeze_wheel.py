import importlib.util, json, os, sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import freeze_wheel as fw  # noqa: E402

spec = importlib.util.spec_from_file_location("score_live", os.path.join(HERE, "..", "scripts", "score_live.py"))
sl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sl)


def test_frozen_wheel_is_15_lines_with_its_guarantee():
    fz = json.load(open(os.path.join(HERE, "..", "results", "frozen_wheel.json")))
    nums = set(fz["numbers"])
    assert len(fz["lines"]) == 15 and len(nums) == 15 and all(set(l) <= nums and len(set(l)) == 6 for l in fz["lines"])
    import itertools
    assert all(any(len(set(l) & set(q)) >= 3 for l in fz["lines"]) for q in itertools.combinations(sorted(nums), 4))


def test_exports_list_every_line(tmp_path):
    fz = {"wheel": "3 if 4", "numbers": list(range(1, 16)), "price": 2.0, "first_draw": "2026-11-07",
          "lines": [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]}
    fw.write_exports(fz, str(tmp_path))
    txt = (tmp_path / "frozen_wheel_app.txt").read_text()
    assert "Line  1:   1   2   3   4   5   6" in txt and "Line  2:" in txt
    assert (tmp_path / "frozen_wheel.csv").read_text().splitlines()[2] == "2,7,8,9,10,11,12"


def test_frozen_lines_are_scored_from_the_first_draw(tmp_path, monkeypatch):
    fz = {"wheel": "3 if 4", "numbers": list(range(1, 16)), "first_draw": "2026-11-07",
          "lines": [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]}
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(fz))
    monkeypatch.setattr(sl, "FROZEN", str(path))
    rounds = [{"draw_no": 1, "round": 1, "numbers": [1, 2, 3, 20, 30, 40], "bonus": 7}]
    record = {"draws": [{"date": "2026-11-04", "rounds": rounds}, {"date": "2026-11-07", "rounds": rounds}]}
    sl.score_frozen(record)
    assert "frozen_wheel" not in record["draws"][0]
    f = record["draws"][1]["frozen_wheel"]
    assert f["in_numbers"] == [3] and [s["matches"] for s in f["lines"]] == [[3], [0]]
