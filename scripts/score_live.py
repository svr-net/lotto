"""Score the wheels committed before each draw against the real result and keep a running record in
results/live_scores.json (shown in the report by scripts/build_report.py).

For each git ref given (default origin/main) it reads results/combined_picks.json as it was at that ref,
finds the draw on its next_draw_date in data/uk_lotto_full.csv and records, for every line of the
59-number wheel ("wheel59") and of the frozen wheel (results/frozen_wheel.json, from its first draw on),
how many numbers it matched in each round and whether it held the bonus ball. A date that is already
recorded, or whose draw isn't in the data yet, is skipped, so running it again is harmless.

    python scripts/score_live.py
"""
import json, os, subprocess, sys
from math import comb

import pandas as pd

POOL, PICK = 59, 6
OUT = "results/live_scores.json"
FROZEN = "results/frozen_wheel.json"


def git_json(ref: str, path: str):
    try:
        return json.loads(subprocess.run(["git", "show", f"{ref}:{path}"], check=True,
                                         capture_output=True, text=True).stdout)
    except subprocess.CalledProcessError:
        return None


def draw_rounds(df: pd.DataFrame, date: str) -> list:
    rows = df[df.date == date].sort_values("round")
    return [{"draw_no": int(r.draw_no), "round": int(r["round"]),
             "numbers": [int(r[f"n{i}"]) for i in range(1, 7)], "bonus": int(r.bonus)}
            for _, r in rows.iterrows()]


def score_line(line: list, rounds: list) -> dict:
    line = sorted(int(x) for x in line)
    return {"line": line, "matches": [len(set(line) & set(r["numbers"])) for r in rounds],
            "bonus": [r["bonus"] in line for r in rounds]}


def score_entry(combined: dict, frozen: dict | None, rounds: list, ref: str) -> dict:
    entry = {"date": combined["next_draw_date"], "draw_no": rounds[0]["draw_no"], "ref": ref, "rounds": rounds,
             "wheel59": {"source": "committed", "lines": [score_line(l, rounds) for l in combined["wheel"]["lines"]]}}
    if frozen and entry["date"] >= frozen["first_draw"]:
        entry["frozen_wheel"] = {"wheel": frozen["wheel"], "numbers": frozen["numbers"],
                                 "in_numbers": [len(set(frozen["numbers"]) & set(r["numbers"])) for r in rounds],
                                 "lines": [score_line(l, rounds) for l in frozen["lines"]]}
    return entry


def chance() -> dict:
    """Probability a single random line matches k numbers in one round."""
    total = comb(POOL, PICK)
    return {str(k): comb(PICK, k) * comb(POOL - PICK, PICK - k) / total for k in range(PICK + 1)}


def tally(scored_lines: list) -> dict:
    counts = [0] * (PICK + 1)
    bonus = 0
    for s in scored_lines:
        for m, b in zip(s["matches"], s["bonus"]):
            counts[m] += 1
            bonus += b
    n = sum(counts)
    return {"line_rounds": n, "counts": counts, "bonus": bonus,
            "mean": sum(k * c for k, c in enumerate(counts)) / n if n else None, "prize": sum(counts[3:])}


def summarise(entries: list) -> dict:
    return {"wheel59_lines": tally([s for e in entries for s in e["wheel59"]["lines"]]),
            "frozen_lines": tally([s for e in entries for s in (e.get("frozen_wheel") or {}).get("lines", [])]),
            "chance": chance(), "chance_mean": PICK * PICK / POOL}


def main(refs: list):
    df = pd.read_csv("data/uk_lotto_full.csv")
    record = json.load(open(OUT)) if os.path.exists(OUT) else {"draws": []}
    frozen = json.load(open(FROZEN)) if os.path.exists(FROZEN) else None
    done = {e["date"] for e in record["draws"]}
    for ref in refs:
        combined = git_json(ref, "results/combined_picks.json")
        if not combined:
            print(f"{ref}: no combined picks, skipped")
            continue
        date = combined["next_draw_date"]
        rounds = draw_rounds(df, date)
        if date in done:
            print(f"{ref}: {date} already recorded")
        elif not rounds:
            print(f"{ref}: draw for {date} not in the data yet")
        else:
            record["draws"].append(score_entry(combined, frozen, rounds, ref))
            done.add(date)
            best = max(m for s in record["draws"][-1]["wheel59"]["lines"] for m in s["matches"])
            print(f"{ref}: scored {date} (draw {rounds[0]['draw_no']}), best wheel line {best} matches")
    record["draws"].sort(key=lambda e: e["date"])
    record["summary"] = summarise(record["draws"])
    with open(OUT, "w") as fh:
        json.dump(record, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["origin/main"])
