"""Score the picks committed before each draw against the real result and keep a running
record in results/live_scores.json (shown in the report by scripts/build_report.py).

For each git ref given (default origin/main) it reads results/combined_picks.json and
results/walk_forward.json as they were at that ref, finds the draw on their next_draw_date in
data/uk_lotto_full.csv and records how many numbers each line matched in each round, and whether
it held the bonus ball. A date that is already recorded, or whose draw isn't in the data yet,
is skipped, so running it again is harmless.

    python scripts/score_live.py                     # after a draw: score main's picks
    python scripts/score_live.py 80dd700 ef73d8e     # backfill from older commits
"""
import json, os, subprocess, sys
from math import comb

import pandas as pd

POOL, PICK = 59, 6
OUT = "results/live_scores.json"


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


def score_entry(combined: dict, wf: dict, rounds: list, ref: str) -> dict:
    strategies = {}
    for name, lines in (wf or {}).get("predictions", {}).items():
        strategies[name] = [score_line(l, rounds) for l in lines]
    return {"date": combined["next_draw_date"], "draw_no": rounds[0]["draw_no"], "ref": ref,
            "rounds": rounds, "combined": [score_line(l, rounds) for l in combined["lines"]],
            "popularity_lines": [score_line(l, rounds) for l in combined.get("popularity_lines", [])],
            "wheel_lines": [score_line(l, rounds) for l in (combined.get("wheel") or {}).get("lines", [])],
            "strategies": strategies}


def chance() -> dict:
    """Probability a single random line matches k numbers in one round."""
    total = comb(POOL, PICK)
    return {str(k): comb(PICK, k) * comb(POOL - PICK, PICK - k) / total for k in range(PICK + 1)}


def summarise(entries: list) -> dict:
    def tally(groups):
        counts = [0] * (PICK + 1)
        bonus = 0
        for s in groups:
            for m, b in zip(s["matches"], s["bonus"]):
                counts[m] += 1
                bonus += b
        n = sum(counts)
        return {"line_rounds": n, "counts": counts, "bonus": bonus,
                "mean": sum(k * c for k, c in enumerate(counts)) / n if n else None,
                "prize": sum(counts[3:])}
    return {"combined": tally([s for e in entries for s in e["combined"]]),
            "strategies": tally([s for e in entries for ls in e["strategies"].values() for s in ls]),
            "popularity_lines": tally([s for e in entries for s in e.get("popularity_lines", [])]),
            "wheel_lines": tally([s for e in entries for s in e.get("wheel_lines", [])]),
            "chance": chance(), "chance_mean": PICK * PICK / POOL}


def main(refs: list):
    df = pd.read_csv("data/uk_lotto_full.csv")
    record = json.load(open(OUT)) if os.path.exists(OUT) else {"draws": []}
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
            continue
        if not rounds:
            print(f"{ref}: draw for {date} not in the data yet")
            continue
        wf = git_json(ref, "results/walk_forward.json")
        if wf and wf.get("next_draw_date") != date:
            wf = None  # strategy picks were for a different draw
        record["draws"].append(score_entry(combined, wf, rounds, ref))
        done.add(date)
        best = max(m for s in record["draws"][-1]["combined"] for m in s["matches"])
        print(f"{ref}: scored {date} (draw {rounds[0]['draw_no']}), best combined line {best} matches")
    record["draws"].sort(key=lambda e: e["date"])
    record["summary"] = summarise(record["draws"])
    with open(OUT, "w") as fh:
        json.dump(record, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["origin/main"])
