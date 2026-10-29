"""Score the picks committed before each draw against the real result and keep a running
record in results/live_scores.json (shown in the report by scripts/build_report.py).

For each git ref given (default origin/main) it reads results/combined_picks.json and
results/walk_forward.json as they were at that ref, finds the draw on their next_draw_date in
data/uk_lotto_full.csv and records how many numbers each line matched in each round, and whether
it held the bonus ball. A date that is already recorded, or whose draw isn't in the data yet,
is skipped, so running it again is harmless.

    python scripts/score_live.py                     # after a draw: score main's picks
    python scripts/score_live.py 80dd700 ef73d8e     # backfill from older commits

Every draw also gets the wheel on all 59 numbers ("wheel59"): the committed wheel's lines once it is
the one committed, and before that the lines today's wheel makes from the combined support committed
for that draw (or, for a draw with no committed picks, the support the three-month replay rebuilt
from the draws before it), so the wheel's record runs back to the first scored draw.
"""
import json, os, subprocess, sys
from math import comb

import pandas as pd

POOL, PICK = 59, 6
OUT = "results/live_scores.json"
REPLAY = "results/replay_five_lines.json"


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
            "wheel59_lines": tally([s for e in entries for s in (e.get("wheel59") or {}).get("lines", [])]),
            "chance": chance(), "chance_mean": PICK * PICK / POOL}


def committed_scores() -> dict:
    """{next_draw_date: (commit, combined support)} from the history of results/combined_picks.json,
    keeping the latest commit for each date (the picks as they stood before that draw)."""
    out = {}
    log = subprocess.run(["git", "log", "--format=%h", "--", "results/combined_picks.json"],
                         capture_output=True, text=True).stdout.split()
    for sha in log:  # newest first
        c = git_json(sha, "results/combined_picks.json")
        if c and c.get("score") and c.get("next_draw_date") not in out:
            out[c["next_draw_date"]] = (sha, c["score"])
    return out


def backfill_wheel59(record: dict, df: pd.DataFrame) -> None:
    """Give every scored draw (and any replayed draw since the first one missing from the record) the
    wheel on all 59 numbers; see the module docstring."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import build_report as br
    if not record["draws"]:
        return
    first = min(e["date"] for e in record["draws"])
    replay = {d["date"]: d["score"] for d in (json.load(open(REPLAY))["draws"] if os.path.exists(REPLAY) else [])}
    have = {e["date"] for e in record["draws"]}
    for date in sorted(d for d in replay if d >= first and d not in have):
        rounds = draw_rounds(df, date)
        if rounds:  # a draw nobody committed picks for (the routine ran after it)
            record["draws"].append({"date": date, "draw_no": rounds[0]["draw_no"], "ref": None, "rounds": rounds,
                                    "combined": [], "popularity_lines": [], "wheel_lines": [], "strategies": {}})
    commits = None
    for e in record["draws"]:
        if "wheel59" in e:
            continue
        committed = e.get("wheel_lines") or []
        if len({v for s in committed for v in s["line"]}) == POOL:
            e["wheel59"] = {"source": "committed", "lines": committed}
            continue
        if commits is None:
            commits = committed_scores()
        if e["date"] in commits:
            source, score = f"rebuilt from {commits[e['date']][0]}", commits[e["date"]][1]
        elif e["date"] in replay:
            source, score = "rebuilt from the replay", replay[e["date"]]
        else:
            continue
        _, lines = br.wheel_lines(score, int(e["date"].replace("-", "")))
        e["wheel59"] = {"source": source, "lines": [score_line(l, e["rounds"]) for l in lines]}


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
    backfill_wheel59(record, df)
    record["draws"].sort(key=lambda e: e["date"])
    record["summary"] = summarise(record["draws"])
    with open(OUT, "w") as fh:
        json.dump(record, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["origin/main"])
