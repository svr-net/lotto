"""Hindsight as reality: fit every model to the whole 59-ball history, take the consensus 15 numbers and
play them in two wheels.

The models: the 15 numbers that would have caught 4+ of a round's six most often (a local search over
sets), the most likely biased machine (each number weighted by how often it has been drawn), the
combined support of all investigations (results/combined_picks.json), the vote of the backtest's
strategies (results/walk_forward.json), the four sequence rules (scripts/sequence_study.py) and three
frequency rules (scripts/wheel_study.py). Each model names 15 numbers; the consensus is the 15 named
most often (ties to the more often drawn number).

The wheels on the consensus 15:
  - "3 if 4", 15 lines: any 4 of the 15 share at least 3 numbers with some line;
  - "4 if 4", 141 lines (data/designs/c15_6_4_141.json): any 4 of the 15 sit together in some line.
Numbers are placed on each design so that as few lines as possible are pattern tickets or have two
consecutive pairs. Both guarantees are checked on every 4 of the 15. Odds for fair draws are exact
(every way a draw can fall on the 15 is enumerated); "if the fitted machine were real" draws balls in
proportion to each number's count so far. Out-of-sample tests find no such bias, so the fair odds are
the realistic ones.

    python scripts/consensus_wheels.py --out results/consensus_wheels.json
"""
import argparse, itertools, json, os, sys
from collections import Counter
from math import comb

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sequence_study as sq  # noqa: E402
import wheel_study as ws  # noqa: E402
from popularity_study import Popularity, pattern_line  # noqa: E402

POOL, PICK, V = 59, 6, 15
N_COLS = [f"n{i}" for i in range(1, 7)]
PRIZE = {2: 2.0, 3: 30, 4: 140, 5: 1750}  # 2 matches: a free Lucky Dip, counted as one line's price
W34 = [(0, 1, 6, 10, 12, 14), (0, 2, 4, 9, 10, 11), (3, 8, 9, 10, 13, 14), (1, 2, 3, 4, 6, 8), (2, 5, 8, 11, 12, 14),
       (0, 4, 5, 6, 8, 13), (5, 6, 7, 8, 9, 12), (1, 2, 5, 7, 10, 13), (3, 5, 6, 7, 10, 11), (3, 4, 7, 10, 12, 14),
       (1, 4, 9, 11, 12, 13), (0, 1, 7, 8, 11, 14), (0, 1, 3, 5, 9, 14), (0, 2, 3, 7, 12, 13), (2, 6, 9, 11, 13, 14)]


def onehot(sets):
    x = np.zeros((len(sets), POOL), np.int8)
    np.put_along_axis(x, sets - 1, 1, axis=1)
    return x


def best_set(x, restarts=60, seed=7):
    """Local search (one number out, one in) for the 15 numbers with the most rounds holding 4+ of them."""
    rng = np.random.default_rng(seed)
    best = (-1, None)
    for _ in range(restarts):
        s = rng.choice(POOL, V, replace=False)
        c = x[:, s].sum(1).astype(int)
        cur, improved = int((c >= 4).sum()), True
        while improved:
            improved = False
            for a in rng.permutation(s):
                for b in rng.permutation(np.setdiff1d(np.arange(POOL), s)):
                    c2 = c - x[:, a] + x[:, b]
                    v = int((c2 >= 4).sum())
                    if v > cur:
                        s, c, cur, improved = np.where(s == a, b, s), c2, v, True
                        break
                if improved:
                    break
        if cur > best[0]:
            best = (cur, np.sort(s))
    return best


def model_picks(sets, combined_score=None, wf_predictions=None):
    x = onehot(sets)
    xb = x.astype(bool)
    rng = np.random.default_rng(2026)
    top = lambda score: sorted(int(i) + 1 for i in np.argsort(-(np.asarray(score, float) + rng.random(POOL) * 1e-9))[:V])
    picks = {"best set in hindsight": [int(v) + 1 for v in best_set(x)[1]],
             "most likely biased machine": top(x.sum(0))}
    if combined_score is not None:
        picks["combined support of all investigations"] = top(combined_score)
    if wf_predictions:
        votes = Counter(v for s, ls in wf_predictions.items() if not s.startswith("random") for l in ls for v in l)
        picks["vote of the backtest's strategies"] = top([votes.get(n, 0) for n in range(1, POOL + 1)])
    g = sq.gaps_before(np.vstack([xb, np.zeros((1, POOL), bool)]))[-1]
    for name, fn in sq.RULES.items():
        if name != "random":
            picks[f"sequence: {name}"] = sorted(int(v) + 1 for v in fn(xb, g, np.random.default_rng(1), POOL))
    cand = ws.CAND
    ws.CAND = V
    try:
        for name in ("hot, last 208 rounds", "cold, whole era", "most cohesive pairs, last 208"):
            picks[f"frequency: {name}"] = sorted(int(v) for v in ws.RULES[name](x.astype(float), sets, np.random.default_rng(1)))
    finally:
        ws.CAND = cand
    votes = Counter(n for p in picks.values() for n in p)
    counts = x.sum(0)
    consensus = sorted(sorted(votes, key=lambda n: (-votes[n], -counts[n - 1]))[:V])
    return picks, consensus, {int(n): int(votes[n]) for n in consensus}


def place(cons, design, seed=0, steps=20000):
    """Put the 15 numbers on the design's points with as few pattern lines as possible."""
    rng = np.random.default_rng(seed)
    perm = list(range(V))
    mk = lambda: [sorted(int(cons[perm[i]]) for i in row) for row in design]
    lines = mk()
    bad = sum(map(pattern_line, lines))
    for _ in range(steps):
        if not bad:
            break
        i, j = rng.choice(V, 2, replace=False)
        perm[i], perm[j] = perm[j], perm[i]
        new = mk()
        b = sum(map(pattern_line, new))
        if b <= bad:
            lines, bad = new, b
        else:
            perm[i], perm[j] = perm[j], perm[i]
    return lines, int(bad)


def guarantee(lines, cons, match):
    """Every 4 of the 15 share at least `match` numbers with some line."""
    L = [set(l) for l in lines]
    return all(any(len(l & set(q)) >= match for l in L) for q in itertools.combinations(cons, 4))


def exact_odds(lines, cons):
    """Fair draws: chance some line matches 2+ ... 6, and fixed prizes per draw, by enumerating which of
    the 15 a draw holds (weighted by how many draws give each)."""
    idx = {n: i for i, n in enumerate(cons)}
    D = np.zeros((len(lines), V), bool)
    for i, l in enumerate(lines):
        D[i, [idx[n] for n in l]] = True
    tot = comb(POOL, PICK)
    best = {t: 0.0 for t in (2, 3, 4, 5, 6)}
    money = 0.0
    for k in range(PICK + 1):
        w = comb(POOL - V, PICK - k) / tot
        for S in itertools.combinations(range(V), k):
            m = D[:, list(S)].sum(1) if k else np.zeros(len(lines), int)
            for t in best:
                best[t] += w * (m.max() >= t)
            money += w * sum(PRIZE.get(int(v), 0) for v in m)
    return {str(t): float(p) for t, p in best.items()}, float(money)


def fitted_odds(lines, counts, n=300_000, seed=1):
    """If the fitted machine were real: balls drawn one at a time in proportion to each number's count."""
    rng = np.random.default_rng(seed)
    w = counts / counts.mean()
    g = np.log(w)[None, :] - np.log(-np.log(rng.random((n, POOL))))
    d = np.argsort(-g, axis=1)[:, :PICK] + 1
    hit = np.zeros((n, POOL + 1), bool)
    np.put_along_axis(hit, d, True, axis=1)
    m = hit[:, np.array(lines)].sum(2).max(1)
    return {str(t): float((m >= t).mean()) for t in (3, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/uk_lotto_full.csv")
    ap.add_argument("--design", default="data/designs/c15_6_4_141.json")
    ap.add_argument("--price", type=float, default=2.0)
    ap.add_argument("--out", default="results/consensus_wheels.json")
    a = ap.parse_args()
    e = pd.read_csv(a.csv)
    e = e[e.draw_no >= 2066].sort_values(["draw_no", "round"])
    sets = e[N_COLS].to_numpy(int)
    combined = json.load(open("results/combined_picks.json")) if os.path.exists("results/combined_picks.json") else {}
    wf = json.load(open("results/walk_forward.json")) if os.path.exists("results/walk_forward.json") else {}
    picks, cons, votes = model_picks(sets, combined.get("score"), wf.get("predictions"))
    counts = onehot(sets).sum(0)
    x = onehot(sets)
    print(f"history: {len(sets)} rounds to {e.date.iloc[-1]}; consensus 15: {cons}")
    pop = Popularity(*(lambda p: (p["beta"], p["lam"]))(json.load(open("results/popularity.json")))) \
        if os.path.exists("results/popularity.json") else None
    last = sorted(set(sets[-2:].ravel().tolist()))
    wheels = {}
    for name, design, match in (("3 if 4", W34, 3), ("4 if 4", json.load(open(a.design))["lines"], 4)):
        lines, bad = place(cons, design)
        ok = guarantee(lines, cons, match)
        assert ok, f"{name} guarantee failed"
        p, money = exact_odds(lines, cons)
        vals = [pop.value(l, last) for l in lines] if pop else []
        wheels[name] = {"lines": lines, "cost": len(lines) * a.price, "guarantee": ok, "pattern_lines": bad, "p": p,
                        "fixed_prizes": money, "fitted": fitted_odds(lines, counts),
                        "jackpot_value": [float(min(vals)), float(max(vals))] if vals else None}
        print(f"  {name}: {len(lines)} lines, cost {len(lines) * a.price:.0f}; fair 3+ {p['3']:.2%}, 4+ {p['4']:.3%}; "
              f"if fitted real 3+ {wheels[name]['fitted']['3']:.2%}, 4+ {wheels[name]['fitted']['4']:.3%}; pattern lines {bad}")
    rate = lambda s: float((x[:, np.array(s) - 1].sum(1) >= 4).mean())
    out = {"next_draw_date": combined.get("next_draw_date"), "rounds": len(sets), "last_date": str(e.date.iloc[-1]),
           "consensus": cons, "votes": votes, "models": {k: {"numbers": v, "history_rate": rate(v)} for k, v in picks.items()},
           "consensus_history_rate": rate(cons), "fair_4plus": sum(comb(V, k) * comb(POOL - V, PICK - k) for k in range(4, 7)) / comb(POOL, PICK),
           "price": a.price, "wheels": wheels}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
