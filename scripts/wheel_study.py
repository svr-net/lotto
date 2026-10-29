"""The wheel (15 lines on all 59 numbers, no pair of numbers in two lines), UK Lotto 59-ball era.

The wheel plays every number, 31 of them twice (15 lines need 90 places), so in a round its lines
match 6 + k numbers between them, where k is how many of the drawn numbers are among the 31 played
twice. Choosing those 31 is the only prediction the wheel makes.

Part A backtests how the 31 are chosen. For every night from --split on, each rule sees only the
draws on earlier dates and picks 31 numbers, played in both of the night's rounds; what counts is k
(chance gives 6 x 31/59 = 3.15). A rule beats chance only if its z is above the Bonferroni cut over
all rules (z_cut); expected_best is the best z expected from that many rules with no edge. The
combined support the report uses is scored on the replayed draws too.

Part B compares the wheel's layout with 15 lines drawn independently, on simulated fair draws. Both
average the same matches and the same prize money; the layout changes how the wins are spread over
rounds: with no pair in two lines, their 2+ and 3+ matches almost never come together, so some line
wins in as many rounds as 15 lines can.

Part C plays the wheel on every real night of each period (since --split, and the backtest's test
period since --test-split), laid out from support the draws before that night give (decayed
frequencies, half-life 52 rounds; support carries no information, Part A, so it only decides which
numbers are played twice), and compares how often some line won with the simulated rates of Part B,
and its matches per line with chance.

Prize money counts the fixed tiers: 2 matches a free Lucky Dip (2), 3 matches 30, 4 matches 140 and
5 matches 1,750 (5 plus the bonus and the jackpot are left out).

    python scripts/wheel_study.py --out results/wheel_study.json
"""
import argparse, json, math, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_report as br  # noqa: E402

POOL, PICK, LINES = 59, 6, 15
CAND = LINES * PICK - POOL  # numbers played twice
N_COLS = [f"n{i}" for i in range(1, 7)]
IN_MEAN = PICK * CAND / POOL
IN_VAR = PICK * (CAND / POOL) * ((POOL - CAND) / POOL) * ((POOL - PICK) / (POOL - 1))
CHANCE_LINE = PICK * PICK / POOL
PRIZE = np.array([0, 0, 2, 30, 140, 1750, 0])  # by matches; 6 (the jackpot) left out


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def norm_isf(p):
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if norm_sf(mid) > p else (lo, mid)
    return (lo + hi) / 2


def onehot(sets):
    x = np.zeros((len(sets), POOL))
    x[np.repeat(np.arange(len(sets)), PICK), sets.ravel() - 1] = 1
    return x


# ---------------------------------------------------------------- Part A: choosing the 31 played twice
def top_n(score, rng):
    return np.argsort(-(np.asarray(score, float) + rng.random(POOL) * 1e-9))[:CAND] + 1


def decay(x, hl):
    w = 0.5 ** (np.arange(len(x))[::-1] / hl)
    return w @ x


def overdue(x):
    seen = np.where(x.any(0), len(x) - 1 - np.array([np.max(np.nonzero(x[:, j])[0]) if x[:, j].any() else -1 for j in range(POOL)]), len(x))
    return seen.astype(float)


def last_night(x, sets, rng):
    s = np.zeros(POOL)
    s[sets[-2:].ravel() - 1] = 2
    return s + decay(x, 52) / 1e3  # up to 12 numbers: fill with recent hot


def cohesive(x, rng, window=208):
    """Greedy: start from the hottest number, add the one that came up most often with those chosen."""
    h = x[-window:]
    pair = h.T @ h
    np.fill_diagonal(pair, 0)
    chosen = [int(np.argmax(h.sum(0) + rng.random(POOL) * 1e-9))]
    while len(chosen) < CAND:
        s = pair[chosen].sum(0) + rng.random(POOL) * 1e-9
        s[chosen] = -1
        chosen.append(int(np.argmax(s)))
    return np.array(chosen) + 1


RULES = {
    "random": lambda x, sets, rng: rng.choice(np.arange(1, POOL + 1), CAND, replace=False),
    "hot, last 26 rounds": lambda x, sets, rng: top_n(x[-26:].sum(0), rng),
    "hot, last 52 rounds": lambda x, sets, rng: top_n(x[-52:].sum(0), rng),
    "hot, last 104 rounds": lambda x, sets, rng: top_n(x[-104:].sum(0), rng),
    "hot, last 208 rounds": lambda x, sets, rng: top_n(x[-208:].sum(0), rng),
    "hot, whole era": lambda x, sets, rng: top_n(x.sum(0), rng),
    "cold, last 52 rounds": lambda x, sets, rng: top_n(-x[-52:].sum(0), rng),
    "cold, last 208 rounds": lambda x, sets, rng: top_n(-x[-208:].sum(0), rng),
    "cold, whole era": lambda x, sets, rng: top_n(-x.sum(0), rng),
    "most overdue": lambda x, sets, rng: top_n(overdue(x), rng),
    "least overdue": lambda x, sets, rng: top_n(-overdue(x), rng),
    "decayed hot, half-life 13": lambda x, sets, rng: top_n(decay(x, 13), rng),
    "decayed hot, half-life 52": lambda x, sets, rng: top_n(decay(x, 52), rng),
    "last night's numbers": lambda x, sets, rng: top_n(last_night(x, sets, rng), rng),
    "most cohesive pairs, last 208": lambda x, sets, rng: cohesive(x, rng),
}


def candidate_backtest(e, split, seeds):
    e = e.sort_values(["draw_no", "round"]).reset_index(drop=True)
    sets = e[N_COLS].to_numpy(int)
    x = onehot(sets)
    dates = e.date.to_numpy()
    nights = sorted(set(dates[dates > split]))
    res = {}
    for name, fn in RULES.items():
        per_seed = []
        for seed in range(seeds):
            rng = np.random.default_rng(3000 + seed)
            ins = []
            for d in nights:
                h = dates < d
                cand = set(int(c) for c in fn(x[h], sets[h], rng))
                ins += [len(cand & set(sets[r])) for r in np.where(dates == d)[0]]
            ins = np.array(ins)
            per_seed.append({"mean": float(ins.mean()), "z": float((ins.mean() - IN_MEAN) / math.sqrt(IN_VAR / len(ins)))})
        zs = [p["z"] for p in per_seed]
        res[name] = {"mean_in": float(np.mean([p["mean"] for p in per_seed])), "z_mean": float(np.mean(zs)),
                     "z_min": float(min(zs)), "z_max": float(max(zs))}
        print(f"  {name:32s} drawn among the {CAND}: {res[name]['mean_in']:.3f} (chance {IN_MEAN:.3f}) z {res[name]['z_mean']:+.2f} "
              f"({res[name]['z_min']:+.2f}..{res[name]['z_max']:+.2f})", flush=True)
    return res, len(nights), len(ins)


def replay_check(path="results/replay_five_lines.json"):
    """The report's own choice (the 31 numbers with most combined support, played twice) in the replayed draws."""
    if not os.path.exists(path):
        return None
    ks = []
    for d in json.load(open(path))["draws"]:
        lines = [s["line"] for s in (d.get("wheel") or {}).get("lines", [])]
        uses = np.bincount(np.array(lines, int).ravel(), minlength=POOL + 1) if lines else None
        if uses is None or (uses[1:] == 0).any():
            continue  # not the wheel on all 59 numbers
        twice = set(np.where(uses == 2)[0].tolist())
        ks += [len(twice & set(r["numbers"])) for r in d["rounds"]]
    if not ks:
        return None
    m = float(np.mean(ks))
    return {"rounds": len(ks), "mean_in": m, "chance_in": IN_MEAN, "z": (m - IN_MEAN) / math.sqrt(IN_VAR / len(ks))}


# ---------------------------------------------------------------- Part B: the layout
def random_lines(seed):
    rng = np.random.default_rng(seed)
    return [sorted(int(v) for v in rng.choice(np.arange(1, POOL + 1), PICK, replace=False)) for _ in range(LINES)]


def layout_stats(lines, n=1_000_000, seed=0, chunk=100_000):
    """Per round: chance some line matches 2+, 3+, 4+; mean prize money and its spread; and how the 3+
    wins come (rounds with any, lines won per such round)."""
    rng = np.random.default_rng(seed)
    L = np.array(lines)
    best = np.zeros(7)
    money, money2, wins3, rounds3 = 0.0, 0.0, 0, 0
    for _ in range(n // chunk):
        hits = np.zeros((chunk, POOL + 1), bool)
        np.put_along_axis(hits, np.argsort(rng.random((chunk, POOL)), axis=1)[:, :PICK] + 1, True, axis=1)
        m = hits[:, L].sum(2)  # rounds x lines
        b = m.max(1)
        for k in range(7):
            best[k] += (b >= k).sum()
        pay = PRIZE[m].sum(1)
        money += pay.sum()
        money2 += (pay.astype(float) ** 2).sum()
        w3 = (m >= 3).sum(1)
        wins3 += int(w3.sum())
        rounds3 += int((w3 > 0).sum())
    mean = money / n
    return {"distinct": len({v for l in lines for v in l}), "p2": best[2] / n, "p3": best[3] / n, "p4": best[4] / n,
            "money_mean": mean, "money_sd": math.sqrt(money2 / n - mean ** 2),
            "lines_won_per_3plus_round": wins3 / max(rounds3, 1)}


def real_draws(e, split):
    """Part C: the wheel on the real rounds since split."""
    e = e.sort_values(["draw_no", "round"]).reset_index(drop=True)
    sets = e[N_COLS].to_numpy(int)
    x = onehot(sets)
    dates = e.date.to_numpy()
    nights = sorted(set(dates[dates > split]))
    best, money, means = [], [], []
    for d in nights:
        h = dates < d
        L = np.array(br.spread_lines(decay(x[h], 52), int(d.replace("-", ""))))
        for r in np.where(dates == d)[0]:
            m = np.isin(L, sets[r]).sum(1)
            best.append(int(m.max()))
            money.append(int(PRIZE[m].sum()))
            means.append(float(m.mean()))
    best, means = np.array(best), np.array(means)
    # every number is played once or twice, so a round's lines match 6 + k between them (k: drawn among the 31)
    z = (means.mean() - CHANCE_LINE) / math.sqrt(IN_VAR / LINES ** 2 / len(means))
    o = {"split": split, "nights": len(nights), "rounds": len(best), "p2": float((best >= 2).mean()), "p3": float((best >= 3).mean()),
         "p4": float((best >= 4).mean()), "rounds_2plus": int((best >= 2).sum()), "rounds_3plus": int((best >= 3).sum()),
         "rounds_4plus": int((best >= 4).sum()), "money_mean": float(np.mean(money)), "line_mean": float(means.mean()), "line_z": float(z)}
    print(f"  since {split}: {o['rounds']} rounds  2+ {o['p2']:.3f}  3+ {o['p3']:.4f} ({o['rounds_3plus']})  4+ {o['rounds_4plus']}  "
          f"prize {o['money_mean']:.2f}  line mean {o['line_mean']:.4f} (z {z:+.2f})", flush=True)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/uk_lotto_full.csv")
    ap.add_argument("--split", default="2016-12-31")
    ap.add_argument("--test-split", default="2022-12-31")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sims", type=int, default=1_000_000)
    ap.add_argument("--out", default="results/wheel_study.json")
    a = ap.parse_args()
    e = pd.read_csv(a.csv)
    e = e[e.draw_no >= 2066].copy()
    print(f"Part A: choosing the {CAND} numbers played twice, nights from {a.split}, {a.seeds} seeds", flush=True)
    rules, n_nights, n_rounds = candidate_backtest(e, a.split, a.seeds)
    S = len(rules)
    z_cut = norm_isf(0.05 / S)
    rng = np.random.default_rng(0)
    expected_best = float(np.mean(rng.standard_normal((20000, S)).max(1)))
    replay = replay_check()
    if replay:
        print(f"  combined support (replayed draws): {replay['mean_in']:.3f} drawn among its {CAND} "
              f"(chance {replay['chance_in']:.3f}) over {replay['rounds']} rounds, z {replay['z']:+.2f}")
    print(f"  z_cut {z_cut:.2f}, expected best {expected_best:.2f}")
    print("Part B: the layout (simulated fair draws)", flush=True)
    score = np.ones(POOL)
    if os.path.exists("results/combined_picks.json"):
        score = np.array(json.load(open("results/combined_picks.json"))["score"])
    layouts = {"spread (59 numbers, no shared pair)": br.spread_lines(score),
               "random (15 independent lines)": random_lines(1)}
    stats = {}
    for name, lines in layouts.items():
        stats[name] = {"lines": lines, **layout_stats(lines, a.sims)}
        s = stats[name]
        print(f"  {name:38s} {s['distinct']:2d} numbers  2+ {s['p2']:.3f}  3+ {s['p3']:.4f}  4+ {s['p4']:.5f}  "
              f"prize {s['money_mean']:.2f} (sd {s['money_sd']:.1f})  lines won per 3+ round {s['lines_won_per_3plus_round']:.2f}", flush=True)
    print("Part C: the wheel on the real draws", flush=True)
    real = {"since 2017": real_draws(e, a.split), "test period": real_draws(e, a.test_split)}
    out = {"split": a.split, "test_split": a.test_split, "real": real, "nights": n_nights, "rounds": n_rounds, "candidates": CAND,
           "chance_in": IN_MEAN, "rules": rules, "z_cut": z_cut, "expected_best": expected_best, "replay": replay,
           "layouts": stats, "prizes": PRIZE.tolist(), "sims": a.sims}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
