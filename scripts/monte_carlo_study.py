"""Monte Carlo simulations for the Backtest tab, UK Lotto 59-ball era.

Part 1, what playing the wheel costs and returns: the report's 15 wheel lines (results/combined_picks.json)
are played on simulated fair rounds (six numbers and a bonus ball). Each round played costs --price per
line. Fixed prizes: 2 matches a free Lucky Dip (counted as one line's price), 3 matches 30, 4 matches 140,
5 matches 1,750, 5 plus the bonus 1,000,000. The jackpot (6 matches) is the median jackpot in the data
times the expected share a winner keeps (results/popularity.json). The rare tiers (5 plus the bonus, the
jackpot) almost never come up in a simulation, so their chances and value are worked out exactly; the
season figures (net result, chance of a prize, of breaking even) are simulated from the other tiers, and
the expected value adds the rare tiers back.

Part 2, how small a bias the tests could detect: simulated histories as long as the real era are drawn
with a planted bias (one number, or a set of six, made likelier by a factor 1 + delta; balls drawn one at
a time in proportion to their weights, as a biased machine would), and each history is put through:
  - the report's fairness test, a chi-square of the 59 numbers' counts (p < 0.05, calibrated on fair
    histories: six balls drawn without replacement vary a little less than the chi-square formula assumes);
  - the most-drawn number against the most-drawn number of fair histories (p < 0.05), the test that
    suits a single favoured number;
  - a walk-forward hot-numbers strategy over the last test_rounds rounds (each round, the six numbers
    drawn most often before it), which beats chance only if its z is above the cut the backtest uses.
The power of each test (share of biased histories it flags) gives the smallest bias it catches nine times
in ten, and how many extra matches a line on the favoured numbers would then average.

    python scripts/monte_carlo_study.py --out results/monte_carlo.json
"""
import argparse, json, math, os
from math import comb

import numpy as np
import pandas as pd

POOL, PICK = 59, 6
N_COLS = [f"n{i}" for i in range(1, 7)]
FIXED = {2: None, 3: 30, 4: 140, 5: 1750}  # 2 matches: a free Lucky Dip, worth one line's price
BONUS5 = 1_000_000
SEASONS = {"3 months": 26, "1 year": 104, "5 years": 520}  # rounds played (two draws a week, one round each)


def line_probs() -> dict:
    """Exact chance that one line matches k numbers (and, for 5, also the bonus ball)."""
    tot = comb(POOL, PICK)
    p = {k: comb(PICK, k) * comb(POOL - PICK, PICK - k) / tot for k in range(PICK + 1)}
    p5b = p[5] * 1 / (POOL - PICK)  # the bonus is the one missing number among the 53 others
    return {**p, "5b": p5b, "5": p[5] - p5b}


# ---------------------------------------------------------------- Part 1: playing the wheel
def round_prizes(lines, n, price, rng, chunk=200_000):
    """Fixed prizes (2 to 5 matches, bonus ignored) won by the lines in each of n simulated rounds."""
    L = np.array(lines)
    pay = np.zeros(PICK + 1)
    pay[2], pay[3], pay[4], pay[5] = price, FIXED[3], FIXED[4], FIXED[5]
    out, best = [], []
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        hits = np.zeros((m, POOL + 1), bool)
        np.put_along_axis(hits, np.argsort(rng.random((m, POOL)), axis=1)[:, :PICK] + 1, True, axis=1)
        k = hits[:, L].sum(2)
        out.append(pay[k].sum(1))
        best.append(k.max(1))
    return np.concatenate(out), np.concatenate(best)


def wheel_season(lines, price, jackpot, rounds_sim, seasons, rng) -> dict:
    n_lines = len(lines)
    cost = n_lines * price
    prize, best = round_prizes(lines, rounds_sim, price, rng)
    p = line_probs()
    rare_ev = n_lines * (p["5b"] * BONUS5 + p[6] * jackpot)
    per_round = {"cost": cost, "fixed_mean": float(prize.mean()), "rare_ev": rare_ev,
                 "ev": float(prize.mean()) + rare_ev, "return_pct": (float(prize.mean()) + rare_ev) / cost * 100,
                 "p_any_prize": float((best >= 3).mean()), "p_free_line": float((best >= 2).mean())}
    out = {}
    cash = (best >= 3)
    for label, k in SEASONS.items():
        idx = rng.integers(0, len(prize), (seasons, k))
        won, prizes = prize[idx].sum(1), cash[idx].sum(1)
        net = won - k * cost
        out[label] = {"rounds": k, "cost": k * cost, "won_mean": float(won.mean()), "net_mean": float(net.mean()),
                      "net_p5": float(np.percentile(net, 5)), "net_median": float(np.median(net)),
                      "net_p95": float(np.percentile(net, 95)), "p_break_even": float((net >= 0).mean()),
                      "prizes_mean": float(prizes.mean()), "p_no_prize": float((prizes == 0).mean()),
                      "p_jackpot": 1 - (1 - n_lines * p[6]) ** k, "p_bonus5": 1 - (1 - n_lines * p["5b"]) ** k,
                      "p_five": 1 - (1 - n_lines * (p[5] + p["5b"])) ** k}
    return {"per_round": per_round, "seasons": out, "price": price, "jackpot": jackpot, "lines": n_lines}


# ---------------------------------------------------------------- Part 2: power against planted biases
def biased_history(n, weights, rng):
    """n rounds of six balls drawn one at a time in proportion to their weights (a biased machine)."""
    g = np.log(weights)[None, :] - np.log(-np.log(rng.random((n, POOL))))  # Gumbel top-k
    return np.argsort(-g, axis=1)[:, :PICK] + 1


def counts_stats(h):
    c = np.bincount(h.ravel(), minlength=POOL + 1)[1:]
    e = len(h) * PICK / POOL
    return float(((c - e) ** 2 / e).sum()), int(c.max())


def tests(h, test_rounds, z_cut, null):
    x = np.zeros((len(h), POOL))
    np.put_along_axis(x, h - 1, 1, axis=1)
    chi, top_count = counts_stats(h)
    p_chi = (np.sum(null[:, 0] >= chi) + 1) / (len(null) + 1)
    p_max = (np.sum(null[:, 1] >= top_count) + 1) / (len(null) + 1)
    # walk-forward: each test round, play the six numbers drawn most often before it
    start = len(h) - test_rounds
    cum = np.cumsum(x, axis=0)
    prior = cum[start - 1 : len(h) - 1]  # counts before each test round
    top = np.argsort(-(prior + np.random.default_rng(0).random(prior.shape) * 1e-6), axis=1)[:, :PICK]
    m = np.take_along_axis(x[start:], top, axis=1).sum(1)
    z = (m.mean() - PICK * PICK / POOL) / (m.std(ddof=1) / math.sqrt(len(m)) + 1e-12)
    return p_chi < 0.05, p_max < 0.05, z > z_cut


def detect_at(rows, test, level=0.9):
    """The bias at which a test's power first reaches level, interpolated between grid points."""
    for lo, hi in zip(rows, rows[1:]):
        if hi[test] >= level > lo[test]:
            return lo["delta"] + (level - lo[test]) / (hi[test] - lo[test]) * (hi["delta"] - lo["delta"])
    return None if rows[-1][test] < level else rows[0]["delta"]


def power(n_rounds, test_rounds, z_cut, sims, deltas, rng) -> dict:
    null = np.array([counts_stats(biased_history(n_rounds, np.ones(POOL), rng)) for _ in range(2000)])
    out = {}
    for scen, favoured in (("one number", 1), ("six numbers", 6)):
        rows = []
        for d in deltas:
            w = np.ones(POOL)
            w[:favoured] = 1 + d
            hits = np.array([tests(biased_history(n_rounds, w, rng), test_rounds, z_cut, null) for _ in range(sims)])
            # matches a line holding the favoured numbers (and others to make six) averages, against 36/59:
            # f favoured numbers drawn per round on average, the other 6 - f spread over the 59 - favoured
            f = float(np.isin(biased_history(200_000, w, rng), np.arange(1, favoured + 1)).sum(1).mean())
            mean_fav = f + (PICK - favoured) * (PICK - f) / (POOL - favoured)
            rows.append({"delta": d, "chi_square": float(hits[:, 0].mean()), "max_count": float(hits[:, 1].mean()),
                         "strategy": float(hits[:, 2].mean()), "line_mean": mean_fav})
            print(f"  {scen:12s} +{d:4.0%}: power chi-square {rows[-1]['chi_square']:.2f}, most-drawn {rows[-1]['max_count']:.2f}, "
                  f"strategy {rows[-1]['strategy']:.2f}; a line on them averages {mean_fav:.3f}", flush=True)
        out[scen] = {"rows": rows, "detect90": {t: detect_at(rows, t) for t in ("chi_square", "max_count", "strategy")}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/uk_lotto_full.csv")
    ap.add_argument("--picks", default="results/combined_picks.json")
    ap.add_argument("--price", type=float, default=2.0)
    ap.add_argument("--rounds-sim", type=int, default=2_000_000)
    ap.add_argument("--seasons", type=int, default=100_000)
    ap.add_argument("--sims", type=int, default=500)
    ap.add_argument("--test-rounds", type=int, default=440)
    ap.add_argument("--z-cut", type=float, default=2.9)
    ap.add_argument("--out", default="results/monte_carlo.json")
    a = ap.parse_args()
    rng = np.random.default_rng(2026)
    e = pd.read_csv(a.csv)
    e = e[e.draw_no >= 2066]
    share = 1.0
    if os.path.exists("results/popularity.json"):
        share = json.load(open("results/popularity.json")).get("typical_share", 1.0)
    jackpot = float(e.jackpot.median()) * share
    lines = json.load(open(a.picks))["wheel"]["lines"]
    print(f"Part 1: the wheel's {len(lines)} lines at {a.price:.2f} a line, jackpot {jackpot:,.0f}", flush=True)
    season = wheel_season(lines, a.price, jackpot, a.rounds_sim, a.seasons, rng)
    pr = season["per_round"]
    print(f"  per round: cost {pr['cost']:.2f}, fixed prizes {pr['fixed_mean']:.2f}, rare tiers {pr['rare_ev']:.2f}, "
          f"return {pr['return_pct']:.1f}%")
    for k, s in season["seasons"].items():
        print(f"  {k:9s}: cost {s['cost']:,.0f}, net median {s['net_median']:,.0f} (5-95% {s['net_p5']:,.0f} to {s['net_p95']:,.0f}), "
              f"break even {s['p_break_even']:.1%}, no prize {s['p_no_prize']:.1%}, jackpot {s['p_jackpot']:.4%}", flush=True)
    n_rounds = len(e)
    print(f"Part 2: planted biases in {n_rounds}-round histories, {a.sims} each", flush=True)
    pw = power(n_rounds, a.test_rounds, a.z_cut, a.sims, [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0], rng)
    out = {"season": season, "power": pw, "rounds": n_rounds, "test_rounds": a.test_rounds, "z_cut": a.z_cut,
           "sims": a.sims, "rounds_sim": a.rounds_sim, "chance_line": PICK * PICK / POOL}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
