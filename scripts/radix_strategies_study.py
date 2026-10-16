"""The backtest's classic strategies, applied to the digits of each number in the first ten prime bases.

prime_radix_study.py writes every ball in bases 2, 3, 5, ... 29 and tests the digits for structure.
This study plays the same strategies the walk-forward backtest plays on numbers (hot, cold, overdue,
repeat the last draw), but on digits: a number is scored by how its digits, position by position,
have behaved, so 23 in base 5 (043) borrows from every number whose middle digit is 4. In base 59 or
above each number is its own digit, which gives the plain number versions as a reference.

Digits are not uniform over 1-59 (in base 7, a leading 1 covers 7-13 but a leading 8 only 56-59), so
every digit count is compared with what a fair draw gives for that digit class, and a number's score
is the product of its digits' lifts. Strategies, per base:

  hot-weighted   tickets drawn in proportion to the lift over the last 100 rounds
  cold-weighted  tickets drawn in proportion to the inverse lift
  hot top-6      the six highest lifts over the last 50 rounds
  cold top-6     the six lowest
  most overdue   the six whose digit classes have waited longest, relative to their expected wait
  repeat digits  the six sharing most digit positions with the last draw

Each is replayed walk-forward over the rounds after --split, scored against the 36/59 chance rate
with z from per-round means, and checked against a Bonferroni threshold over all base x strategy
pairs. A planted-bias check runs the same strategies on simulated draws where numbers ending in 0 in
base 5 are favoured, to show they would find a digit bias if there were one.

    python scripts/radix_strategies_study.py --out results/radix_strategies.json
"""
import argparse, json, math, os, sys, time
from math import comb

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lotto_nn.prime_pipeline import FIRST_10_PRIMES, digit_width  # noqa: E402

POOL, PICK = 59, 6
EXP_MEAN = PICK * PICK / POOL
SETS = ["n1", "n2", "n3", "n4", "n5", "n6"]
NUMBERS = 60  # "base 60": every number is its own digit, the plain-number reference
BASES = FIRST_10_PRIMES + [NUMBERS]
STRATEGIES = ["hot-weighted", "cold-weighted", "hot top-6", "cold top-6", "most overdue", "repeat digits"]
SAMPLED = {"hot-weighted", "cold-weighted"}


def base_label(b: int) -> str:
    return "numbers" if b == NUMBERS else f"base {b}"


class DigitClasses:
    """One-hot map from numbers 1-59 to (position, digit) classes in a base, with fair expectations."""

    def __init__(self, base: int):
        n = np.arange(1, POOL + 1)
        w = 1 if base == NUMBERS else digit_width(base)
        cols, self.cls = [], []
        for j in range(w):
            d = n if base == NUMBERS else (n // base ** j) % base
            for v in np.unique(d):
                cols.append(d == v)
                self.cls.append((j, int(v)))
        self.M = np.array(cols, dtype=float).T  # [59 numbers, classes]
        self.size = self.M.sum(0)  # numbers in each class
        self.w = w
        # chance a draw holds at least one number of the class, and the expected wait for it
        self.q = np.array([1 - comb(POOL - int(k), PICK) / comb(POOL, PICK) for k in self.size])

    def lift(self, counts: np.ndarray, rounds: int) -> np.ndarray:
        """Per-number product over positions of (class count / fair expectation), lightly smoothed."""
        exp = rounds * PICK * self.size / POOL
        cls_lift = (counts @ self.M + 1) / (exp + 1)
        return np.exp(np.log(cls_lift) @ self.M.T)

    def overdue(self, gaps: np.ndarray) -> np.ndarray:
        """Per-number product over positions of (rounds since the class last came up x its chance)."""
        cls_gap = np.array([gaps[self.M[:, c] > 0].min() for c in range(self.M.shape[1])])
        return np.exp(np.log(cls_gap * self.q + 1e-9) @ self.M.T)

    def shared(self, last: np.ndarray) -> np.ndarray:
        """How many digit positions each number shares with any number of the last draw."""
        seen = (self.M[last - 1].sum(0) > 0).astype(float)
        return self.M @ seen


def weighted(weights: np.ndarray, n: int, rng) -> np.ndarray:
    p = weights / weights.sum()
    return np.array([np.sort(rng.choice(np.arange(1, POOL + 1), PICK, replace=False, p=p)) for _ in range(n)])


def top6(score: np.ndarray, rng) -> np.ndarray:
    return np.sort(np.argsort(-(score + rng.random(POOL) * 1e-9))[:PICK] + 1)[None, :]


def tickets(X: np.ndarray, gaps: np.ndarray, t: int, dc: DigitClasses, strategy: str, n: int, rng) -> np.ndarray:
    """Tickets for round t from the indicator history X[:t] ([rounds, 59])."""
    if strategy in ("hot-weighted", "cold-weighted"):
        lift = dc.lift(X[max(0, t - 100):t].sum(0), min(t, 100))
        return weighted(lift if strategy == "hot-weighted" else 1 / lift, n, rng)
    if strategy in ("hot top-6", "cold top-6"):
        lift = dc.lift(X[max(0, t - 50):t].sum(0), min(t, 50))
        return top6(lift if strategy == "hot top-6" else -lift, rng)
    if strategy == "most overdue":
        return top6(dc.overdue(gaps), rng)
    return top6(dc.shared(np.flatnonzero(X[t - 1]) + 1), rng)  # repeat digits


def indicators(sets: np.ndarray) -> np.ndarray:
    X = np.zeros((len(sets), POOL), dtype=float)
    np.put_along_axis(X, sets - 1, 1.0, axis=1)
    return X


def gaps_before(X: np.ndarray) -> np.ndarray:
    """gaps[t, n]: rounds since number n was last drawn, as seen before round t."""
    G = np.zeros_like(X)
    g = np.full(POOL, len(X), dtype=float)
    for t in range(len(X)):
        G[t] = g
        g = np.where(X[t] > 0, 1.0, g + 1)
    return G


def backtest(sets: np.ndarray, test_idx: list, seeds: list, n_tickets: int) -> dict:
    X, G = indicators(sets), gaps_before(indicators(sets))
    out = {}
    for b in BASES:
        dc = DigitClasses(b)
        for s in STRATEGIES:
            zs, means = [], []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                rm = np.array([X[t][tickets(X, G[t], t, dc, s, n_tickets, rng) - 1].sum(1).mean() for t in test_idx])
                zs.append(float((rm.mean() - EXP_MEAN) / (rm.std(ddof=1) / math.sqrt(len(rm)))))
                means.append(float(rm.mean()))
                if s not in SAMPLED:
                    break  # deterministic apart from exact ties: one seed is enough
            out[f"{s}, {base_label(b)}"] = {"strategy": s, "base": base_label(b), "z_mean": float(np.mean(zs)),
                                             "z_min": float(np.min(zs)), "z_max": float(np.max(zs)),
                                             "mean_matches": float(np.mean(means))}
    return out


def planted(seed: int = 5, rounds: int = 1185, split: int = 755, boost: float = 1.5) -> dict:
    """Simulated draws where numbers that end in 0 in base 5 are `boost` times as likely."""
    rng = np.random.default_rng(seed)
    w = np.where(np.arange(1, POOL + 1) % 5 == 0, boost, 1.0)
    sets = np.array([np.sort(rng.choice(np.arange(1, POOL + 1), PICK, replace=False, p=w / w.sum()))
                     for _ in range(rounds)])
    X, G = indicators(sets), gaps_before(indicators(sets))
    res = {}
    for b in (5, 7, NUMBERS):
        dc, r = DigitClasses(b), np.random.default_rng(seed)
        rm = np.array([X[t][tickets(X, G[t], t, dc, "hot-weighted", 20, r) - 1].sum(1).mean() for t in range(split, rounds)])
        res[base_label(b)] = float((rm.mean() - EXP_MEAN) / (rm.std(ddof=1) / math.sqrt(len(rm))))
    return {"boost": boost, "favoured": "numbers ending in 0 in base 5 (5, 10, ... 55)", "hot_weighted_z": res}


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/uk_lotto_full.csv")
    ap.add_argument("--split-date", default="2022-12-31")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 1, 2])
    ap.add_argument("--tickets", type=int, default=20)
    ap.add_argument("--out", default="results/radix_strategies.json")
    a = ap.parse_args()
    t0 = time.time()
    df = pd.read_csv(a.csv)
    df = df[df.draw_no >= 2066].sort_values(["draw_no", "round"]).reset_index(drop=True)
    sets = df[SETS].to_numpy(int)
    test_idx = [i for i, d in enumerate(df.date) if d > a.split_date]
    print(f"1-59 era: {len(df)} rounds; replaying {len(STRATEGIES)} strategies x {len(BASES)} bases over {len(test_idx)} rounds")
    bt = backtest(sets, test_idx, a.seeds, a.tickets)
    radix = [k for k, v in bt.items() if v["base"] != "numbers"]
    alpha = 0.05 / len(radix)
    z_cut = next(z / 100 for z in range(100, 600) if norm_sf(z / 100) < alpha)
    best = max(radix, key=lambda k: bt[k]["z_mean"])
    for k in sorted(bt, key=lambda k: -bt[k]["z_mean"])[:8]:
        print(f"  {k:32s} z {bt[k]['z_mean']:+.2f} [{bt[k]['z_min']:+.2f}, {bt[k]['z_max']:+.2f}]")
    print(f"best radix strategy: {best} z {bt[best]['z_mean']:+.2f} (corrected threshold {z_cut:.2f} for {len(radix)} pairs)")
    # next-draw picks: one line per radix strategy and base, from every round
    X, G = indicators(sets), gaps_before(indicators(sets))
    Xn = np.vstack([X, np.zeros(POOL)])
    Gn = np.vstack([G, np.where(X[-1] > 0, 1.0, G[-1] + 1)])
    rng = np.random.default_rng(a.seeds[0])
    picks = {}
    for b in FIRST_10_PRIMES:
        dc = DigitClasses(b)
        for s in STRATEGIES:
            picks[f"{s}, {base_label(b)}"] = [sorted(int(x) for x in tickets(Xn, Gn[len(X)], len(X), dc, s, 1, rng)[0])]
    # what the best of this many unrelated strategies scores by chance alone
    exp_best = float(np.mean(np.max(np.random.default_rng(0).standard_normal((20000, len(radix))), axis=1)))
    pl = planted()
    print(f"planted bias check (hot-weighted z): {pl['hot_weighted_z']}")
    with open(a.out, "w") as fh:
        json.dump({"rounds": len(df), "split": a.split_date, "test_rounds": len(test_idx), "seeds": a.seeds,
                   "tickets_per_round": a.tickets, "strategies": STRATEGIES, "bases": [base_label(b) for b in BASES],
                   "results": bt, "pairs": len(radix), "z_cut": z_cut, "best": best, "expected_best": exp_best, "planted": pl,
                   "backtest": {"picks": picks}}, fh, indent=1)
    print(f"wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
