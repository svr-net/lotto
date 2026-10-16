"""Every strategy family, applied to the digits of each number in the first ten prime bases.

prime_radix_study.py writes every ball in bases 2, 3, 5, ... 29 and tests the digits for structure.
This study plays the strategies of all the other investigations on those digits: a number is
scored by how its digits, position by position, have behaved, so 23 in base 5 (043) borrows from
every number whose middle digit is 4. In "base 60" each number is its own digit, which gives
whole-number versions of the same strategies as a reference.

Digits are not uniform over 1-59 (in base 7, a leading 1 covers 7-13 but a leading 8 only 56-59), so
every digit count is compared with what a fair draw gives that digit class, and a number's score is
the product over positions of its digits' lifts. The strategies, by the investigation they come from:

  classic         hot-weighted, cold-weighted, hot top-6, cold top-6, most overdue, repeat digits
  grammar         a chain over the sorted draw: next digit given the previous ball's digit
  draw order      early-ball digits (1st/2nd out), drawn-order digit Markov chain,
                  machine-matched hot digits, recent bonus-ball digits
  Fourier         each digit class's count series extended one round by its 3 strongest sinusoids
  wavelets        digit hotness smoothed over the last 8 rounds
  Bayesian        posterior-mean and Thompson-sampled digit probabilities (Dirichlet prior at fair)
  machine learn.  logistic regression on digit features (lifts over 4 windows, overdue, repeats)
  reinforcement   Hedge with full information over all the strategies above, in each base

Each is replayed walk-forward over the rounds after --split, scored against the 36/59 chance rate
with z from per-round means, and checked against a Bonferroni threshold over all base x strategy
pairs. A planted-bias check runs hot-weighted on simulated draws where numbers ending in 0 in base 5
are favoured, to show a digit bias would be found if there were one.

    python scripts/radix_strategies_study.py --out results/radix_strategies.json
"""
import argparse, json, math, os, sys, time
from math import comb

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from lotto_nn.prime_pipeline import FIRST_10_PRIMES, digit_width  # noqa: E402
from spectral_study import fourier_forecast  # noqa: E402

POOL, PICK = 59, 6
EXP_MEAN = PICK * PICK / POOL
SETS = ["n1", "n2", "n3", "n4", "n5", "n6"]
DRAWN = ["d1", "d2", "d3", "d4", "d5", "d6"]
NUMBERS = 60  # "base 60": every number is its own digit, the whole-number reference
BASES = FIRST_10_PRIMES + [NUMBERS]
GROUPS = {
    "Classic": ["hot-weighted", "cold-weighted", "hot top-6", "cold top-6", "most overdue", "repeat digits"],
    "Grammar": ["digit grammar"],
    "Draw order": ["early-ball digits", "drawn-order digit chain", "machine-matched digits", "bonus-ball digits"],
    "Fourier and wavelets": ["Fourier extrapolation", "8-round smoothed hotness"],
    "Bayesian": ["posterior-mean digits", "Thompson-sampled digits"],
    "Machine learning": ["logistic regression"],
    "Reinforcement learning": ["Hedge over the strategies"],
}
STRATEGIES = [s for g in GROUPS.values() for s in g]
SAMPLED = {"hot-weighted", "cold-weighted", "digit grammar", "early-ball digits", "drawn-order digit chain",
           "machine-matched digits", "Fourier extrapolation", "8-round smoothed hotness", "Thompson-sampled digits"}
WARMUP, RETRAIN, PRIOR = 200, 26, 1000.0


def base_label(b: int) -> str:
    return "numbers" if b == NUMBERS else f"base {b}"


class DigitClasses:
    """One-hot maps from numbers 1-59 to digit classes, per position, with fair expectations."""

    def __init__(self, base: int):
        n = np.arange(1, POOL + 1)
        self.w = 1 if base == NUMBERS else digit_width(base)
        self.pos = []  # per position: [59, classes] one-hot
        for j in range(self.w):
            d = n if base == NUMBERS else (n // base ** j) % base
            self.pos.append(np.array([d == v for v in np.unique(d)], dtype=float).T)
        self.M = np.hstack(self.pos)  # [59, all classes]
        self.size = self.M.sum(0)
        self.share = self.size / POOL  # fair chance a ball falls in the class (within its position)
        self.q = np.array([1 - comb(POOL - int(k), PICK) / comb(POOL, PICK) for k in self.size])

    def number_score(self, cls_lift: np.ndarray) -> np.ndarray:
        """Per-number product of its digit classes' lifts (works on [..., classes])."""
        return np.exp(np.log(np.maximum(cls_lift, 1e-9)) @ self.M.T)

    def lift(self, counts: np.ndarray, balls: float) -> np.ndarray:
        """Number scores from per-number counts over `balls` drawn balls, lightly smoothed."""
        exp = balls * self.share
        return self.number_score((counts @ self.M + 1) / (exp + 1))

    def overdue(self, gaps: np.ndarray) -> np.ndarray:
        cls_gap = np.array([gaps[self.M[:, c] > 0].min() for c in range(self.M.shape[1])])
        return self.number_score(cls_gap * self.q + 1e-9)

    def shared(self, last: np.ndarray) -> np.ndarray:
        return self.M @ (self.M[last - 1].sum(0) > 0).astype(float)

    def transitions(self, seqs: np.ndarray, smooth: float = 20.0):
        """Start and next-ball lift matrices [59] and [59, 59] from ball sequences (rows in order)."""
        start = np.zeros(POOL)
        trans = np.zeros((POOL, POOL))
        for P in self.pos:
            k = P.shape[1]
            e = P.sum(0) / POOL
            first = np.bincount(np.argmax(P[seqs[:, 0] - 1], 1), minlength=k)
            a, b = np.argmax(P[seqs[:, :-1].ravel() - 1], 1), np.argmax(P[seqs[:, 1:].ravel() - 1], 1)
            T = np.zeros((k, k))
            np.add.at(T, (a, b), 1)
            s_lift = (first + smooth * e) / (first.sum() + smooth) / e
            t_lift = (T + smooth * e) / (T.sum(1, keepdims=True) + smooth) / e
            start += np.log(s_lift) @ P.T
            trans += P @ np.log(t_lift) @ P.T
        return np.exp(start), np.exp(trans)


# ---------------------------------------------------------------- history and ticket helpers
def indicators(sets: np.ndarray) -> np.ndarray:
    X = np.zeros((len(sets), POOL), dtype=float)
    np.put_along_axis(X, np.asarray(sets) - 1, 1.0, axis=1)
    return X


def gaps_before(X: np.ndarray) -> np.ndarray:
    """G[t, n]: rounds since number n was last drawn, as seen before round t."""
    G = np.zeros_like(X)
    g = np.full(POOL, len(X), dtype=float)
    for t in range(len(X)):
        G[t] = g
        g = np.where(X[t] > 0, 1.0, g + 1)
    return G


def weighted(weights: np.ndarray, n: int, rng) -> np.ndarray:
    p = np.maximum(weights, 1e-12)
    p = p / p.sum()
    return np.array([np.sort(rng.choice(np.arange(1, POOL + 1), PICK, replace=False, p=p)) for _ in range(n)])


def top6(score: np.ndarray, rng) -> np.ndarray:
    return np.sort(np.argsort(-(score + rng.random(POOL) * 1e-9))[:PICK] + 1)[None, :]


def chain(start: np.ndarray, trans: np.ndarray, n: int, rng, ascending: bool) -> np.ndarray:
    """Sample n lines ball by ball: first in proportion to `start`, then to `trans[previous]`.
    With `ascending`, each ball is above the last and leaves room for the rest (a sorted draw)."""
    out = []
    nums = np.arange(1, POOL + 1)
    for _ in range(n):
        line, w = [], start.copy()
        for k in range(PICK):
            ok = np.ones(POOL, dtype=bool)
            ok[np.array(line, dtype=int) - 1] = False
            if ascending:
                ok &= (nums > (line[-1] if line else 0)) & (nums <= POOL - (PICK - 1 - k))
            p = np.where(ok, w, 0)
            nxt = int(rng.choice(nums, p=p / p.sum()))
            line.append(nxt)
            w = trans[nxt - 1]
        out.append(sorted(line))
    return np.array(out)


class History:
    """Everything the strategies may look at, for rounds before t."""

    def __init__(self, df: pd.DataFrame):
        self.sets = df[SETS].to_numpy(int)
        self.drawn = df[DRAWN].to_numpy(int)
        self.bonus = df.bonus.to_numpy(int)
        self.machine = df.machine.astype(str).str.strip().to_numpy()
        self.round = df["round"].to_numpy(int)
        self.X = indicators(self.sets)
        self.G = gaps_before(self.X)
        z = np.zeros((1, POOL))
        self.C = np.vstack([z, np.cumsum(self.X, 0)])  # C[t] = counts over rounds < t
        self.E = np.vstack([z, np.cumsum(indicators(self.drawn[:, :2]), 0)])  # 1st/2nd balls out
        Bx = np.zeros((len(self.bonus), POOL))
        Bx[np.arange(len(self.bonus)), self.bonus - 1] = 1
        self.B = np.vstack([z, np.cumsum(Bx, 0)])

    def window(self, A: np.ndarray, t: int, w: int) -> np.ndarray:
        return A[t] - A[max(0, t - w)]

    def machine_counts(self, t: int, target_round: int) -> tuple:
        same = np.flatnonzero(self.round[:t] == target_round)
        m = self.machine[same[-1] if len(same) else t - 1]
        rows = np.flatnonzero(self.machine[:t] == m)[-300:]
        if len(rows) < 30:
            rows = np.arange(max(0, t - 300), t)
        return self.X[rows].sum(0), len(rows)


def digit_features(h: History, dc: DigitClasses) -> np.ndarray:
    """F[t, n, f] for every round t, from rounds before t: log lift over 10/50/200/all rounds,
    log overdue score, and digit positions shared with the last draw."""
    T = len(h.sets)
    feats = []
    for w in (10, 50, 200, None):
        lo = np.arange(T) if w is None else np.maximum(np.arange(T) - w, 0)
        cnt = h.C[np.arange(T)] - (0 if w is None else h.C[lo])
        balls = PICK * (np.arange(T) - (0 if w is None else lo))
        cls = (cnt @ dc.M + 1) / (balls[:, None] * dc.share + 1)
        feats.append(np.log(cls) @ dc.M.T)
    over = np.stack([np.log(dc.overdue(h.G[t])) for t in range(T)])
    last = np.vstack([np.zeros(POOL), np.stack([dc.shared(h.sets[t]) for t in range(T - 1)])])
    return np.stack(feats + [over, last], axis=-1)


# ---------------------------------------------------------------- strategies
def make_tickets(h: History, dc: DigitClasses, s: str, t: int, n: int, rng, target_round: int, ctx: dict) -> np.ndarray:
    """Tickets for round t of strategy s in the digit classes dc, from rounds before t."""
    if s in ("hot-weighted", "cold-weighted"):
        lift = dc.lift(h.window(h.C, t, 100), PICK * min(t, 100))
        return weighted(lift if s == "hot-weighted" else 1 / lift, n, rng)
    if s in ("hot top-6", "cold top-6"):
        lift = dc.lift(h.window(h.C, t, 50), PICK * min(t, 50))
        return top6(lift if s == "hot top-6" else -lift, rng)
    if s == "most overdue":
        return top6(dc.overdue(h.G[t]), rng)
    if s == "repeat digits":
        return top6(dc.shared(h.sets[t - 1]), rng)
    if s == "digit grammar":
        return chain(*dc.transitions(h.sets[:t]), n, rng, ascending=True)
    if s == "drawn-order digit chain":
        return chain(*dc.transitions(h.drawn[:t]), n, rng, ascending=False)
    if s == "early-ball digits":
        return weighted(dc.lift(h.window(h.E, t, 200), 2 * min(t, 200)), n, rng)
    if s == "machine-matched digits":
        cnt, rows = h.machine_counts(t, target_round)
        return weighted(dc.lift(cnt, PICK * rows), n, rng)
    if s == "bonus-ball digits":
        return top6(dc.lift(h.window(h.B, t, 30), min(t, 30)), rng)
    if s == "Fourier extrapolation":
        Y = h.X[t - 256:t] @ dc.M  # [rounds, classes]
        pred = np.maximum(fourier_forecast(Y.T, 3), 0) + 0.02
        return weighted(dc.number_score(pred / (PICK * dc.share)), n, rng)
    if s == "8-round smoothed hotness":
        return weighted(dc.lift(h.window(h.C, t, 8), PICK * min(t, 8)), n, rng)
    if s in ("posterior-mean digits", "Thompson-sampled digits"):
        cnt = h.C[t] @ dc.M
        a = PRIOR * dc.share + cnt
        if s == "posterior-mean digits":
            post = np.concatenate([a[c] / a[c].sum() for c in ctx["cols"]])
            return top6(dc.number_score(post / dc.share), rng)
        out = []
        for _ in range(n):
            g = rng.gamma(a)
            post = np.concatenate([g[c] / g[c].sum() for c in ctx["cols"]])
            out.append(top6(dc.number_score(post / dc.share), rng)[0])
        return np.array(out)
    if s == "logistic regression":
        model = ctx.get("model")
        if model is None or t - ctx["trained_on"] >= RETRAIN:
            from sklearn.linear_model import LogisticRegression
            F = ctx["F"]
            rows = np.arange(WARMUP, t)
            model = LogisticRegression(C=1.0, max_iter=500).fit(F[rows].reshape(-1, F.shape[-1]), h.X[rows].ravel())
            ctx.update(model=model, trained_on=t)
        return top6(model.predict_proba(ctx["Fn"](t))[:, 1], rng)
    raise ValueError(s)


def position_columns(dc: DigitClasses) -> list:
    cols, i = [], 0
    for P in dc.pos:
        cols.append(np.arange(i, i + P.shape[1]))
        i += P.shape[1]
    return cols


def backtest(h: History, test_idx: list, seeds: list, n_tickets: int) -> dict:
    out = {}
    T = len(h.sets)
    for b in BASES:
        t0 = time.time()
        dc = DigitClasses(b)
        F = digit_features(h, dc)
        per_round = {}  # first seed's per-round mean matches, for Hedge
        for s in STRATEGIES[:-1]:
            zs, means = [], []
            for si, seed in enumerate(seeds):
                rng = np.random.default_rng(seed)
                ctx = {"cols": position_columns(dc), "F": F, "Fn": lambda t: F[t]}
                rm = np.array([h.X[t][make_tickets(h, dc, s, t, n_tickets, rng, h.round[t], ctx) - 1].sum(1).mean()
                               for t in test_idx])
                if si == 0:
                    per_round[s] = rm
                zs.append(float((rm.mean() - EXP_MEAN) / (rm.std(ddof=1) / math.sqrt(len(rm)))))
                means.append(float(rm.mean()))
                if s not in SAMPLED:
                    break  # deterministic apart from exact ties: one seed is enough
            out[f"{s}, {base_label(b)}"] = {"strategy": s, "base": base_label(b), "z_mean": float(np.mean(zs)),
                                             "z_min": float(np.min(zs)), "z_max": float(np.max(zs)),
                                             "mean_matches": float(np.mean(means))}
        # Hedge: weights from each strategy's cumulative reward so far, applied to the next round
        R = np.array([per_round[s] for s in STRATEGIES[:-1]])
        eta = math.sqrt(8 * math.log(len(R)) / R.shape[1])
        cum = np.cumsum(R, axis=1) - R
        w = np.exp(eta * (cum - cum.max(0)))
        hedge = (w / w.sum(0) * R).sum(0)
        z = float((hedge.mean() - EXP_MEAN) / (hedge.std(ddof=1) / math.sqrt(len(hedge))))
        lead = STRATEGIES[int(np.argmax(cum[:, -1] + R[:, -1]))]
        out[f"Hedge over the strategies, {base_label(b)}"] = {
            "strategy": "Hedge over the strategies", "base": base_label(b), "z_mean": z, "z_min": z, "z_max": z,
            "mean_matches": float(hedge.mean()), "leader": lead}
        best = max((k for k in out if k.endswith(base_label(b))), key=lambda k: out[k]["z_mean"])
        print(f"  {base_label(b):8s} best {best.split(',')[0]:28s} z {out[best]['z_mean']:+.2f} ({time.time() - t0:.0f}s)", flush=True)
    return out


def next_picks(h: History, seed: int) -> dict:
    """One next-draw line per strategy and prime base, from every round."""
    T = len(h.sets)
    # extend the history by a blank round so every strategy can look "before" it
    hx = History.__new__(History)
    for k, v in vars(h).items():
        setattr(hx, k, v)
    hx.G = np.vstack([h.G, np.where(h.X[-1] > 0, 1.0, h.G[-1] + 1)])
    hx.round = np.append(h.round, 1)
    rng = np.random.default_rng(seed)
    picks = {}
    for b in FIRST_10_PRIMES:
        dc = DigitClasses(b)
        F = digit_features(h, dc)
        # features for the round after the last: rebuild with one round appended
        hn = History.__new__(History)
        for k, v in vars(hx).items():
            setattr(hn, k, v)
        hn.sets = np.vstack([h.sets, h.sets[-1]])
        Fn = digit_features(hn, dc)[-1]
        ctx = {"cols": position_columns(dc), "F": F, "Fn": lambda t, Fn=Fn: Fn}
        lines = {}
        for s in STRATEGIES[:-1]:
            tk = make_tickets(hx, dc, s, T, 1, rng, 1, ctx)
            lines[s] = sorted(int(x) for x in tk[0])
            picks[f"{s}, {base_label(b)}"] = [lines[s]]
    return picks


def planted(seed: int = 5, rounds: int = 1185, split: int = 755, boost: float = 1.5) -> dict:
    """Simulated draws where numbers that end in 0 in base 5 are `boost` times as likely."""
    rng = np.random.default_rng(seed)
    w = np.where(np.arange(1, POOL + 1) % 5 == 0, boost, 1.0)
    sets = np.array([np.sort(rng.choice(np.arange(1, POOL + 1), PICK, replace=False, p=w / w.sum()))
                     for _ in range(rounds)])
    X = indicators(sets)
    C = np.vstack([np.zeros((1, POOL)), np.cumsum(X, 0)])
    res = {}
    for b in (5, 7, NUMBERS):
        dc, r = DigitClasses(b), np.random.default_rng(seed)
        rm = np.array([X[t][weighted(dc.lift(C[t] - C[t - 100], PICK * 100), 20, r) - 1].sum(1).mean()
                       for t in range(split, rounds)])
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
    h = History(df)
    test_idx = [i for i, d in enumerate(df.date) if d > a.split_date]
    print(f"1-59 era: {len(df)} rounds; replaying {len(STRATEGIES)} strategies x {len(BASES)} bases over {len(test_idx)} rounds")
    bt = backtest(h, test_idx, a.seeds, a.tickets)
    radix = [k for k, v in bt.items() if v["base"] != "numbers"]
    alpha = 0.05 / len(radix)
    z_cut = next(z / 100 for z in range(100, 700) if norm_sf(z / 100) < alpha)
    best = max(radix, key=lambda k: bt[k]["z_mean"])
    for k in sorted(bt, key=lambda k: -bt[k]["z_mean"])[:8]:
        print(f"  {k:46s} z {bt[k]['z_mean']:+.2f} [{bt[k]['z_min']:+.2f}, {bt[k]['z_max']:+.2f}]")
    print(f"best radix strategy: {best} z {bt[best]['z_mean']:+.2f} (corrected threshold {z_cut:.2f} for {len(radix)} pairs)")
    picks = next_picks(h, a.seeds[0])
    # what the best of this many unrelated strategies scores by chance alone
    exp_best = float(np.mean(np.max(np.random.default_rng(0).standard_normal((20000, len(radix))), axis=1)))
    pl = planted()
    print(f"planted bias check (hot-weighted z): {pl['hot_weighted_z']}")
    with open(a.out, "w") as fh:
        json.dump({"rounds": len(df), "split": a.split_date, "test_rounds": len(test_idx), "seeds": a.seeds,
                   "tickets_per_round": a.tickets, "groups": GROUPS, "strategies": STRATEGIES,
                   "bases": [base_label(b) for b in BASES], "results": bt, "pairs": len(radix), "z_cut": z_cut,
                   "best": best, "expected_best": exp_best, "planted": pl, "backtest": {"picks": picks}}, fh, indent=1)
    print(f"wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
