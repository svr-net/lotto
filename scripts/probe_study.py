"""New probes for a predictive edge in UK Lotto (59-ball era, Oct 2015 on).

Part A tests the physical routes to a bias against simulated fair draws (each round's six numbers
drawn uniformly, the rounds' ball sets and machines as they were):
  - ball sets: do the numbers' frequencies differ between ball sets (a number x set contingency
    chi-square), and is each set fair on its own (smallest p, Bonferroni over the sets)?
  - machines: the same for the drawing machines.
  - pairs: do some pairs of numbers come up together more often than chance (the dispersion of the
    1,711 pair counts)?
  - nights with two rounds: do the two rounds share more numbers than chance?

Part B backtests new strategies walk-forward: for every round from --split on, each strategy sees
only the draws on earlier dates and plays --tickets lines; each line's matches with the round's six
numbers are counted. A strategy beats chance only if its mean is above 36/59 (0.610) by more than
the Bonferroni cut over all strategies (z_cut); expected_best is the best z expected from that many
strategies with no edge.

    python scripts/probe_study.py --out results/probe_study.json
"""
import argparse, json, math, time
from math import comb

import numpy as np
import pandas as pd

POOL, PICK = 59, 6
N_COLS = [f"n{i}" for i in range(1, 7)]
CHANCE = PICK * PICK / POOL
VAR1 = PICK * (PICK / POOL) * ((POOL - PICK) / POOL) * ((POOL - PICK) / (POOL - 1))  # one line's variance
P3 = sum(comb(PICK, k) * comb(POOL - PICK, PICK - k) for k in range(3, 7)) / comb(POOL, PICK)


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


def fair_sets(n, rng):
    return np.argsort(rng.random((n, POOL)), axis=1)[:, :PICK] + 1


# ---------------------------------------------------------------- Part A: bias tests
def chi_groups(x, groups):
    """Chi-square of the number x group table (each group's counts against its own total)."""
    stat = 0.0
    for g in np.unique(groups):
        c = x[groups == g].sum(0)
        e = c.sum() / POOL
        stat += ((c - e) ** 2 / e).sum()
    return stat


def chi_between(x, groups):
    """Heterogeneity between groups: the contingency chi-square of number x group."""
    tab = np.array([x[groups == g].sum(0) for g in np.unique(groups)])
    e = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    return float(((tab - e) ** 2 / np.where(e > 0, e, 1)).sum())


def pair_dispersion(sets):
    c = np.zeros((POOL, POOL))
    for s in sets - 1:
        c[np.ix_(s, s)] += 1
    iu = np.triu_indices(POOL, 1)
    return float(c[iu].var())


def bias_tests(e, sims, rng):
    sets = e[N_COLS].to_numpy(int)
    x = onehot(sets)
    out = []
    null = [fair_sets(len(sets), rng) for _ in range(sims)]
    nullx = [onehot(s) for s in null]
    for name, col in (("ball set", "ball_set"), ("machine", "machine")):
        g = e[col].astype(str).str.strip().to_numpy()
        keep = pd.Series(g).map(pd.Series(g).value_counts()) >= 30  # groups with 30+ rounds
        gk, xk = g[keep.to_numpy()], x[keep.to_numpy()]
        obs = chi_between(xk, gk)
        nb = np.array([chi_between(nx[keep.to_numpy()], gk) for nx in nullx])
        out.append({"test": f"numbers differ between {name}s ({len(np.unique(gk))} {name}s, {keep.sum()} rounds)", "stat": obs,
                    "p": float((1 + (nb >= obs).sum()) / (1 + sims))})
        # Each group on its own: its chi-square of uniformity, p by simulation; the smallest p.
        ps = []
        for v in np.unique(gk):
            m = gk == v
            o = chi_groups(xk[m], gk[m])
            nn = np.array([chi_groups(nx[keep.to_numpy()][m], gk[m]) for nx in nullx])
            ps.append((float((1 + (nn >= o).sum()) / (1 + sims)), v, int(m.sum())))
        pmin, v, n = min(ps)
        out.append({"test": f"the least fair {name} on its own ({name} {v}, {n} rounds)", "stat": pmin,
                    "p": min(1.0, pmin * len(ps)), "note": f"Bonferroni over {len(ps)} {name}s"})
    obs = pair_dispersion(sets)
    nb = np.array([pair_dispersion(s) for s in null[: min(sims, 300)]])
    out.append({"test": "some pairs come up together more often than chance (dispersion of 1,711 pair counts)", "stat": obs,
                "p": float((1 + (nb >= obs).sum()) / (1 + len(nb)))})
    two = e.groupby("date").filter(lambda g: len(g) == 2).sort_values(["date", "round"])
    if len(two):
        a, b = two[N_COLS].to_numpy(int)[0::2], two[N_COLS].to_numpy(int)[1::2]
        ov = np.array([len(set(p) & set(q)) for p, q in zip(a, b)])
        z = (ov.mean() - CHANCE) / math.sqrt(VAR1 / len(ov))
        out.append({"test": f"the two rounds of a night share numbers ({len(ov)} nights)", "stat": float(ov.mean()), "p": 2 * norm_sf(abs(z)),
                    "note": f"mean overlap {ov.mean():.3f} vs {CHANCE:.3f}, z {z:+.2f}"})
    return out


# ---------------------------------------------------------------- Part B: strategies
def weighted_lines(w, n, rng):
    w = np.clip(np.asarray(w, float), 1e-12, None)
    w = w / w.sum()
    g = -np.log(rng.random((n, POOL))) / w
    return np.sort(np.argsort(g, axis=1)[:, :PICK] + 1, axis=1)


def top_lines(score, n, rng):
    """The six highest, then lines drawn around them (each line the top six of the score plus noise)."""
    s = np.asarray(score, float)
    sd = s.std() + 1e-12
    out = [np.sort(np.argsort(-s)[:PICK] + 1)]
    while len(out) < n:
        out.append(np.sort(np.argsort(-(s + rng.normal(0, 0.5 * sd, POOL)))[:PICK] + 1))
    return np.array(out)


class Ctx:
    """What a strategy may see: the earlier rounds (sets, ball sets, machines) and a memory."""

    def __init__(self, sets, ball, mach, mem):
        self.sets, self.ball, self.mach = sets, ball, mach
        self.mem = mem  # kept across rounds (a strategy's fitted model)


def s_random(c, n, rng, target):
    return fair_sets(n, rng)


def s_shrunk(c, n, rng, target):
    """Frequencies shrunk to uniform, the strength learnt by the Dirichlet-multinomial evidence."""
    cnt = onehot(c.sets).sum(0)
    best, ba = -1e300, 1.0
    for a in (1, 3, 10, 30, 100, 300, 1000, 3000):  # the evidence of a symmetric Dirichlet(a) prior
        ll = math.lgamma(POOL * a) - math.lgamma(POOL * a + cnt.sum()) + sum(math.lgamma(a + k) - math.lgamma(a) for k in cnt)
        if ll > best:
            best, ba = ll, a
    return top_lines(cnt + ba, n, rng)


def s_decay(c, n, rng, target):
    """Exponentially weighted frequencies; the half-life chosen by the trailing log-likelihood."""
    x = onehot(c.sets)
    best, bw = -1e300, None
    for h in (25, 50, 100, 200, 400, 1e9):
        lam = 0.5 ** (1 / h)
        w = np.ones(POOL)
        ll = 0.0
        for t in range(max(0, len(x) - 300), len(x)):  # score on the last 300 rounds
            p = w / w.sum()
            ll += np.log(p[x[t] > 0]).sum()
            w = lam * w + x[t]
        if ll > best:
            best, bw = ll, w
    return top_lines(bw, n, rng)


def set_freq(c, key, value):
    m = c.ball == value if key == "ball" else c.mach == value
    return onehot(c.sets[m]).sum(0) + 1


def s_ballset_mix(c, n, rng, target):
    """Each ball set's frequencies, mixed by how often each set was used in the last 100 rounds."""
    recent = pd.Series(c.ball[-100:]).value_counts(normalize=True)
    w = sum(p * (f := set_freq(c, "ball", v)) / f.sum() for v, p in recent.items())
    return weighted_lines(w, n, rng)


def s_ballset_oracle(c, n, rng, target):
    """ORACLE (not playable): the frequencies of the ball set the round actually used."""
    f = set_freq(c, "ball", target["ball"])
    return top_lines(f, n, rng)


def s_machine_mix(c, n, rng, target):
    recent = pd.Series(c.mach[-100:]).value_counts(normalize=True)
    w = sum(p * (f := set_freq(c, "mach", v)) / f.sum() for v, p in recent.items())
    return weighted_lines(w, n, rng)


def s_pairs(c, n, rng, target):
    """Lines of numbers that came up together most often (greedy, from a random start each line)."""
    co = np.zeros((POOL, POOL))
    for s in c.sets - 1:
        co[np.ix_(s, s)] += 1
    np.fill_diagonal(co, 0)
    out = []
    for _ in range(n):
        line = [int(rng.integers(POOL))]
        while len(line) < PICK:
            sc = co[line].sum(0) + rng.random(POOL) * 1e-3
            sc[line] = -1
            line.append(int(np.argmax(sc)))
        out.append(np.sort(np.array(line) + 1))
    return np.array(out)


def s_knn(c, n, rng, target):
    """Analogues: the rounds after the 50 earlier rounds most like the last one (shared numbers)."""
    x = onehot(c.sets)
    sim = x[:-1] @ x[-1]
    idx = np.argsort(-sim + rng.random(len(sim)) * 1e-3)[:50]
    return top_lines(x[idx + 1].sum(0) + 1, n, rng)


def s_thompson(c, n, rng, target):
    """Thompson sampling: each line from a draw of the numbers' Dirichlet posterior."""
    a = onehot(c.sets).sum(0) + 1
    return np.array([np.sort(np.argsort(-rng.dirichlet(a))[:PICK] + 1) for _ in range(n)])


def s_boosted(c, n, rng, target):
    """Gradient-boosted trees: each number's chance next round from its recent history (refitted
    every 52 rounds on every earlier round)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    x = onehot(c.sets)

    def feats(t):  # features of every number at the end of round t-1, for round t
        f = []
        for w in (5, 10, 25, 50, 100, 200):
            f.append(x[max(0, t - w):t].mean(0) if t > 0 else np.zeros(POOL))
        last = np.full(POOL, 200.0)
        seen = np.where(x[max(0, t - 200):t][::-1].any(0))[0]
        for j in seen:
            last[j] = np.argmax(x[max(0, t - 200):t][::-1][:, j] > 0)
        f.append(last / 200)
        return np.stack(f, 1)

    T = len(x)
    if "hgb" not in c.mem or T - c.mem["hgb_t"] >= 52:
        rows = range(max(50, T - 600), T)
        X = np.concatenate([feats(t) for t in rows])
        y = np.concatenate([x[t] for t in rows])
        c.mem["hgb"] = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(X, y)
        c.mem["hgb_t"] = T
    p = c.mem["hgb"].predict_proba(feats(T))[:, 1]
    return top_lines(p, n, rng)


STRATEGIES = {
    "random (baseline)": s_random,
    "shrunk frequencies (learnt Dirichlet)": s_shrunk,
    "decayed frequencies (learnt half-life)": s_decay,
    "ball-set mixture": s_ballset_mix,
    "machine mixture": s_machine_mix,
    "pair cohesion": s_pairs,
    "nearest-neighbour analogues": s_knn,
    "Thompson sampling": s_thompson,
    "gradient-boosted trees": s_boosted,
}
ORACLES = {"ORACLE: the round's own ball set (not playable)": s_ballset_oracle}


def backtest(e, split, tickets, seeds):
    e = e.sort_values(["draw_no", "round"]).reset_index(drop=True)
    sets = e[N_COLS].to_numpy(int)
    ball = e.ball_set.astype(str).str.strip().to_numpy()
    mach = e.machine.astype(str).str.strip().to_numpy()
    dates = e.date.to_numpy()
    test = np.where(dates > split)[0]
    res = {}
    for name, fn in {**STRATEGIES, **ORACLES}.items():
        t0 = time.time()
        per_seed = []
        for seed in range(seeds):
            rng = np.random.default_rng(1000 + seed)
            ctxs, mem = {}, {}
            means, hits3 = [], 0
            for r in test:
                h = np.where(dates < dates[r])[0]
                key = len(h)
                if key not in ctxs:
                    ctxs = {key: Ctx(sets[h], ball[h], mach[h], mem)}
                c = ctxs[key]
                lines = fn(c, tickets, rng, {"ball": ball[r], "mach": mach[r]})
                m = np.array([len(set(l) & set(sets[r])) for l in lines])
                means.append(m.mean())
                hits3 += int((m >= 3).sum())
            means = np.array(means)
            z = (means.mean() - CHANCE) / (means.std(ddof=1) / math.sqrt(len(means)))
            per_seed.append({"mean": float(means.mean()), "z": float(z), "hits3": hits3})
            if fn is s_boosted:
                break  # deterministic given the data (refits are seeded), one seed is enough
        zs = [p["z"] for p in per_seed]
        res[name] = {"mean": float(np.mean([p["mean"] for p in per_seed])), "z_mean": float(np.mean(zs)), "z_min": float(min(zs)),
                     "z_max": float(max(zs)), "hits3": float(np.mean([p["hits3"] for p in per_seed])),
                     "hits3_expected": len(test) * tickets * P3, "seeds": len(per_seed), "seconds": round(time.time() - t0)}
        print(f"  {name:48s} mean {res[name]['mean']:.4f} z {res[name]['z_mean']:+.2f} ({res[name]['z_min']:+.2f}..{res[name]['z_max']:+.2f}) "
              f"3+ {res[name]['hits3']:.0f} vs {res[name]['hits3_expected']:.0f} ({res[name]['seconds']}s)", flush=True)
    return res, len(test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/uk_lotto_full.csv")
    ap.add_argument("--split", default="2022-12-31")
    ap.add_argument("--tickets", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--out", default="results/probe_study.json")
    a = ap.parse_args()
    e = pd.read_csv(a.csv)
    e = e[e.draw_no >= 2066].copy()
    rng = np.random.default_rng(2026)
    print("Part A: bias tests", flush=True)
    tests = bias_tests(e, a.sims, rng)
    for t in tests:
        print(f"  p {t['p']:.4f}  {t['test']}" + (f"  [{t['note']}]" if t.get("note") else ""))
    print(f"Part B: strategies, {a.tickets} lines a round, {a.seeds} seeds, test from {a.split}", flush=True)
    res, n_test = backtest(e, a.split, a.tickets, a.seeds)
    S = len(STRATEGIES) - 1  # the baseline aside
    z_cut = norm_isf(0.05 / S)
    expected_best = norm_isf(1 / (S + 1))  # roughly the best of S null z-scores
    best = max((k for k in STRATEGIES if k != "random (baseline)"), key=lambda k: res[k]["z_mean"])
    print(f"best: {best} z {res[best]['z_mean']:+.2f}; z_cut {z_cut:.2f} (Bonferroni over {S}); expected best with no edge ~{expected_best:.2f}")
    alpha_t = 0.05 / len(tests)
    with open(a.out, "w") as fh:
        json.dump({"split": a.split, "test_rounds": n_test, "tickets": a.tickets, "seeds": a.seeds, "chance_mean": CHANCE,
                   "tests": tests, "test_alpha": alpha_t, "strategies": res, "z_cut": z_cut, "expected_best": expected_best, "best": best}, fh, indent=1)


if __name__ == "__main__":
    main()
