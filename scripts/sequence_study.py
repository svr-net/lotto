"""The draws as one continuous sequence: does time help catch 4 of the 6 drawn numbers in 15 chosen ones?

For any 15 numbers, the chance that 4 or more of a round's six are among them is 3.17% if the draws are
fair and independent. Time could only help if the sequence carries information, so this study asks four
things of the whole history (both eras, every round in order):

A. Memory. Does a number's chance of being drawn depend on how many rounds it has been away (the hazard:
   "due" numbers would rise with the gap, "hot" ones fall), or on whether it came up in the previous round
   or the round before? Each is a chi-square of drawn against expected over the gap buckets, with its
   p-value from fair histories of the same length (simulated, so the six-without-replacement structure is
   kept).

B. Sequence-aware choice of the 15. Walk-forward over every night from --split, each rule sees only the
   rounds before and picks 15 numbers for both of the night's rounds:
     - hazard: the 15 whose current gap has had the highest drawn rate in the history so far;
     - transitions: the 15 most often drawn in the round after rounds that held the last round's numbers;
     - momentum: the 15 whose rate over the last 26 rounds most exceeds their rate over the last 208;
     - due: the 15 away longest relative to their own typical gap;
     - random, for comparison.
   What counts is how often 4 or more of a round's six fall among the 15 (3.17% by chance), with z and the
   Bonferroni cut over the rules.

C. Timing. For each rule's sequence of hits (rounds with 4+ among its 15), is a hit likelier right after a
   hit, or after a long drought (more than 32 rounds, about the average wait), than otherwise? Playing
   only in such rounds would help only if it is.

D. Over time. The chance of at least one hit within 26, 52, 104 and 520 rounds, 1 - (1 - p)^n, against how
   often the real sequence delivered one in every window of that length; and the real waits between hits
   against the geometric waits of independent rounds (mean 31.5).

    python scripts/sequence_study.py --out results/sequence_study.json
"""
import argparse, json, math
from math import comb

import numpy as np
import pandas as pd

PICK, V = 6, 15
N_COLS = [f"n{i}" for i in range(1, 7)]
GAP_BINS = [0, 1, 2, 3, 5, 8, 12, 18, 27, 40, 60, 10 ** 9]  # rounds since last drawn (0 = in the last round)


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def norm_isf(p):
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if norm_sf(mid) > p else (lo, mid)
    return (lo + hi) / 2


def p_hit(pool):
    return sum(comb(V, k) * comb(pool - V, PICK - k) for k in range(4, PICK + 1)) / comb(pool, PICK)


def onehot(sets, pool):
    x = np.zeros((len(sets), pool), bool)
    np.put_along_axis(x, sets - 1, True, axis=1)
    return x


def gaps_before(x):
    """g[t, n]: rounds since number n was last drawn, before round t (len(history) if never)."""
    T, pool = x.shape
    g = np.zeros((T, pool), int)
    last = np.full(pool, -1)
    for t in range(T):
        g[t] = np.where(last >= 0, t - 1 - last, t)
        last[x[t]] = t
    return g


# ---------------------------------------------------------------- A: memory
def hazard_stat(x, warmup=60):
    """Chi-square of drawn vs expected per gap bucket, and the drawn rate in each bucket."""
    g = gaps_before(x)[warmup:]
    y = x[warmup:]
    rate = PICK / x.shape[1]
    b = np.digitize(g, GAP_BINS[1:-1])
    obs = np.bincount(b[y], minlength=len(GAP_BINS) - 1).astype(float)
    n = np.bincount(b.ravel(), minlength=len(GAP_BINS) - 1).astype(float)
    exp = n * rate
    ok = exp > 0
    return float(((obs[ok] - exp[ok]) ** 2 / exp[ok]).sum()), obs, n


def lag_stat(x, lag):
    """Drawn again after `lag` rounds: observed overlap against expected."""
    rep = (x[lag:] & x[:-lag]).sum()
    exp = x[lag:].sum() * PICK / x.shape[1]
    return float(rep), float(exp)


def memory(sets, pool, sims, rng):
    x = onehot(sets, pool)
    chi, obs, n = hazard_stat(x)
    fair = lambda: onehot(np.argsort(rng.random((len(sets), pool)), axis=1)[:, :PICK] + 1, pool)
    null_chi, null_lag = [], {1: [], 2: []}
    for _ in range(sims):
        f = fair()
        null_chi.append(hazard_stat(f)[0])
        for lag in null_lag:
            null_lag[lag].append(lag_stat(f, lag)[0])
    out = {"rounds": len(sets), "hazard": {"chi2": chi, "p": float((np.sum(np.array(null_chi) >= chi) + 1) / (sims + 1)),
                                         "bins": GAP_BINS[:-1], "drawn": obs.tolist(), "chances": n.tolist(),
                                         "rate": (obs / np.maximum(n, 1)).tolist(), "fair_rate": PICK / pool}}
    for lag, null in null_lag.items():
        o, e = lag_stat(x, lag)
        null = np.array(null)
        out[f"lag{lag}"] = {"observed": o, "expected": e, "p": float(2 * min(np.mean(null >= o), np.mean(null <= o)) + 1 / (sims + 1))}
    return out


# ---------------------------------------------------------------- B: sequence-aware choice of 15
def pick(score, rng):
    return np.argsort(-(score + rng.random(len(score)) * 1e-9))[:V]


def r_random(h, g, rng, pool):
    return rng.choice(pool, V, replace=False)


def r_hazard(h, g, rng, pool):
    """Each number's current gap, scored by the drawn rate of that gap bucket in the history so far."""
    if len(h) < 120:
        return r_random(h, g, rng, pool)
    gg = gaps_before(h)[60:]
    b = np.digitize(gg, GAP_BINS[1:-1])
    obs = np.bincount(b[h[60:]], minlength=len(GAP_BINS) - 1)
    n = np.bincount(b.ravel(), minlength=len(GAP_BINS) - 1)
    rate = (obs + 1) / (n + pool / PICK)
    return pick(rate[np.digitize(g, GAP_BINS[1:-1])], rng)


def r_transitions(h, g, rng, pool):
    """Numbers most often drawn in the round after a round holding each of the last round's numbers."""
    if len(h) < 30:
        return r_random(h, g, rng, pool)
    trans = h[:-1].T.astype(float) @ h[1:].astype(float)  # [from, to]
    return pick(trans[h[-1]].sum(0), rng)


def r_momentum(h, g, rng, pool):
    if len(h) < 208:
        return r_random(h, g, rng, pool)
    return pick(h[-26:].mean(0) - h[-208:].mean(0), rng)


def r_due(h, g, rng, pool):
    """Away longest relative to the number's own typical gap (its mean gap so far)."""
    freq = h.sum(0) + 1
    typical = len(h) / freq
    return pick(g / typical, rng)


RULES = {"random": r_random, "hazard (gap since last drawn)": r_hazard, "transitions from the last round": r_transitions,
         "momentum (last 26 vs last 208)": r_momentum, "due (gap against own typical gap)": r_due}


def walk_forward(sets, dates, pool, split, seeds):
    x = onehot(sets, pool)
    g_all = gaps_before(x)
    nights = sorted(set(dates[dates > split]))
    out = {}
    for name, fn in RULES.items():
        runs = []
        for seed in range(seeds):
            rng = np.random.default_rng(4000 + seed)
            hits = []
            for d in nights:
                idx = np.where(dates == d)[0]
                t0 = idx[0]
                cand = np.zeros(pool, bool)
                cand[fn(x[:t0], g_all[t0], rng, pool)] = True
                hits += [int((x[t] & cand).sum() >= 4) for t in idx]
            runs.append(np.array(hits))
        p = p_hit(pool)
        zs = [(h.sum() - len(h) * p) / math.sqrt(len(h) * p * (1 - p)) for h in runs]
        out[name] = {"rounds": len(runs[0]), "hits_mean": float(np.mean([h.sum() for h in runs])), "expected": len(runs[0]) * p,
                     "z_mean": float(np.mean(zs)), "z_min": float(min(zs)), "z_max": float(max(zs)), "sequence": runs[0].tolist()}
        o = out[name]
        print(f"  {name:36s} hits {o['hits_mean']:5.1f} of {o['rounds']} (expected {o['expected']:.1f}), "
              f"z {o['z_mean']:+.2f} ({o['z_min']:+.2f}..{o['z_max']:+.2f})", flush=True)
    return out


# ---------------------------------------------------------------- C and D: timing and the sequence over time
def timing(seq, drought=32):
    s = np.asarray(seq)
    after_hit, after_long, other = [], [], []
    since = 10 ** 6
    for t in range(1, len(s)):
        since = 0 if s[t - 1] else since + 1
        (after_hit if since == 0 else after_long if since > drought else other).append(s[t])
    rate = lambda a: (float(np.mean(a)) if a else None, len(a))
    return {"after_hit": rate(after_hit), "after_drought": rate(after_long), "otherwise": rate(other)}


def over_time(seq, p):
    s = np.asarray(seq)
    out = {"windows": {}}
    for n in (26, 52, 104, 520):
        if len(s) >= n:
            c = np.convolve(s, np.ones(n, int), "valid")
            out["windows"][str(n)] = {"real": float(np.mean(c > 0)), "theory": 1 - (1 - p) ** n}
    hits = np.where(s)[0]
    waits = np.diff(hits)
    if len(waits):
        out["waits"] = {"n": int(len(waits)), "mean": float(waits.mean()), "median": float(np.median(waits)),
                        "max": int(waits.max()), "theory_mean": 1 / p, "theory_median": math.log(0.5) / math.log(1 - p)}
        # chi-square of waits against the geometric distribution, in quartile bins of the geometric
        edges = [0] + [math.log(1 - q) / math.log(1 - p) for q in (0.25, 0.5, 0.75)] + [1e9]
        obs = np.histogram(waits, bins=edges)[0]
        exp = np.full(4, len(waits) / 4)
        out["waits"]["chi2"] = float(((obs - exp) ** 2 / exp).sum())
        from scipy.stats import chi2
        out["waits"]["p"] = float(chi2.sf(out["waits"]["chi2"], 3))
        out["waits"]["quartiles"] = obs.tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/uk_lotto_full.csv")
    ap.add_argument("--split", default="2016-12-31")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--out", default="results/sequence_study.json")
    a = ap.parse_args()
    rng = np.random.default_rng(2026)
    df = pd.read_csv(a.csv).sort_values(["draw_no", "round"])
    eras = {"59-ball era (Oct 2015 - now)": (df[df.draw_no >= 2066], 59), "49-ball era (1994 - Oct 2015)": (df[df.draw_no < 2066], 49)}
    out = {"p_hit_59": p_hit(59), "p_hit_49": p_hit(49), "sims": a.sims, "memory": {}}
    print("A: memory", flush=True)
    for label, (e, pool) in eras.items():
        m = memory(e[N_COLS].to_numpy(int), pool, a.sims, rng)
        out["memory"][label] = m
        print(f"  {label}: hazard by gap p {m['hazard']['p']:.3f}; repeat after 1 round {m['lag1']['observed']:.0f} vs {m['lag1']['expected']:.0f} "
              f"(p {m['lag1']['p']:.3f}); after 2 {m['lag2']['observed']:.0f} vs {m['lag2']['expected']:.0f} (p {m['lag2']['p']:.3f})", flush=True)
    e59 = eras["59-ball era (Oct 2015 - now)"][0]
    print(f"B: choosing the 15 from the sequence, nights from {a.split}", flush=True)
    rules = walk_forward(e59[N_COLS].to_numpy(int), e59.date.to_numpy(), 59, a.split, a.seeds)
    S = len(rules) - 1
    out.update({"split": a.split, "rules": rules, "z_cut": norm_isf(0.05 / S),
                "expected_best": float(np.mean(np.random.default_rng(0).standard_normal((20000, S)).max(1)))})
    print("C, D: timing and the sequence over time", flush=True)
    out["timing"], out["over_time"] = {}, {}
    for name, r in rules.items():
        out["timing"][name] = timing(r["sequence"])
        out["over_time"][name] = over_time(r["sequence"], p_hit(59))
        t, w = out["timing"][name], out["over_time"][name].get("waits", {})
        fmt = lambda v: "-" if v[0] is None else f"{v[0]:.3f} ({v[1]})"
        print(f"  {name:36s} hit rate after a hit {fmt(t['after_hit'])}, after a drought {fmt(t['after_drought'])}, "
              f"otherwise {fmt(t['otherwise'])}; waits mean {w.get('mean', float('nan')):.1f} (geometric {w.get('theory_mean', float('nan')):.1f}), "
              f"p {w.get('p', float('nan')):.2f}", flush=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
