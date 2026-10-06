"""Conditional probabilities and copulas on the prime-radix encoding of UK Lotto draws.

Usage:
    python scripts/prime_radix_study.py --csv data/uk_lotto_full.csv --out results/prime_radix.json

Every ball is written in each of the first 10 prime bases (2, 3, 5, ..., 29) as a
fixed-width digit string, as in lotto_nn.prime_pipeline (59 in base 7 is 1-1-3).
Only the 1-59 era (draw 2066, 2015-10-10, onward) is used.

Part 1 tests the digits for structure, per base, against simulated fair histories
(drawing without replacement makes digits slightly dependent even in a fair
machine, so the fair null is simulated rather than assumed):
  - within-draw conditional: P(digit of the next ball drawn | digit of this ball)
  - across-draw conditional: P(a digit value appears c times | it appeared c' times
    in the previous round)
  - temporal copula: rank (Spearman) correlation between one round's digit totals
    and the next round's; rank correlation depends only on the copula
  - within-draw copula: rank correlation between the last digits (residues) of
    the six balls in drawn order

Part 2 replays strategies built on those ideas through the walk-forward backtest
(each round predicted only from earlier rounds, scored against 36/59 chance):
  - conditional probability (base p): numbers are weighted by the product, over
    their digits, of how often that digit value came up after rounds in which it
    appeared as often as it did in the last round (smoothed lift)
  - conditional copula (base p): a Gaussian copula fitted to pairs of consecutive
    rounds' digit vectors; the next round's digits are sampled conditional on the
    last round's and decoded back to numbers
  - the repo prime-radix transformer per base (trained once on draws up to the
    split, scored on the later draws), for reference
"""
from __future__ import annotations

import argparse, json, math, os, sys, time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lotto_nn.prime_pipeline import FIRST_10_PRIMES, digit_width  # noqa: E402

POOL, PICK = 59, 6
DRAWN = ["d1", "d2", "d3", "d4", "d5", "d6"]
SETS = ["n1", "n2", "n3", "n4", "n5", "n6"]
EXP_MEAN = PICK * PICK / POOL
EXP_VAR = PICK * (PICK / POOL) * (1 - PICK / POOL) * (POOL - PICK) / (POOL - 1)


# ---------------------------------------------------------------- digits
def digits(values: np.ndarray, base: int) -> np.ndarray:
    """values [...] -> [..., width] digits, least significant first (j=0 is the residue mod base)."""
    w = digit_width(base)
    return np.stack([(values // base ** j) % base for j in range(w)], axis=-1)


DIGITS = {p: digits(np.arange(POOL + 1), p) for p in FIRST_10_PRIMES}  # row x -> digits of x


# ---------------------------------------------------------------- normal helpers (no scipy)
def norm_cdf(z):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(z) / math.sqrt(2)))


def norm_ppf(p):
    """Acklam's rational approximation to the inverse normal CDF (|error| < 1.2e-9)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    out = np.empty_like(p)
    lo, hi = p < 0.02425, p > 1 - 0.02425
    mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(p[lo]))
    out[lo] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p[mid] - 0.5
    r = q * q
    out[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = np.sqrt(-2 * np.log(1 - p[hi]))
    out[hi] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    return out


def rank_normal(x: np.ndarray, rng) -> np.ndarray:
    """Column-wise normal scores of (discrete) data, ties broken at random."""
    n = len(x)
    jitter = rng.random(x.shape)
    if x.ndim == 1:
        order = np.lexsort((jitter, x))
        r = np.empty(n)
        r[order] = np.arange(1, n + 1)
        return norm_ppf((r - 0.5) / n)
    return np.column_stack([rank_normal(x[:, j], rng) for j in range(x.shape[1])])


def spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman correlation matrix between columns of a and columns of b (average ranks)."""
    def ranks(m):
        return pd.DataFrame(m).rank().to_numpy()
    ra, rb = ranks(a), ranks(b)
    ra = (ra - ra.mean(0)) / (ra.std(0) + 1e-12)
    rb = (rb - rb.mean(0)) / (rb.std(0) + 1e-12)
    return ra.T @ rb / len(a)


# ---------------------------------------------------------------- part 1: structure tests
def chi2_indep(table: np.ndarray) -> float:
    table = table[table.sum(1) > 0][:, table.sum(0) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    exp = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / table.sum()
    return float(((table - exp) ** 2 / exp).sum())


def count_profile(drawn: np.ndarray, base: int) -> np.ndarray:
    """[rounds, width, base]: how many of the round's balls have digit value v at position j."""
    d = DIGITS[base][drawn]  # [rounds, 6, width]
    w = d.shape[-1]
    out = np.zeros((len(drawn), w, base), dtype=int)
    for j in range(w):
        for k in range(PICK):
            np.add.at(out[:, j], (np.arange(len(drawn)), d[:, k, j]), 1)
    return out


def stats_for(drawn: np.ndarray, base: int) -> dict:
    d = DIGITS[base][drawn]  # [rounds, 6, width]
    w = d.shape[-1]
    # within-draw conditional: digit of ball k+1 given digit of ball k, per digit position
    within = 0.0
    for j in range(w):
        t = np.zeros((base, base))
        np.add.at(t, (d[:, :-1, j].ravel(), d[:, 1:, j].ravel()), 1)
        within += chi2_indep(t)
    # across-draw conditional: count of a digit value this round given its count last round (0/1/2+)
    prof = np.minimum(count_profile(drawn, base), 2)
    across = 0.0
    for j in range(w):
        for v in range(base):
            t = np.zeros((3, 3))
            np.add.at(t, (prof[:-1, j, v], prof[1:, j, v]), 1)
            across += chi2_indep(t)
    # temporal copula: Spearman between consecutive rounds' digit totals per position
    tot = d.sum(1)  # [rounds, width]
    r = spearman(tot[:-1], tot[1:])
    temporal = float((r ** 2).sum() * (len(tot) - 1))
    # within-draw copula: Spearman among the six balls' residues in drawn order
    rr = spearman(d[:, :, 0], d[:, :, 0])
    iu = np.triu_indices(PICK, 1)
    within_cop = float((rr[iu] ** 2).sum() * len(d))
    return {"within_cond": within, "across_cond": across, "temporal_copula": temporal,
            "within_copula": within_cop, "temporal_rho_max": float(np.abs(r).max())}


TEST_LABELS = {
    "within_cond": "next ball's digit depends on this ball's digit",
    "across_cond": "a digit's count depends on its count last round",
    "temporal_copula": "consecutive rounds' digit totals are rank-correlated",
    "within_copula": "the six balls' residues are rank-correlated",
}


def structure_tests(drawn: np.ndarray, sims: int, rng) -> dict:
    obs = {p: stats_for(drawn, p) for p in FIRST_10_PRIMES}
    null = {p: {k: [] for k in TEST_LABELS} for p in FIRST_10_PRIMES}
    for s in range(sims):
        fake = np.argsort(rng.random((len(drawn), POOL)), axis=1)[:, :PICK] + 1
        for p in FIRST_10_PRIMES:
            st = stats_for(fake, p)
            for k in TEST_LABELS:
                null[p][k].append(st[k])
    rows = []
    for p in FIRST_10_PRIMES:
        row = {"base": p, "width": digit_width(p), "temporal_rho_max": obs[p]["temporal_rho_max"]}
        for k in TEST_LABELS:
            arr = np.array(null[p][k])
            row[k] = obs[p][k]
            row[k + "_fair"] = float(arr.mean())
            row[k + "_p"] = float(((arr >= obs[p][k]).sum() + 1) / (sims + 1))
        rows.append(row)
        print(f"  base {p:2d}: " + "  ".join(f"{k} p={row[k + '_p']:.3f}" for k in TEST_LABELS))
    return {"rows": rows, "labels": TEST_LABELS, "sims": sims, "n_tests": len(rows) * len(TEST_LABELS)}


# ---------------------------------------------------------------- part 2: strategies
def weighted_tickets(weights: np.ndarray, n: int, rng) -> np.ndarray:
    p = weights / weights.sum()
    return np.array([np.sort(rng.choice(np.arange(1, POOL + 1), PICK, replace=False, p=p)) for _ in range(n)])


def cond_prob_tickets(hist: np.ndarray, base: int, n: int, rng, smooth: float = 30.0) -> np.ndarray:
    """Lift of each digit value given how often it appeared last round, multiplied over digits."""
    prof = np.minimum(count_profile(hist, base), 2)  # [rounds, width, base]
    w = prof.shape[1]
    allx = DIGITS[base][1:]  # [59, width]
    # fair expected count of digit v at position j per round
    fair = np.stack([np.bincount(allx[:, j], minlength=base) * PICK / POOL for j in range(w)])
    full = count_profile(hist, base)
    weight = np.ones(POOL)
    last = prof[-1]
    for j in range(w):
        for v in range(base):
            if fair[j, v] == 0:  # digit value that no number in 1..59 has at this position
                continue
            c = last[j, v]
            rows = np.flatnonzero(prof[:-1, j, v] == c) + 1  # rounds following a round with count c
            obs = full[rows, j, v].sum()
            exp = fair[j, v] * len(rows)
            lift = (obs + smooth * fair[j, v]) / (exp + smooth * fair[j, v])
            weight[allx[:, j] == v] *= lift
    return weighted_tickets(weight, n, rng)


def copula_tickets(hist: np.ndarray, base: int, n: int, rng, shrink: float = 0.2) -> np.ndarray:
    """Gaussian copula over (last round, next round) digit vectors of the sorted draw;
    sample the next round's digits given the last round's, decode, keep valid draws."""
    x = DIGITS[base][np.sort(hist, axis=1)].reshape(len(hist), -1).astype(float)  # [rounds, 6*width]
    d = x.shape[1]
    z = rank_normal(x, rng)
    pairs = np.hstack([z[:-1], z[1:]])
    S = np.corrcoef(pairs, rowvar=False)
    S = (1 - shrink) * S + shrink * np.eye(2 * d)
    A, B, C = S[:d, :d], S[:d, d:], S[d:, d:]
    K = np.linalg.solve(A, B).T  # B^T A^-1
    mu = K @ z[-1]
    cov = C - K @ B
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(d))
    sorted_cols = np.sort(x, axis=0)
    width = digit_width(base)
    place = base ** np.arange(width)
    out = []
    tries = 0
    while len(out) < n and tries < 50:
        tries += 1
        u = norm_cdf(mu + rng.standard_normal((400, d)) @ L.T)
        idx = np.minimum((u * len(x)).astype(int), len(x) - 1)
        dig = np.take_along_axis(sorted_cols, idx, axis=0).astype(int).reshape(400, PICK, width)
        nums = (dig * place).sum(-1)
        ok = ((nums >= 1) & (nums <= POOL)).all(1) & (np.sort(nums, 1)[:, 1:] != np.sort(nums, 1)[:, :-1]).all(1)
        out.extend(np.sort(nums[ok], 1)[: n - len(out)])
    while len(out) < n:  # fall back to a random ticket if the copula can't produce enough valid ones
        out.append(np.sort(rng.choice(np.arange(1, POOL + 1), PICK, replace=False)))
    return np.array(out[:n])


def summarize(per_round: list) -> dict:
    m = np.array([x for r in per_round for x in r])
    n = len(m)
    z = (m.mean() - EXP_MEAN) / math.sqrt(EXP_VAR / n)
    return {"tickets": n, "mean_matches": float(m.mean()), "z": float(z), "pct_3plus": float((m >= 3).mean() * 100)}


def backtest(sets: np.ndarray, dates: np.ndarray, split, seeds: list, tickets: int) -> dict:
    test_idx = [i for i, d in enumerate(dates) if d > split]
    names = ["random (baseline)"] + [f"conditional probability, base {p}" for p in FIRST_10_PRIMES] \
        + [f"conditional copula, base {p}" for p in FIRST_10_PRIMES]
    res = {s: [] for s in names}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        per = {s: [] for s in names}
        t0 = time.time()
        for i in test_idx:
            hist = sets[dates < dates[i]]
            actual = set(sets[i].tolist())
            tk = {"random (baseline)": weighted_tickets(np.ones(POOL), tickets, rng)}
            for p in FIRST_10_PRIMES:
                tk[f"conditional probability, base {p}"] = cond_prob_tickets(hist, p, tickets, rng)
                tk[f"conditional copula, base {p}"] = copula_tickets(hist, p, tickets, rng)
            for s, ts in tk.items():
                per[s].append([len(actual & set(t.tolist())) for t in ts])
        for s in names:
            res[s].append(summarize(per[s]))
        print(f"  seed {seed} ({time.time() - t0:.0f}s): " + ", ".join(f"{s.split(', ')[-1] if ',' in s else s} {res[s][-1]['z']:+.2f}" for s in names[:3]) + " ...")
    out = {}
    for s in names:
        zs = [r["z"] for r in res[s]]
        out[s] = {"z_mean": float(np.mean(zs)), "z_min": float(np.min(zs)), "z_max": float(np.max(zs)),
                  "mean_matches": float(np.mean([r["mean_matches"] for r in res[s]])),
                  "pct_3plus": float(np.mean([r["pct_3plus"] for r in res[s]])), "tickets": res[s][0]["tickets"]}
        print(f"    {s:36s} z {out[s]['z_mean']:+.2f} [{out[s]['z_min']:+.2f}, {out[s]['z_max']:+.2f}]")
    return {"strategies": out, "test_rounds": len(test_idx), "seeds": seeds, "tickets_per_round": tickets}


def prime_nn_reference(csv: str, split: str, epochs: int, samples: int) -> list:
    """The repo prime-radix transformer per base, trained up to the split and scored on later draws."""
    from lotto_nn.prime_pipeline import run_prime_pipeline
    import torch
    torch.manual_seed(0)
    res = run_prime_pipeline(csv, epochs=epochs, num_samples=samples, verbose=False, split_date=split)
    rows = []
    for p, r in res.items():
        h = r["holdout"]
        rows.append({"base": p, "mean": h["mean"], "z": h["z"], "test_draws": h["test_draws"], "final_loss": r["final_loss"]})
        print(f"    prime-radix transformer base {p:2d}: holdout mean {h['mean']:.4f} z={h['z']:+.2f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/uk_lotto_full.csv")
    ap.add_argument("--split-date", default="2022-12-31")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 1, 2])
    ap.add_argument("--seed", type=int, default=7, help="seed for the simulated fair histories")
    ap.add_argument("--tickets", type=int, default=20)
    ap.add_argument("--nn-epochs", type=int, default=30)
    ap.add_argument("--nn-samples", type=int, default=300)
    ap.add_argument("--out", default="results/prime_radix.json")
    a = ap.parse_args()
    t0 = time.time()
    df = pd.read_csv(a.csv)
    df = df[df.draw_no >= 2066].sort_values(["draw_no", "round"]).reset_index(drop=True)
    drawn = df[DRAWN].to_numpy(int)
    sets = df[SETS].to_numpy(int)
    dates = pd.to_datetime(df.date).dt.date.to_numpy()
    split = pd.to_datetime(a.split_date).date()
    print(f"1-59 era: {len(df)} rounds; structure tests against {a.sims} fair histories")
    tests = structure_tests(drawn, a.sims, np.random.default_rng(a.seed))
    print(f"walk-forward backtest over rounds after {split}")
    bt = backtest(sets, dates, split, a.seeds, a.tickets)
    print("repo prime-radix transformers (trained up to the split):")
    tmp = os.path.join(os.path.dirname(a.out) or ".", ".prime_radix_59.csv")
    df.to_csv(tmp, index=False)
    try:
        nn = prime_nn_reference(tmp, a.split_date, a.nn_epochs, a.nn_samples)
    finally:
        os.remove(tmp)
    with open(a.out, "w") as fh:
        json.dump({"rounds": len(df), "split": a.split_date, "tests": tests, "backtest": bt, "prime_nn": nn},
                  fh, indent=1, default=float)
    print(f"wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
