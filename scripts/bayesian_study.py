"""Bayesian inference on the UK Lotto draws.

Usage:
    python scripts/bayesian_study.py --csv data/uk_lotto_full.csv --out results/bayesian.json

Model. Each ball drawn is treated as a categorical draw over the K balls with
probabilities theta. (A draw of six is really sequential and without replacement;
the categorical model is an approximation that slightly overstates the variance of
the counts, so every Bayes factor is also calibrated against simulated fair
histories drawn without replacement.)

  H_fair : theta = (1/K, ..., 1/K)
  H_bias : theta ~ Dirichlet(a, ..., a), with the concentration a unknown and given a
           log-uniform prior on [1, 1000]. Under Dirichlet(a) a ball's probability
           varies by about 1/sqrt(a) relative to fair, so H_bias means biases of
           roughly 3% (a = 1000) up to 100% (a = 1). Biases far smaller than that can't
           be told apart from fair with a few thousand draws; letting H_bias include
           them would stop the data from ever strongly favouring fair.

The Dirichlet-multinomial marginal likelihood gives the Bayes factor BF = p(counts |
H_fair) / p(counts | H_bias), averaged over a. With prior odds 1:1 the posterior
probability of a fair machine is BF / (1 + BF).

The same machinery compares "one shared theta" against "a separate theta per group"
for machines, ball sets and calendar years (does anything make some balls more
likely?), and gives:
  - posterior 95% credible intervals for each ball's probability relative to fair
  - the posterior probability that any ball is more than 10% above or below fair
  - the Bayes factor accumulated draw by draw
  - walk-forward strategies (posterior-mean top 6, Thompson sampling, and Thompson
    sampling from the posterior for the machine used in the last round)
"""
from __future__ import annotations

import argparse, json, math, os, time

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

PICK = 6
SETS = ["n1", "n2", "n3", "n4", "n5", "n6"]
LOG_A_TEST = np.linspace(0, 3, 49)  # log10 concentration grid for H_bias (prior uniform on it)
LOG_A = np.linspace(0, 5, 81)  # wider grid for estimation (credible intervals, strategies)
LN10 = math.log(10)


# ---------------------------------------------------------------- marginal likelihoods
def log_dm(counts: np.ndarray, a: np.ndarray) -> np.ndarray:
    """log Dirichlet-multinomial likelihood of `counts` (sequence, not multiset) for each concentration in `a`.
    counts [..., K] -> [..., len(a)]"""
    c = np.asarray(counts, dtype=float)[..., None, :]
    a = np.asarray(a, dtype=float)[:, None]
    K, N = c.shape[-1], c.sum(-1)
    return gammaln(K * a[..., 0]) - gammaln(N + K * a[..., 0]) + (gammaln(c + a) - gammaln(a)).sum(-1)


def log_fair(counts: np.ndarray) -> np.ndarray:
    c = np.asarray(counts, dtype=float)
    return -c.sum(-1) * math.log(c.shape[-1])


def log_bias(counts: np.ndarray) -> np.ndarray:
    """log p(counts | H_bias), averaging the concentration over its log-uniform prior grid."""
    return logsumexp(log_dm(counts, 10 ** LOG_A_TEST), axis=-1) - math.log(len(LOG_A_TEST))


def log10_bf_fair(counts: np.ndarray) -> float:
    return float((log_fair(counts) - log_bias(counts)) / LN10)


def log10_bf_same(groups: np.ndarray) -> float:
    """log10 BF for "every group shares one theta" against "each group has its own theta,
    differing materially". groups [G, K].

    Shared: one theta ~ Dirichlet(a) with a on the wide grid, so it may be fair or biased.
    Separate: theta_g ~ Dirichlet(a) independently per group with a on the test grid, so
    groups differ from each other by roughly 3% or more per ball."""
    same = logsumexp(log_dm(groups.sum(0), 10 ** LOG_A)) - math.log(len(LOG_A))
    lg = log_dm(groups, 10 ** LOG_A_TEST).sum(0)  # each group its own theta, one shared concentration
    diff = logsumexp(lg) - math.log(len(LOG_A_TEST))
    return float((same - diff) / LN10)


def concentration_posterior(counts: np.ndarray) -> np.ndarray:
    lp = log_dm(counts, 10 ** LOG_A)
    return np.exp(lp - logsumexp(lp))


# ---------------------------------------------------------------- per-era analysis
def counts_of(sets: np.ndarray, K: int) -> np.ndarray:
    return np.bincount(sets.ravel(), minlength=K + 1)[1:]


def fair_sets(rng, n, K):
    return np.sort(np.argsort(rng.random((n, K)), axis=1)[:, :PICK] + 1, 1)


def grouped(sets: np.ndarray, labels: np.ndarray, K: int, min_rounds: int):
    keep = [g for g in pd.unique(labels) if (labels == g).sum() >= min_rounds]
    return keep, np.array([counts_of(sets[labels == g], K) for g in keep])


def evidence_word(l10: float, sides=("fair", "bias")) -> str:
    """Kass & Raftery style label for a log10 Bayes factor (positive favours sides[0])."""
    a = abs(l10)
    side = sides[0] if l10 > 0 else sides[1]
    if a < 0.5:
        return "inconclusive"
    return {True: "decisive", False: "strong" if a >= 1 else "substantial"}[a >= 2] + f" for {side}"


def era_study(df: pd.DataFrame, K: int, label: str, sims: int, rng) -> dict:
    df = df.sort_values(["draw_no", "round"]).reset_index(drop=True)
    sets = df[SETS].to_numpy(int)
    n = len(df)
    counts = counts_of(sets, K)
    bf = log10_bf_fair(counts)
    null = np.array([log10_bf_fair(counts_of(fair_sets(rng, n, K), K)) for _ in range(sims)])
    p_low = float(((null <= bf).sum() + 1) / (sims + 1))  # how often a fair history gives as little support for fair
    post_a = concentration_posterior(counts)
    # posterior samples of theta (mixture over the concentration grid)
    draws = 20000
    ai = rng.choice(len(LOG_A), draws, p=post_a)
    theta = rng.gamma(10 ** LOG_A[ai][:, None] + counts[None, :])
    theta = theta / theta.sum(1, keepdims=True)
    rel = theta * K
    lo, mid, hi = np.quantile(rel, [0.025, 0.5, 0.975], axis=0)
    p_any_10 = float((np.abs(rel - 1) > 0.10).any(1).mean())
    p_any_5 = float((np.abs(rel - 1) > 0.05).any(1).mean())
    # accumulated evidence, every 10 rounds
    cum = np.zeros((n, K))
    np.put_along_axis(cum, sets - 1, 1, axis=1)
    cum = np.cumsum(cum, 0)
    steps = list(range(9, n, 10)) + ([n - 1] if (n - 1) % 10 else [])
    path = [{"round": int(t + 1), "log10_bf": log10_bf_fair(cum[t])} for t in steps]
    # groups: machines, ball sets, calendar years
    comps = []
    for name, labels, mn in [("machines", df.machine.astype(str).str.strip().to_numpy(), 100),
                             ("ball sets", df.ball_set.astype(str).str.strip().to_numpy(), 80),
                             ("calendar years", df.date.str[:4].to_numpy(), 60)]:
        keep, G = grouped(sets, labels, K, mn)
        l10 = log10_bf_same(G)
        sim = []
        for _ in range(min(sims, 200)):
            fs = fair_sets(rng, n, K)
            sim.append(log10_bf_same(np.array([counts_of(fs[labels == g], K) for g in keep])))
        sim = np.array(sim)
        comps.append({"question": f"do {name} favour different balls?", "groups": [str(g) for g in keep],
                      "log10_bf_same": l10, "verdict": evidence_word(l10, ("no difference", "a difference")),
                      "fair_median": float(np.median(sim)), "p_low": float(((sim <= l10).sum() + 1) / (len(sim) + 1))})
    print(f"== {label}: {n} rounds; log10 BF(fair vs biased) = {bf:+.2f} ({evidence_word(bf)}); "
          f"fair histories median {np.median(null):+.2f}, p = {p_low:.3f}; P(fair) = {1 / (1 + 10 ** -bf):.4f}")
    print(f"   posterior concentration: median a = {10 ** np.interp(0.5, np.cumsum(post_a), LOG_A):.0f}; "
          f"P(any ball >10% off fair) = {p_any_10:.3f}, >5%: {p_any_5:.3f}")
    for c in comps:
        print(f"   {c['question']:38s} log10 BF(same) = {c['log10_bf_same']:+.2f} ({c['verdict']}); fair median {c['fair_median']:+.2f}, p = {c['p_low']:.3f}")
    return {"label": label, "pool": K, "rounds": n, "first_date": str(df.date.min()), "last_date": str(df.date.max()),
            "log10_bf_fair": bf, "verdict": evidence_word(bf), "p_fair_posterior": float(1 / (1 + 10 ** -bf)),
            "fair_sims_median": float(np.median(null)), "fair_sims_q05": float(np.quantile(null, 0.05)), "p_low": p_low,
            "concentration_posterior": {"log10_a": LOG_A.tolist(), "p": post_a.tolist()},
            "ball_rel": {"lo": lo.tolist(), "mid": mid.tolist(), "hi": hi.tolist(), "count": counts.tolist()},
            "p_any_off_10pct": p_any_10, "p_any_off_5pct": p_any_5, "evidence_path": path, "groups": comps}


# ---------------------------------------------------------------- strategies
def posterior_theta(counts: np.ndarray, rng, n: int) -> np.ndarray:
    post = concentration_posterior(counts)
    a = 10 ** LOG_A[rng.choice(len(LOG_A), n, p=post)]
    t = rng.gamma(a[:, None] + counts[None, :])
    return t / t.sum(1, keepdims=True)


def top6(score: np.ndarray, rng) -> np.ndarray:
    return np.argsort(-(score + rng.random(len(score)) * 1e-12))[:PICK] + 1


def strategies(hist_sets, hist_machine, last_machine, K, n, rng):
    counts = counts_of(hist_sets, K)
    post = concentration_posterior(counts)
    a_mean = float((10 ** LOG_A * post).sum())
    mean = (counts + a_mean) / (counts.sum() + K * a_mean)
    out = {
        "random (baseline)": np.array([np.sort(rng.choice(np.arange(1, K + 1), PICK, replace=False)) for _ in range(n)]),
        "posterior-mean top 6": np.sort(top6(mean, rng))[None, :],
        "Thompson sampling": np.array([np.sort(top6(t, rng)) for t in posterior_theta(counts, rng, n)]),
    }
    mc = counts_of(hist_sets[hist_machine == last_machine], K) if (hist_machine == last_machine).sum() >= 30 else counts
    out["Thompson sampling, last machine's posterior"] = np.array([np.sort(top6(t, rng)) for t in posterior_theta(mc, rng, n)])
    return out


def backtest(df: pd.DataFrame, K: int, split: str, seeds: list, tickets: int) -> dict:
    df = df.sort_values(["draw_no", "round"]).reset_index(drop=True)
    sets = df[SETS].to_numpy(int)
    machine = df.machine.astype(str).str.strip().to_numpy()
    test = np.flatnonzero(df.date.to_numpy() > split)
    exp = PICK * PICK / K
    var = PICK * (PICK / K) * (1 - PICK / K) * (K - PICK) / (K - 1)
    res = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        m = {}
        for i in test:
            tk = strategies(sets[:i], machine[:i], machine[i - 1], K, tickets, rng)
            actual = set(sets[i].tolist())
            for s, ts in tk.items():
                m.setdefault(s, []).extend(len(actual & set(t.tolist())) for t in ts)
        for s, v in m.items():
            v = np.array(v)
            res.setdefault(s, []).append(float((v.mean() - exp) / math.sqrt(var / len(v))))
    out = {s: {"z_mean": float(np.mean(z)), "z_min": float(np.min(z)), "z_max": float(np.max(z))} for s, z in res.items()}
    for s, r in out.items():
        print(f"    {s:46s} z {r['z_mean']:+.2f} [{r['z_min']:+.2f}, {r['z_max']:+.2f}]")
    rng = np.random.default_rng(seeds[0])
    picks = {s: [sorted(int(x) for x in t) for t in ts[:3]] for s, ts in
             strategies(sets, machine, machine[-1], K, 3, rng).items() if not s.startswith("random")}
    return {"strategies": out, "test_rounds": int(len(test)), "seeds": seeds, "tickets_per_round": tickets, "picks": picks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/uk_lotto_full.csv")
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--split-date", default="2022-12-31")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 1, 2, 3, 4])
    ap.add_argument("--tickets", type=int, default=20)
    ap.add_argument("--out", default="results/bayesian.json")
    a = ap.parse_args()
    t0 = time.time()
    df = pd.read_csv(a.csv)
    rng = np.random.default_rng(a.seed)
    e59, e49 = df[df.draw_no >= 2066], df[df.draw_no < 2066]
    eras = [era_study(e59, 59, "59-ball era (Oct 2015 - now)", a.sims, rng),
            era_study(e49, 49, "49-ball era (1994 - Oct 2015)", a.sims, rng)]
    print("== strategies (59-ball era, walk-forward)")
    bt = backtest(e59, 59, a.split_date, a.seeds, a.tickets)
    with open(a.out, "w") as fh:
        json.dump({"eras": eras, "backtest": bt, "split": a.split_date, "sims": a.sims}, fh, indent=1, default=float)
    print(f"wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
