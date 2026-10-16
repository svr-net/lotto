"""Which lines do other players pick? Fit it from the number of jackpot winners in every UK
Lotto round since 1994, then use it to choose lines that are just as likely to win but less likely
to share the jackpot.

Every line has the same chance of being drawn, but the jackpot is split between everyone holding
the winning line. When a popular line comes up (birthdays, runs like 4-5-6, last draw's numbers)
more people share it. The number of jackpot winners in each round is Poisson with a mean that
grows with ticket sales and with how popular the drawn line is. The model is

    log E[winners] = year/era/round effect + b_pot * log(total pot) + sum_j beta_j * feature_j(line)

fitted by Poisson regression. Features: how many numbers are 1-31 (birthdays) and 1-12 (months),
whether 7 is in the line, the number of consecutive pairs, the largest group sharing a last digit,
and how many numbers repeat the previous round. Only the jackpot is shared: Match 3 to Match 5 +
bonus pay fixed prizes, so choosing unpopular lines changes nothing below the jackpot.

    python scripts/popularity_study.py --out results/popularity.json
"""
import argparse, json

import numpy as np
import pandas as pd

FEATURES = ["birthday numbers (1-31)", "month numbers (1-12)", "has 7", "consecutive pairs",
            "same last digit", "repeats last round"]
N = [f"n{i}" for i in range(1, 7)]


def line_features(line, prev=()) -> np.ndarray:
    a = np.sort(np.asarray(line))
    tails = np.bincount(a % 10, minlength=10)
    return np.array([(a <= 31).sum(), (a <= 12).sum(), float(7 in a), (np.diff(a) == 1).sum(),
                     tails.max() - 1, len(set(a.tolist()) & set(prev))], dtype=float)


def design(df: pd.DataFrame):
    """Feature matrix X, nuisance matrix Z (year x era x round dummies + log total pot), y."""
    df = df.reset_index(drop=True)
    prev, last = [], {}
    for _, r in df.iterrows():  # previous round = previous row of the same round number, or round 1
        prev.append(last.get(int(r["round"]), last.get(1, ())))
        last[int(r["round"])] = tuple(int(r[c]) for c in N)
    X = np.array([line_features(df.loc[i, N].astype(int).values, prev[i]) for i in range(len(df))])
    era = np.where(df.draw_no >= 2066, 59, 49)
    grp = pd.Series([f"{d[:4]}-{e}-r{r}" for d, e, r in zip(df.date, era, df["round"])])
    Z = pd.get_dummies(grp).values.astype(float)
    # The feed's jackpot is each winner's share once it is won, so it shrinks when a popular line
    # comes up. The whole pot (share x winners) tracks sales and rollovers instead.
    pot = df.jackpot.values * np.maximum(df.wins.values, 1)
    Z = np.column_stack([Z, np.log(np.clip(pot, 1e5, None))])
    return X, Z, df.wins.values.astype(float), grp.values


def poisson_fit(M: np.ndarray, y: np.ndarray, iters: int = 100):
    """Poisson GLM (log link) by IRLS with a tiny ridge for the year dummies. Returns beta, cov, loglik."""
    beta = np.zeros(M.shape[1])
    beta[:] = 0
    ridge = 1e-6 * np.eye(M.shape[1])
    for _ in range(iters):
        mu = np.exp(np.clip(M @ beta, -30, 30))
        H = M.T @ (M * mu[:, None]) + ridge
        step = np.linalg.solve(H, M.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-9:
            break
    mu = np.exp(M @ beta)
    cov = np.linalg.inv(M.T @ (M * mu[:, None]) + ridge)
    ll = float((y * np.log(np.where(mu > 0, mu, 1)) - mu).sum())  # up to a constant in y
    return beta, cov, ll


def random_lines(pool: int, n: int, rng) -> np.ndarray:
    return np.argsort(rng.random((n, pool)), axis=1)[:, :6] + 1


class Popularity:
    """Relative popularity of a line (1 = an average random line) and its expected jackpot share."""

    def __init__(self, beta: np.ndarray, lam: float, pool: int = 59, seed: int = 0):
        self.beta, self.lam = np.asarray(beta), lam
        sample = random_lines(pool, 20000, np.random.default_rng(seed))
        raw = np.exp(np.array([line_features(l) for l in sample]) @ self.beta)
        self.norm = raw.mean()
        self.typical_share = float(np.mean(self.share_of(raw / self.norm)))

    def popularity(self, line, prev=()) -> float:
        return float(np.exp(line_features(line, prev) @ self.beta) / self.norm)

    def share_of(self, pop):
        """Expected fraction of the jackpot a winning ticket keeps: other winners ~ Poisson(lam*pop),
        so E[1/(1+K)] = (1 - exp(-mu)) / mu."""
        mu = np.maximum(self.lam * np.asarray(pop, dtype=float), 1e-12)
        return (1 - np.exp(-mu)) / mu

    def value(self, line, prev=()) -> float:
        """Expected jackpot payout relative to an average random line (same chance of winning)."""
        return float(self.share_of(self.popularity(line, prev)) / self.typical_share)


def pattern_line(line) -> bool:
    """Pattern tickets (an even step like 7-14-21-28-35-42, or a run of four like 4-5-6-7) are played
    by thousands of people, but none has ever been drawn, so the model can't see them; skip them.
    Lines with two or more consecutive pairs are skipped too: few drawn lines have them, so the
    model's estimate for them is an extrapolation."""
    d = np.diff(np.sort(np.asarray(line)))
    return len(set(d.tolist())) == 1 or (d == 1).sum() >= 2


def choose_lines(score, pop: Popularity, prev=(), n_lines: int = 5, n_cand: int = 20000,
                 max_overlap: int = 1, seed: int = 0) -> list:
    """Sample candidate lines in proportion to the investigations' combined support (as the combined
    lines are), then keep the least popular ones, with at most max_overlap numbers in common (relaxed
    if that leaves too few lines). Sharing at most one number keeps nearly all of the chance that some
    line wins a prize; sharing none would force birthday numbers in and cut the jackpot advantage."""
    rng = np.random.default_rng(seed)
    p = np.asarray(score, dtype=float)
    p = p + p.mean() * 0.2
    p /= p.sum()
    g = -np.log(rng.random((n_cand, len(p)))) / p  # Gumbel-style weighted sampling without replacement
    cand = np.sort(np.argsort(g, axis=1)[:, :6] + 1, axis=1)
    cand = np.unique(cand, axis=0)
    cand = cand[[not pattern_line(c) for c in cand]]
    pops = np.array([pop.popularity(c, prev) for c in cand])
    chosen = []
    for i in np.argsort(pops):
        c = set(cand[i].tolist())
        if all(len(c & set(o)) <= max_overlap for o in chosen):
            chosen.append(sorted(int(x) for x in cand[i]))
        if len(chosen) == n_lines:
            break
    if len(chosen) < n_lines and max_overlap < 6:
        return choose_lines(score, pop, prev, n_lines, n_cand, max_overlap + 1, seed)
    return chosen


def fit(df: pd.DataFrame, recent_rounds: int = 208) -> dict:
    X, Z, y, grp = design(df)
    b0, _, ll0 = poisson_fit(Z, y)
    M = np.column_stack([X, Z])
    b, cov, ll1 = poisson_fit(M, y)
    k = X.shape[1]
    se = np.sqrt(np.diag(cov))[:k]
    lr = 2 * (ll1 - ll0)
    from scipy.stats import chi2
    era59 = (df.draw_no >= 2066).values
    by_low = pd.DataFrame({"era": np.where(era59, 59, 49), "low": X[:, 0], "wins": y}).groupby(["era", "low"]).wins
    return {
        "features": [{"name": FEATURES[j], "beta": float(b[j]), "se": float(se[j]), "z": float(b[j] / se[j]),
                      "multiplier": float(np.exp(b[j]))} for j in range(k)],
        "beta": b[:k].tolist(), "lr_stat": float(lr), "df": k, "p": float(chi2.sf(lr, k)),
        "rounds": int(len(y)), "pot_coef": float(b[-1]),
        "lam": float(y[era59][-recent_rounds:].mean()), "recent_rounds": recent_rounds,
        "birthday_table": [{"era": int(e), "low": int(l), "rounds": int(g.size), "mean_wins": float(g.mean())}
                           for (e, l), g in by_low],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/uk_lotto_full.csv")
    ap.add_argument("--out", default="results/popularity.json")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    res = fit(df)
    pop = Popularity(res["beta"], res["lam"])
    examples = {"draw 9 (133 jackpot winners) 7 17 23 32 38 42": [7, 17, 23, 32, 38, 42],
                "all birthdays 3 7 12 19 25 31": [3, 7, 12, 19, 25, 31],
                "high and scattered 33 38 44 49 53 58": [33, 38, 44, 49, 53, 58]}
    res["examples"] = [{"label": k, "popularity": pop.popularity(v), "value": pop.value(v)} for k, v in examples.items()]
    res["typical_share"] = pop.typical_share
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    for f in res["features"]:
        print(f"{f['name']:>26}: x{f['multiplier']:.2f} per unit (z {f['z']:+.1f})")
    print(f"LR test vs sales-only model: {res['lr_stat']:.1f} on {res['df']} df, p = {res['p']:.2g}")
    print(f"expected other winners for a typical line: {res['lam']:.3f}; typical share {pop.typical_share:.3f}")
    for e in res["examples"]:
        print(f"  {e['label']}: popularity x{e['popularity']:.2f}, jackpot value x{e['value']:.2f}")


if __name__ == "__main__":
    main()
