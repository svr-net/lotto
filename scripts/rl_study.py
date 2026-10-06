"""Reinforcement learning on the UK Lotto draws.

Usage:
    python scripts/rl_study.py --csv data/uk_lotto_full.csv --out results/rl.json

Framing. Each round an agent buys tickets (its actions) and is rewarded with the
number of balls each ticket matches. The agent's tickets cannot change future
draws, so there are no state transitions to plan through: the problem is a
contextual bandit, the one-step case of reinforcement learning. Three families of
agent learn online, round by round, from the start of each era:

1. Bandits over strategies. Twelve simple strategies are the arms (hot, cold,
   overdue, repeat-last, n-gram grammar, early balls, drawn-order Markov chain,
   machine-matched hot, recent bonus balls, random...). epsilon-greedy, UCB1,
   Thompson sampling and EXP3 only see the reward of the arm they played; Hedge
   (multiplicative weights) sees every arm's result, as a lotto player does.
2. Contextual bandit over numbers. Linear Thompson sampling: a Bayesian linear
   model of "will this number be drawn?" on the 20 features from
   ml_ensemble_study.py; each ticket is the top six under one posterior sample.
3. Policy gradient. REINFORCE with a Plackett-Luce (sequential softmax) policy over
   the numbers whose logits are a linear function of those features plus a
   per-number bias, trained on its own tickets' rewards with a running baseline.

Agents are scored on the rounds after the split date against the 36/N chance
rate, over several seeds, in the 59-ball and 49-ball eras. z-scores use each
round's mean matches, so an agent whose tickets within a round are near-copies
isn't credited with more independent evidence than it has.
"""
from __future__ import annotations

import argparse, importlib.util, json, math, os, sys, time

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
from lotto_nn.grammar import extract_grammar, generate_batch as grammar_batch  # noqa: E402

_spec = importlib.util.spec_from_file_location("ml_ensemble_study", os.path.join(HERE, "ml_ensemble_study.py"))
ml = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ml)

PICK = 6
SETS = ["n1", "n2", "n3", "n4", "n5", "n6"]
DRAWN = ["d1", "d2", "d3", "d4", "d5", "d6"]
WARMUP = ml.WARMUP


# ---------------------------------------------------------------- arms (strategies)
def weighted(w, pool, n, rng):
    p = np.clip(w, 1e-9, None)
    p = p / p.sum()
    return np.array([rng.choice(pool, PICK, replace=False, p=p) + 1 for _ in range(n)])


def topk(score, pool, n, rng):
    t = np.argsort(-(score + rng.random(pool) * 1e-6))[:PICK] + 1
    return np.repeat(t[None, :], n, axis=0)


def freq(sets, pool, last=None):
    s = sets if last is None else sets[-last:]
    return np.bincount(s.ravel(), minlength=pool + 1)[1:].astype(float)


def arm_tickets(H: dict, pool: int, n: int, rng) -> dict:
    """Each arm's n tickets for the next round, given history H (arrays of earlier rounds)."""
    sets, drawn, bonus, machine = H["sets"], H["drawn"], H["bonus"], H["machine"]
    f100, f50 = freq(sets, pool, 100), freq(sets, pool, 50)
    gaps = np.full(pool, len(sets), dtype=float)
    for i, row in enumerate(sets[::-1][:400]):
        for b in row:
            if gaps[b - 1] == len(sets):
                gaps[b - 1] = i
    start = np.bincount(drawn[:, 0], minlength=pool + 1)[1:] + 1.0
    trans = np.ones((pool + 1, pool + 1))
    np.add.at(trans, (drawn[:, :-1].ravel(), drawn[:, 1:].ravel()), 1)
    markov = []
    for _ in range(n):
        p, seq = start.copy(), []
        for _ in range(PICK):
            p[np.array(seq, dtype=int) - 1] = 0
            nxt = int(rng.choice(pool, p=p / p.sum())) + 1
            seq.append(nxt)
            p = trans[nxt, 1:].copy()
        markov.append(seq)
    mrows = np.flatnonzero(machine == machine[-1])[-300:]
    recent_bonus = list(dict.fromkeys(bonus[::-1].tolist()))[:PICK]
    while len(recent_bonus) < PICK:
        recent_bonus.append(int(rng.integers(1, pool + 1)))
        recent_bonus = list(dict.fromkeys(recent_bonus))
    return {
        "random": weighted(np.ones(pool), pool, n, rng),
        "hot-weighted (last 100)": weighted(f100 + 1, pool, n, rng),
        "cold-weighted (last 100)": weighted(f100.max() - f100 + 1, pool, n, rng),
        "hot top-6 (last 50)": topk(f50, pool, n, rng),
        "cold top-6 (last 50)": topk(-f50, pool, n, rng),
        "most overdue top-6": topk(gaps, pool, n, rng),
        "repeat last draw": np.repeat(np.sort(sets[-1])[None, :], n, axis=0),
        "n-gram grammar": grammar_batch(extract_grammar(sets[-400:]), num_samples=n, max_num=pool, rng=rng),
        "early balls (1st/2nd out)": weighted(np.bincount(drawn[-200:, :2].ravel(), minlength=pool + 1)[1:] + 1.0, pool, n, rng),
        "drawn-order Markov chain": np.array(markov),
        "machine-matched hot": weighted(freq(sets[mrows], pool) + 1, pool, n, rng),
        "recent bonus balls": np.repeat(np.sort(recent_bonus)[None, :], n, axis=0),
    }


ARMS = ["random", "hot-weighted (last 100)", "cold-weighted (last 100)", "hot top-6 (last 50)", "cold top-6 (last 50)",
        "most overdue top-6", "repeat last draw", "n-gram grammar", "early balls (1st/2nd out)",
        "drawn-order Markov chain", "machine-matched hot", "recent bonus balls"]


# ---------------------------------------------------------------- bandit agents over arms
class EpsGreedy:
    name = "epsilon-greedy over strategies"

    def __init__(self, k, rng, eps=0.1):
        self.s, self.n, self.rng, self.eps = np.zeros(k), np.zeros(k), rng, eps

    def choose(self, m):
        mean = np.where(self.n > 0, self.s / np.maximum(self.n, 1), np.inf)
        best = int(np.argmax(mean + self.rng.random(len(mean)) * 1e-9))
        return np.where(self.rng.random(m) < self.eps, self.rng.integers(0, len(mean), m), best)

    def update(self, arms, rewards, all_rewards):
        np.add.at(self.s, arms, rewards)
        np.add.at(self.n, arms, 1)

    def shares(self):
        return self.n / self.n.sum()


class UCB1(EpsGreedy):
    name = "UCB1 over strategies"

    def choose(self, m):
        out = []
        untried = list(np.flatnonzero(self.n == 0))
        for _ in range(m):
            if untried:  # play every arm once before trusting the confidence bounds
                a = int(untried.pop(0))
            else:
                n = np.maximum(self.n, 1)
                ucb = self.s / n / PICK + np.sqrt(2 * math.log(max(self.n.sum(), 1)) / n)
                a = int(np.argmax(ucb))
            out.append(a)
        return np.array(out)


class GaussTS(EpsGreedy):
    name = "Thompson sampling over strategies"

    def __init__(self, k, rng, var=0.5):
        super().__init__(k, rng)
        self.var = var

    def choose(self, m):
        mean = self.s / np.maximum(self.n, 1)
        sd = np.sqrt(self.var / np.maximum(self.n, 1))
        return np.argmax(mean + sd * self.rng.standard_normal((m, len(mean))), axis=1)


class EXP3:
    name = "EXP3 over strategies (adversarial bandit)"

    def __init__(self, k, rng, gamma=0.05):
        self.w, self.rng, self.g, self.k = np.zeros(k), rng, gamma, k
        self.pulls = np.zeros(k)

    def probs(self):
        p = np.exp(self.w - self.w.max())
        return (1 - self.g) * p / p.sum() + self.g / self.k

    def choose(self, m):
        return self.rng.choice(self.k, m, p=self.probs())

    def update(self, arms, rewards, all_rewards):
        p = self.probs()
        for a, r in zip(arms, rewards):
            self.w[a] += self.g * (r / PICK) / p[a] / self.k / len(arms)
        np.add.at(self.pulls, arms, 1)

    def shares(self):
        return self.pulls / self.pulls.sum()


class Hedge:
    name = "Hedge over strategies (sees every strategy's result)"

    def __init__(self, k, rng, eta=0.1):
        self.cum, self.rng, self.eta, self.pulls = np.zeros(k), rng, eta, np.zeros(k)

    def probs(self):
        p = np.exp(self.eta * (self.cum - self.cum.max()))
        return p / p.sum()

    def choose(self, m):
        return self.rng.choice(len(self.cum), m, p=self.probs())

    def update(self, arms, rewards, all_rewards):
        self.cum += all_rewards
        np.add.at(self.pulls, arms, 1)

    def shares(self):
        return self.pulls / self.pulls.sum()


BANDITS = [EpsGreedy, UCB1, GaussTS, EXP3, Hedge]


# ---------------------------------------------------------------- agents over numbers
class LinearTS:
    name = "linear Thompson sampling over numbers (contextual)"

    def __init__(self, d, rng, prior=1.0, noise=0.09):
        self.A = np.eye(d) * prior
        self.b = np.zeros(d)
        self.rng, self.noise = rng, noise

    def tickets(self, X, n):
        cov = np.linalg.inv(self.A) * self.noise
        mu = cov @ self.b / self.noise
        W = self.rng.multivariate_normal(mu, cov, size=n)
        S = X @ W.T  # [pool, n]
        return np.argsort(-S, axis=0)[:PICK].T + 1

    def update(self, X, y):
        self.A += X.T @ X
        self.b += X.T @ y


class Reinforce:
    name = "REINFORCE policy gradient over numbers"

    def __init__(self, d, pool, rng, lr=0.02, lr_bias=0.01, decay=0.005):
        self.theta, self.bias = np.zeros(d), np.zeros(pool)
        self.rng, self.lr, self.lrb, self.baseline = rng, lr, lr_bias, PICK * PICK / pool
        self.decay = decay  # L2 weight decay: without real signal the policy relaxes back to uniform

    def logits(self, X):
        return X @ self.theta + self.bias

    def tickets(self, X, n):
        l = self.logits(X)
        g = -np.log(-np.log(self.rng.random((n, len(l)))))  # Gumbel-top-k = Plackett-Luce sample
        return np.argsort(-(l + g), axis=1)[:, :PICK] + 1

    def update(self, X, tickets, rewards):
        l = self.logits(X)
        grad = np.zeros(len(l))
        for t, r in zip(tickets, rewards):
            adv = r - self.baseline
            mask = np.ones(len(l), dtype=bool)
            for a in t - 1:  # gradient of the log-probability of the ordered sample
                p = np.where(mask, np.exp(l - l[mask].max()), 0)
                p /= p.sum()
                g = -p
                g[a] += 1
                grad += adv * g
                mask[a] = False
        grad /= len(tickets)
        self.theta += self.lr * X.T @ grad - self.decay * self.theta
        self.bias += self.lrb * grad - self.decay * self.bias
        self.baseline = 0.99 * self.baseline + 0.01 * float(np.mean(rewards))

    def entropy_share(self, X):
        """Entropy of the first-pick distribution as a share of the uniform maximum."""
        l = self.logits(X)
        p = np.exp(l - l.max())
        p /= p.sum()
        return float(-(p * np.log(p)).sum() / math.log(len(p)))


# ---------------------------------------------------------------- driver
def run_era(df: pd.DataFrame, pool: int, split: str, label: str, seeds: list, tickets: int, picks: bool) -> dict:
    df = df.sort_values(["draw_no", "round"]).reset_index(drop=True)
    sets, drawn = df[SETS].to_numpy(int), df[DRAWN].to_numpy(int)
    bonus, machine = df.bonus.to_numpy(int), df.machine.astype(str).str.strip().to_numpy()
    X, Y = ml.build_features(df, pool)
    dates = df.date.to_numpy()
    T = len(df)
    train_rows = np.flatnonzero((dates <= split) & (np.arange(T) >= WARMUP))
    mu = X[train_rows].reshape(-1, X.shape[-1]).mean(0)
    sd = X[train_rows].reshape(-1, X.shape[-1]).std(0) + 1e-9
    Xs = np.concatenate([(X - mu) / sd, np.ones(X.shape[:2] + (1,))], axis=-1)  # standardized + bias column
    test = (dates > split) & (np.arange(T) >= WARMUP)
    exp, var = PICK * PICK / pool, PICK * (PICK / pool) * (1 - PICK / pool) * (pool - PICK) / (pool - 1)
    names = [B.name for B in BANDITS] + [LinearTS.name, Reinforce.name]
    print(f"== {label}: {T} rounds; agents learn from round {WARMUP}, scored on {test.sum()} rounds after {split}")
    results = {n: [] for n in names + ["arm: " + a for a in ARMS]}
    paths, extras = {}, {}
    t0 = time.time()
    for si, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        bandits = [B(len(ARMS), np.random.default_rng(seed + 100 + i)) for i, B in enumerate(BANDITS)]
        lin = LinearTS(Xs.shape[-1], np.random.default_rng(seed + 200))
        rf = Reinforce(Xs.shape[-1], pool, np.random.default_rng(seed + 300))
        test_rewards = {n: [] for n in results}
        for t in range(WARMUP, T):
            H = {"sets": sets[:t], "drawn": drawn[:t], "bonus": bonus[:t], "machine": machine[:t]}
            arm_t = arm_tickets(H, pool, tickets, rng)
            actual = set(sets[t].tolist())
            arm_r = np.array([[len(actual & set(tk.tolist())) for tk in arm_t[a]] for a in ARMS])  # [arms, n]
            per_round = {}
            for b in bandits:
                arms = b.choose(tickets)
                r = arm_r[arms, np.arange(tickets)]
                b.update(arms, r, arm_r.mean(1))
                per_round[b.name] = r
            lt = lin.tickets(Xs[t], tickets)
            per_round[lin.name] = np.array([len(actual & set(x.tolist())) for x in lt])
            lin.update(Xs[t], Y[t])
            rt = rf.tickets(Xs[t], tickets)
            rr = np.array([len(actual & set(x.tolist())) for x in rt])
            rf.update(Xs[t], rt, rr)
            per_round[rf.name] = rr
            for i, a in enumerate(ARMS):
                per_round["arm: " + a] = arm_r[i]
            if test[t]:
                for n, r in per_round.items():
                    test_rewards[n].append(r)
        for n, rs in test_rewards.items():
            # z from per-round means: rounds are independent even when an agent's tickets within a
            # round are near-copies of each other (which would overstate confidence per ticket)
            rm = np.array([r.mean() for r in rs])
            results[n].append(float((rm.mean() - exp) / (rm.std(ddof=1) / math.sqrt(len(rm)))))
            if si == 0:
                paths[n] = np.cumsum([r.mean() - exp for r in rs]).round(4).tolist()
        if si == 0:
            extras = {"shares": {b.name: dict(zip(ARMS, b.shares().round(4).tolist())) for b in bandits},
                      "reinforce_entropy_share": rf.entropy_share(Xs[-1]),
                      "reinforce_theta": dict(zip(ml.FEATURES + ["bias"], rf.theta.round(4).tolist()))}
            if picks:  # next-draw picks from every agent, after learning on all rounds
                Xn, _ = ml.build_features(pd.concat([df, df.iloc[[-1]]], ignore_index=True), pool)
                xn = np.concatenate([(Xn[-1] - mu) / sd, np.ones((pool, 1))], axis=-1)
                H = {"sets": sets, "drawn": drawn, "bonus": bonus, "machine": machine}
                arm_t = arm_tickets(H, pool, 3, rng)
                pk = {}
                for b in bandits:
                    pk[b.name] = [sorted(int(x) for x in arm_t[ARMS[a]][i]) for i, a in enumerate(b.choose(3))]
                pk[lin.name] = [sorted(int(x) for x in t) for t in lin.tickets(xn, 3)]
                pk[rf.name] = [sorted(int(x) for x in t) for t in rf.tickets(xn, 3)]
                extras["picks"] = pk
        print(f"  seed {seed} done ({time.time() - t0:.0f}s)")
    out = {n: {"z_mean": float(np.mean(z)), "z_min": float(np.min(z)), "z_max": float(np.max(z))} for n, z in results.items()}
    for n in names:
        r = out[n]
        print(f"    {n:56s} z {r['z_mean']:+.2f} [{r['z_min']:+.2f}, {r['z_max']:+.2f}]")
    best_arm = max(ARMS, key=lambda a: out["arm: " + a]["z_mean"])
    print(f"    best single strategy in hindsight: {best_arm} z {out['arm: ' + best_arm]['z_mean']:+.2f}; "
          f"REINFORCE policy entropy {extras['reinforce_entropy_share']:.4f} of uniform")
    band = [2 * math.sqrt(var / tickets * (i + 1)) for i in range(int(test.sum()))]
    return {"label": label, "pool": pool, "rounds": T, "split": split, "test_rounds": int(test.sum()),
            "agents": {n: out[n] for n in names}, "arms": {a: out["arm: " + a] for a in ARMS},
            "paths": {n: paths[n] for n in names}, "chance_band": [round(x, 4) for x in band], **extras}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/uk_lotto_full.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 1, 2])
    ap.add_argument("--tickets", type=int, default=20)
    ap.add_argument("--out", default="results/rl.json")
    a = ap.parse_args()
    t0 = time.time()
    df = pd.read_csv(a.csv)
    eras = [run_era(df[df.draw_no >= 2066], 59, "2022-12-31", "59-ball era (Oct 2015 - now)", a.seeds, a.tickets, True),
            run_era(df[df.draw_no < 2066], 49, "2010-12-31", "49-ball era (1994 - Oct 2015)", a.seeds, a.tickets, False)]
    with open(a.out, "w") as fh:
        json.dump({"eras": eras, "seeds": a.seeds, "tickets": a.tickets, "arms": ARMS,
                   "backtest": {"picks": eras[0].get("picks", {})}}, fh, indent=1, default=float)
    print(f"wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
