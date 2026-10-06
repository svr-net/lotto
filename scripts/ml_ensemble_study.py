"""Machine-learning ensembles (gradient boosting, random forest, logistic regression,
soft-voting) on the UK Lotto draws.

Usage:
    python scripts/ml_ensemble_study.py --csv data/uk_lotto_full.csv --out results/ml_ensemble.json

Each round x number becomes one supervised example: "will this number be drawn in
this round?", with features computed only from earlier rounds:
  - recent frequency (last 10, 25, 50, 100 rounds and all history)
  - rounds since last seen, average gap, and the gap relative to that average
  - appeared in the last round / either of the last two / was the last bonus
  - average position in the drawing order when it came out
  - rate on the machine used in the previous round
  - the number itself: value, odd/even, decade, residues mod 3, 5 and 7
  - the previous round's sum (shared context, so trees can form interactions)

Models are trained walk-forward: on all rounds before each block of 26 test rounds,
then used to predict that block. Each era (49 and 59 balls) is run separately.
Scoring:
  - log-loss against the fair base rate (6/49 or 6/59) in bits per round; below
    zero would mean the model knows something
  - AUC of each round's ranking of the numbers (0.5 = chance)
  - matches per ticket for the model's top-6 ticket and for tickets sampled in
    proportion to its probabilities, against the 36/N chance rate
  - permutation importance of each feature for gradient boosting
"""
from __future__ import annotations

import argparse, json, math, os, time, warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

PICK = 6
DRAWN = ["d1", "d2", "d3", "d4", "d5", "d6"]
SETS = ["n1", "n2", "n3", "n4", "n5", "n6"]
WARMUP = 100
FEATURES = ["freq_10", "freq_25", "freq_50", "freq_100", "freq_all", "gap", "mean_gap", "gap_ratio",
            "in_last", "in_last_2", "last_bonus", "mean_position", "machine_rate",
            "value", "odd", "decade", "mod3", "mod5", "mod7", "last_sum"]


# ---------------------------------------------------------------- features
def build_features(df: pd.DataFrame, pool: int):
    """Features for every round t (using rounds < t only) and every number.
    Returns X [rounds, pool, features], y [rounds, pool]."""
    sets, drawn = df[SETS].to_numpy(int), df[DRAWN].to_numpy(int)
    bonus, machine = df.bonus.to_numpy(int), df.machine.astype(str).str.strip().to_numpy()
    T = len(df)
    A = np.zeros((T, pool), dtype=float)
    np.put_along_axis(A, sets - 1, 1, axis=1)
    C = np.vstack([np.zeros(pool), np.cumsum(A, 0)])  # C[t] = counts over rounds < t
    X = np.zeros((T, pool, len(FEATURES)))
    last_seen = np.full(pool, -1.0)
    pos_sum, pos_n = np.zeros(pool), np.zeros(pool)
    mach_counts, mach_rounds = {}, {}
    nums = np.arange(1, pool + 1)
    static = np.column_stack([nums, nums % 2, nums // 10, nums % 3, nums % 5, nums % 7])
    for t in range(T):
        f = {}
        for w in (10, 25, 50, 100):
            lo = max(0, t - w)
            f[f"freq_{w}"] = (C[t] - C[lo]) / max(1, t - lo)
        f["freq_all"] = C[t] / max(1, t)
        gap = np.where(last_seen >= 0, t - last_seen, t + 1)
        mean_gap = (t + 1) / (C[t] + 1)
        f["gap"], f["mean_gap"], f["gap_ratio"] = gap, mean_gap, gap / mean_gap
        f["in_last"] = A[t - 1] if t >= 1 else np.zeros(pool)
        f["in_last_2"] = np.maximum(A[t - 1], A[t - 2]) if t >= 2 else f["in_last"]
        lb = np.zeros(pool)
        if t >= 1:
            lb[bonus[t - 1] - 1] = 1
        f["last_bonus"] = lb
        f["mean_position"] = np.where(pos_n > 0, pos_sum / np.maximum(pos_n, 1), 3.5)
        m = machine[t - 1] if t >= 1 else None
        f["machine_rate"] = (mach_counts[m] / mach_rounds[m]) if m in mach_counts else f["freq_all"]
        f["last_sum"] = np.full(pool, sets[t - 1].sum() if t >= 1 else PICK * (pool + 1) / 2)
        X[t] = np.column_stack([f[k] for k in FEATURES[:13]] + [static, f["last_sum"]])
        # update running state with round t
        for k, b in enumerate(drawn[t]):
            last_seen[b - 1] = t
            pos_sum[b - 1] += k
            pos_n[b - 1] += 1
        mach_counts.setdefault(machine[t], np.zeros(pool))
        mach_rounds[machine[t]] = mach_rounds.get(machine[t], 0) + 1
        mach_counts[machine[t]] += A[t]
    return X, A


# ---------------------------------------------------------------- models
def make_models(seed: int) -> dict:
    return {
        "gradient boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.2, n_iter_no_change=20, random_state=seed),
        "random forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=100, max_features="sqrt", n_jobs=-1, random_state=seed),
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500)),
    }


ENSEMBLE = "ensemble (average of the three)"


def per_round_auc(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUC of each row's scores against its 0/1 labels (ties count half)."""
    out = np.empty(len(p))
    for i in range(len(p)):
        pos, neg = p[i][y[i] == 1], p[i][y[i] == 0]
        out[i] = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))
    return out


def bits(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Log-loss in bits per round (summed over the numbers)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log2(p) + (1 - y) * np.log2(1 - p)).sum(1)


def sample_tickets(p: np.ndarray, n: int, rng) -> np.ndarray:
    w = np.clip(p, 1e-9, None)
    w = w / w.sum()
    return np.array([rng.choice(len(p), PICK, replace=False, p=w) + 1 for _ in range(n)])


def zscore(matches: np.ndarray, pool: int) -> float:
    exp = PICK * PICK / pool
    var = PICK * (PICK / pool) * (1 - PICK / pool) * (pool - PICK) / (pool - 1)
    return float((matches.mean() - exp) / math.sqrt(var / len(matches)))


def era_study(df: pd.DataFrame, pool: int, split: str, label: str, seeds: list, block: int, tickets: int) -> dict:
    df = df.sort_values(["draw_no", "round"]).reset_index(drop=True)
    t0 = time.time()
    X, Y = build_features(df, pool)
    dates = df.date.to_numpy()
    test = np.flatnonzero((dates > split) & (np.arange(len(df)) >= WARMUP))
    names = list(make_models(0)) + [ENSEMBLE]
    print(f"== {label}: {len(df)} rounds, features built in {time.time() - t0:.0f}s; testing {len(test)} rounds after {split}")
    base_p = PICK / pool
    base_bits = bits(np.full((len(test), pool), base_p), Y[test])
    runs = {n: {"bits": [], "auc": [], "top6": [], "sampled": []} for n in names}
    last_models = None
    for seed in seeds:
        rng = np.random.default_rng(seed)
        P = {n: np.zeros((len(test), pool)) for n in names}
        for b0 in range(0, len(test), block):
            idx = test[b0:b0 + block]
            tr = np.arange(WARMUP, idx[0])
            Xtr, ytr = X[tr].reshape(-1, X.shape[-1]), Y[tr].ravel()
            models = make_models(seed)
            for n, m in models.items():
                m.fit(Xtr, ytr)
                P[n][b0:b0 + len(idx)] = m.predict_proba(X[idx].reshape(-1, X.shape[-1]))[:, 1].reshape(len(idx), pool)
            last_models = models
        P[ENSEMBLE] = np.mean([P[n] for n in names[:-1]], axis=0)
        for n in names:
            runs[n]["bits"].append(bits(P[n], Y[test]) - base_bits)
            runs[n]["auc"].append(per_round_auc(P[n], Y[test]))
            top = np.argsort(-P[n], axis=1)[:, :PICK]
            runs[n]["top6"].append(np.take_along_axis(Y[test], top, axis=1).sum(1))
            runs[n]["sampled"].append(np.concatenate([Y[test][i][sample_tickets(P[n][i], tickets, rng) - 1].sum(1)
                                                      for i in range(len(test))]))
        print(f"  seed {seed} done ({time.time() - t0:.0f}s)")
    out = {}
    for n in names:
        d = np.mean(runs[n]["bits"], axis=0)  # per-round excess bits averaged over seeds
        auc = np.mean(runs[n]["auc"], axis=0)
        out[n] = {
            "excess_bits": float(d.mean()), "excess_bits_z": float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))),
            "auc": float(auc.mean()), "auc_z": float((auc.mean() - 0.5) / (auc.std(ddof=1) / math.sqrt(len(auc)))),
            "top6_z": [zscore(m, pool) for m in runs[n]["top6"]],
            "sampled_z": [zscore(m, pool) for m in runs[n]["sampled"]],
            "top6_mean": float(np.mean([m.mean() for m in runs[n]["top6"]])),
        }
        r = out[n]
        print(f"    {n:34s} bits {r['excess_bits']:+.4f} (z {r['excess_bits_z']:+.2f})  AUC {r['auc']:.4f} (z {r['auc_z']:+.2f})  "
              f"top-6 z {np.mean(r['top6_z']):+.2f}  sampled z {np.mean(r['sampled_z']):+.2f}")
    # permutation importance of the last gradient-boosting model on the final test block
    gb = last_models["gradient boosting"]
    idx = test[-block:]
    imp = permutation_importance(gb, X[idx].reshape(-1, X.shape[-1]), Y[idx].ravel(), scoring="neg_log_loss",
                                 n_repeats=5, random_state=0)
    importance = sorted(({"feature": f, "mean": float(m), "sd": float(s)} for f, m, s in
                         zip(FEATURES, imp.importances_mean, imp.importances_std)), key=lambda r: -r["mean"])
    print("    top features (gradient boosting):", ", ".join(f"{r['feature']} {r['mean']:+.5f}" for r in importance[:5]))
    # picks for the next round: fit on everything, predict the round after the last
    picks = None
    if pool == 59:
        Xn, _ = build_features(pd.concat([df, df.iloc[[-1]]], ignore_index=True), pool)
        tr = np.arange(WARMUP, len(df))
        rng = np.random.default_rng(seeds[0])
        probs = []
        for n, m in make_models(seeds[0]).items():
            m.fit(X[tr].reshape(-1, X.shape[-1]), Y[tr].ravel())
            probs.append(m.predict_proba(Xn[-1])[:, 1])
        p = np.mean(probs, axis=0)
        picks = {"top6": sorted(int(i) + 1 for i in np.argsort(-p)[:PICK]),
                 "sampled": [sorted(int(x) for x in t) for t in sample_tickets(p, 4, rng)],
                 "max_prob": float(p.max()), "min_prob": float(p.min())}
        print(f"    next-round picks: top-6 {picks['top6']}; probabilities range {p.min():.4f}-{p.max():.4f} (fair {PICK / pool:.4f})")
    return {"label": label, "pool": pool, "rounds": len(df), "split": split, "test_rounds": int(len(test)),
            "base_rate": base_p, "models": out, "importance": importance, "picks": picks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/uk_lotto_full.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 1, 2])
    ap.add_argument("--block", type=int, default=26, help="retrain every N test rounds")
    ap.add_argument("--tickets", type=int, default=20)
    ap.add_argument("--out", default="results/ml_ensemble.json")
    a = ap.parse_args()
    t0 = time.time()
    df = pd.read_csv(a.csv)
    eras = [
        era_study(df[df.draw_no >= 2066], 59, "2022-12-31", "59-ball era (Oct 2015 - now)", a.seeds, a.block, a.tickets),
        era_study(df[df.draw_no < 2066], 49, "2010-12-31", "49-ball era (1994 - Oct 2015)", a.seeds, a.block, a.tickets),
    ]
    with open(a.out, "w") as fh:
        json.dump({"eras": eras, "seeds": a.seeds, "block": a.block, "tickets": a.tickets, "features": FEATURES},
                  fh, indent=1, default=float)
    print(f"wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
