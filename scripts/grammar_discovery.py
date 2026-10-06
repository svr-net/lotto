"""Search the UK Lotto draw history for a "grammar": any structure that makes draws
more predictable than a fair machine.

Usage:
    python scripts/grammar_discovery.py --csv data/uk_lotto_full.csv --out results/grammar.json

A fair draw is maximum-entropy noise: every ordered sequence of 6 distinct balls
from N is equally likely, which costs log2(N*(N-1)*...*(N-5)) bits to encode
(34.87 bits for N=59), and every unordered set costs log2(C(N,6)) bits (25.43).
A real grammar would show up as at least one of:

1. A model fitted on earlier draws that encodes later, unseen draws in fewer
   bits than that bound (held-out cross-entropy).
2. A general-purpose compressor shrinking the real draw stream more than
   simulated fair streams of the same length.
3. "Rules" (pairs, triples, ordered a-then-b pairs) that occur more often than
   the most frequent rule does in simulated fair histories.
4. Draws echoing draws 1-10 rounds earlier more or less than chance.

Each lottery era is analysed separately, because the ball pool changed from
49 to 59 on 2015-10-10.
"""
from __future__ import annotations

import argparse, bz2, json, lzma, math, os, sys, time, zlib
from math import comb

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PICK = 6
DRAWN = ["d1", "d2", "d3", "d4", "d5", "d6"]
SETS = ["n1", "n2", "n3", "n4", "n5", "n6"]
ALPHAS = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 1000, 10000, 100000]
LN2 = math.log(2)


def norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2))


# ---------------------------------------------------------------- probability models
# Every model gives, for each test round, the probability of its balls in the
# order they were drawn (ordered models) or of its sorted set (set models), as
# bits = -log2 P. Probabilities are renormalized over balls still in the machine,
# so every model is a proper distribution over valid draws.
def seq_bits(weights_fn, rows: np.ndarray, pool: int) -> np.ndarray:
    """Bits for ordered rows given weights_fn(row_index, position, prefix) -> weights over 1..pool."""
    out = np.zeros(len(rows))
    for i, row in enumerate(rows):
        bits = 0.0
        for k in range(PICK):
            w = np.asarray(weights_fn(i, k, row[:k]), dtype=float).copy()
            w[row[:k] - 1] = 0.0
            bits -= math.log2(w[row[k] - 1] / w.sum())
        out[i] = bits
    return out


class Uniform:
    name = "fair machine (uniform)"

    def fit(self, train, pool, alpha=None, ctx=None):
        self.pool = pool
        return self

    def bits(self, test, ctx):
        return seq_bits(lambda i, k, p: np.ones(self.pool), test, self.pool)


class Unigram:
    name = "number frequency"

    def fit(self, train, pool, alpha, ctx=None):
        self.pool = pool
        self.w = np.bincount(train.ravel(), minlength=pool + 1)[1:] + alpha
        return self

    def bits(self, test, ctx):
        return seq_bits(lambda i, k, p: self.w, test, self.pool)


class Positional:
    name = "frequency by draw position"

    def fit(self, train, pool, alpha, ctx=None):
        self.pool = pool
        self.w = np.array([np.bincount(train[:, k], minlength=pool + 1)[1:] + alpha for k in range(PICK)])
        return self

    def bits(self, test, ctx):
        return seq_bits(lambda i, k, p: self.w[k], test, self.pool)


class OrderMarkov:
    name = "drawn-order bigram (next ball given previous ball)"

    def fit(self, train, pool, alpha, ctx=None):
        self.pool = pool
        self.start = np.bincount(train[:, 0], minlength=pool + 1)[1:] + alpha
        t = np.zeros((pool + 1, pool + 1))
        np.add.at(t, (train[:, :-1].ravel(), train[:, 1:].ravel()), 1)
        self.trans = t[:, 1:] + alpha
        return self

    def bits(self, test, ctx):
        return seq_bits(lambda i, k, p: self.start if k == 0 else self.trans[p[-1]], test, self.pool)


class OrderTrigram:
    name = "drawn-order trigram (next ball given previous two)"

    def fit(self, train, pool, alpha, ctx=None):
        self.pool, self.alpha = pool, alpha
        self.bigram = OrderMarkov().fit(train, pool, alpha)
        self.tri = {}
        for row in train:
            for a, b, c in zip(row, row[1:], row[2:]):
                self.tri.setdefault((a, b), np.zeros(pool))[c - 1] += 1
        return self

    def bits(self, test, ctx):
        def w(i, k, p):
            if k < 2:
                return self.bigram.start if k == 0 else self.bigram.trans[p[-1]]
            # back off to the bigram distribution as a prior
            prior = self.bigram.trans[p[-1]]
            return self.tri.get((p[-2], p[-1]), np.zeros(self.pool)) + self.alpha * prior / prior.sum() * self.pool
        return seq_bits(w, test, self.pool)


class Echo:
    name = "echo of the previous round (repeat or avoid its numbers)"
    grid = [-0.5, -0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3, 0.5, 1.0, 2.0]

    def fit(self, train, pool, beta, ctx):
        # last round's numbers get weight 1 + beta (beta < 0 avoids them, > 0 repeats them)
        self.pool, self.beta = pool, beta
        return self

    def bits(self, test, ctx):
        prev = ctx["prev_sets"]

        def w(i, k, p):
            v = np.ones(self.pool)
            v[prev[i] - 1] += self.beta
            return v
        return seq_bits(w, test, self.pool)


class MachineFreq:
    name = "number frequency on the machine used"

    def fit(self, train, pool, alpha, ctx):
        self.pool, self.alpha = pool, alpha
        self.glob = np.bincount(train.ravel(), minlength=pool + 1)[1:] + alpha
        self.by = {}
        for m in np.unique(ctx["train_machine"]):
            rows = train[ctx["train_machine"] == m]
            self.by[m] = np.bincount(rows.ravel(), minlength=pool + 1)[1:]
        return self

    def bits(self, test, ctx):
        def w(i, k, p):
            c = self.by.get(ctx["machine"][i])
            prior = self.glob / self.glob.sum() * self.alpha * self.pool
            return prior if c is None else c + prior
        return seq_bits(w, test, self.pool)


class TransformerOrder:
    """The repo's causal LottoTransformer, trained on drawn order."""
    name = "transformer on drawn order"
    epochs = [5, 15, 30, 60]

    def fit(self, train, pool, alpha, ctx=None):
        import torch
        from lotto_nn.model import LottoTransformer
        from lotto_nn.train import train_model

        torch.manual_seed(0)
        self.pool = pool
        self.model = LottoTransformer()
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        train_model(self.model, torch.tensor(self.order(train), dtype=torch.long), opt,
                    epochs=int(alpha), batch_size=128, verbose=False)
        return self

    @staticmethod
    def order(rows):
        return rows

    def allowed(self, row, k):
        m = np.zeros(self.pool, dtype=bool)
        m[:] = True
        m[row[:k] - 1] = False
        return m

    def bits(self, test, ctx):
        import torch
        from lotto_nn.model import shift_right

        rows = self.order(test)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(shift_right(torch.tensor(rows, dtype=torch.long)))[:, :, 1:self.pool + 1].numpy()
        out = np.zeros(len(rows))
        for i, row in enumerate(rows):
            for k in range(PICK):
                l = np.where(self.allowed(row, k), logits[i, k], -np.inf)
                l = l - l.max()
                out[i] -= (l[row[k] - 1] - math.log(np.exp(l).sum())) / LN2
        return out


class SortedGrammar:
    """Repo-style grammar over sorted draws: start number, then next number given
    (position, current number), restricted to numbers that leave room for the rest.

    Counts are smoothed towards the fair conditional: with k numbers still to come
    after x, a fair draw picks x next in proportion to C(pool - x, k), the number of
    ways to finish the draw. So with heavy smoothing the grammar becomes the fair
    machine instead of a skewed sorted chain."""
    name = "sorted positional grammar (repo-style)"

    def fit(self, train, pool, alpha, ctx=None):
        self.pool, self.alpha = pool, alpha
        self.start = np.bincount(train[:, 0], minlength=pool + 1)[1:].astype(float)
        self.trans = np.zeros((PICK - 1, pool + 1, pool))
        for k in range(PICK - 1):
            np.add.at(self.trans[k], (train[:, k], train[:, k + 1] - 1), 1)
        return self

    def bits(self, test, ctx):
        out = np.zeros(len(test))
        for i, row in enumerate(test):
            for k in range(PICK):
                lo = row[k - 1] + 1 if k else 1
                hi = self.pool - (PICK - 1 - k)
                c = self.start if k == 0 else self.trans[k - 1, row[k - 1]]
                xs = np.arange(lo, hi + 1)
                fair = np.array([comb(self.pool - x, PICK - 1 - k) for x in xs], dtype=float)
                w = np.zeros(self.pool)
                w[lo - 1:hi] = c[lo - 1:hi] + self.alpha * len(xs) * fair / fair.sum()
                out[i] -= math.log2(w[row[k] - 1] / w.sum())
        return out


class SortedUniform:
    name = "fair machine (uniform)"

    def fit(self, train, pool, alpha=None, ctx=None):
        self.pool = pool
        return self

    def bits(self, test, ctx):
        return np.full(len(test), math.log2(comb(self.pool, PICK)))


class TransformerSorted(TransformerOrder):
    name = "transformer on sorted draws (repo model)"

    @staticmethod
    def order(rows):
        return np.sort(rows, axis=1)

    def allowed(self, row, k):
        m = np.zeros(self.pool, dtype=bool)
        lo = row[k - 1] + 1 if k else 1
        m[lo - 1:self.pool - (PICK - 1 - k)] = True
        return m


ORDER_MODELS = [Unigram, Positional, OrderMarkov, OrderTrigram, Echo, MachineFreq, TransformerOrder]
SET_MODELS = [SortedGrammar, TransformerSorted]


def evaluate(models, base, train, test, pool, ctx_train, ctx_test, ctx_val_fit, ctx_val):
    """Pick each model's setting on the last 20% of the training draws, refit on all
    training draws, and score the test draws against the uniform baseline."""
    cut = int(len(train) * 0.8)
    fit_rows, val_rows = train[:cut], train[cut:]
    base_bits = base().fit(train, pool).bits(test, ctx_test)
    rows = [{"model": base.name, "bits": float(base_bits.mean()), "excess": 0.0, "z": 0.0, "p_better": 0.5,
             "setting": "-"}]
    for M in models:
        grid = getattr(M, "epochs", getattr(M, "grid", ALPHAS))
        val = {a: M().fit(fit_rows, pool, a, ctx_val_fit).bits(val_rows, ctx_val).mean() for a in grid}
        best = min(val, key=val.get)
        bits = M().fit(train, pool, best, ctx_train).bits(test, ctx_test)
        d = bits - base_bits
        z = float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))) if d.std() > 0 else 0.0
        rows.append({"model": M.name, "bits": float(bits.mean()), "excess": float(d.mean()), "z": z,
                     "p_better": norm_cdf(z), "setting": f"{'epochs' if hasattr(M, 'epochs') else 'boost' if M is Echo else 'alpha'}={best}"})
        print(f"    {M.name:58s} {bits.mean():8.3f} bits  ({d.mean():+.3f} vs fair, z={z:+.2f})")
    return rows


# ---------------------------------------------------------------- compression
COMPRESSORS = {
    "lzma": lambda b: len(lzma.compress(b, preset=9 | lzma.PRESET_EXTREME)),
    "bz2": lambda b: len(bz2.compress(b, 9)),
    "zlib": lambda b: len(zlib.compress(b, 9)),
}


def fair_draws(rng, n, pool):
    return np.argsort(rng.random((n, pool)), axis=1)[:, :PICK] + 1


def compression_test(drawn: np.ndarray, pool: int, sims: int, rng) -> list:
    out = []
    for label, transform in [("drawn order", lambda r: r), ("sorted", lambda r: np.sort(r, axis=1))]:
        real = transform(drawn).astype(np.uint8).tobytes()
        fakes = [transform(fair_draws(rng, len(drawn), pool)).astype(np.uint8).tobytes() for _ in range(sims)]
        for cname, f in COMPRESSORS.items():
            obs = f(real)
            null = np.array([f(b) for b in fakes])
            p = float(((null <= obs).sum() + 1) / (sims + 1))
            out.append({"stream": label, "compressor": cname, "bytes_raw": len(real), "bytes": obs,
                        "fair_mean": float(null.mean()), "fair_min": int(null.min()), "p_more_compressible": p})
            print(f"    {label:12s} {cname:5s} {obs} bytes vs fair {null.mean():.1f} (min {null.min()}), p={p:.3f}")
    return out


# ---------------------------------------------------------------- rule mining
def rule_counts(drawn: np.ndarray, pool: int) -> dict:
    s = np.sort(drawn, axis=1)
    oh = np.zeros((len(s), pool + 1), dtype=np.int32)
    np.put_along_axis(oh, s, 1, axis=1)
    pairs = oh.T @ oh
    iu = np.triu_indices(pool + 1, 1)
    pair_counts = pairs[iu]
    tri_ids = []
    for a in range(PICK):
        for b in range(a + 1, PICK):
            for c in range(b + 1, PICK):
                tri_ids.append((s[:, a] * (pool + 1) + s[:, b]) * (pool + 1) + s[:, c])
    tri_ids, tri_n = np.unique(np.concatenate(tri_ids), return_counts=True)
    dig = drawn[:, :-1] * (pool + 1) + drawn[:, 1:]
    dig_ids, dig_n = np.unique(dig.ravel(), return_counts=True)
    return {"pair": (iu, pair_counts), "triple": (tri_ids, tri_n), "digram": (dig_ids, dig_n)}


def rules_test(drawn: np.ndarray, pool: int, sims: int, rng) -> dict:
    n = len(drawn)
    expected = {
        "pair": n * PICK * (PICK - 1) / (pool * (pool - 1)),
        "triple": n * PICK * (PICK - 1) * (PICK - 2) / (pool * (pool - 1) * (pool - 2)),
        "digram": n * (PICK - 1) / (pool * (pool - 1)),
    }
    obs = rule_counts(drawn, pool)
    null_max = {k: [] for k in obs}
    for _ in range(sims):
        r = rule_counts(fair_draws(rng, n, pool), pool)
        for k in r:
            null_max[k].append(int(r[k][1].max()))
    out = {}
    for kind, (ids, counts) in obs.items():
        order = np.argsort(-counts)[:8]
        top = []
        for j in order:
            if kind == "pair":
                label = f"{ids[0][j]} & {ids[1][j]}"
            elif kind == "triple":
                v = int(ids[j]); c = v % (pool + 1); v //= pool + 1; b = v % (pool + 1); a = v // (pool + 1)
                label = f"{a} & {b} & {c}"
            else:
                v = int(ids[j]); label = f"{v // (pool + 1)} then {v % (pool + 1)}"
            top.append({"rule": label, "count": int(counts[j])})
        nm = np.array(null_max[kind])
        mx = int(counts.max())
        p = float(((nm >= mx).sum() + 1) / (sims + 1))
        out[kind] = {"expected_each": expected[kind], "top": top, "observed_max": mx,
                     "fair_max_median": float(np.median(nm)), "fair_max_95": float(np.quantile(nm, 0.95)), "p": p}
        print(f"    {kind:7s} top {top[0]['rule']:>14s} x{mx} (each expected {expected[kind]:.2f}); "
              f"fair histories' top rule: median {np.median(nm):.0f}, 95% {np.quantile(nm, 0.95):.0f}; p={p:.3f}")
    return out


# ---------------------------------------------------------------- long-range echoes
def lag_test(drawn: np.ndarray, pool: int, max_lag: int = 10) -> list:
    s = np.zeros((len(drawn), pool + 1), dtype=np.int8)
    np.put_along_axis(s, drawn, 1, axis=1)
    exp = PICK * PICK / pool
    var = PICK * (PICK / pool) * (1 - PICK / pool) * (pool - PICK) / (pool - 1)
    out = []
    for lag in range(1, max_lag + 1):
        ov = (s[lag:] & s[:-lag]).sum(1)
        z = (ov.mean() - exp) / math.sqrt(var / len(ov))
        out.append({"lag": lag, "mean_overlap": float(ov.mean()), "expected": exp, "z": float(z),
                    "p": float(math.erfc(abs(z) / math.sqrt(2)))})
    zs = ", ".join(f"{o['z']:+.1f}" for o in out)
    print(f"    overlap with the round k back, z for k=1..{max_lag}: {zs}")
    return out


# ---------------------------------------------------------------- driver
def analyse_era(df: pd.DataFrame, pool: int, split: str, label: str, sims: int, seed: int) -> dict:
    df = df.sort_values(["draw_no", "round"]).reset_index(drop=True)
    drawn = df[DRAWN].to_numpy(int)
    sets = df[SETS].to_numpy(int)
    machine = df.machine.astype(str).str.strip().to_numpy()
    is_train = (df.date <= split).to_numpy()
    tr, te = np.flatnonzero(is_train), np.flatnonzero(~is_train)
    cut = int(len(tr) * 0.8)
    prev = np.vstack([sets[:1], sets[:-1]])  # previous round's numbers (first row: itself, unused in test)
    ctx = lambda idx: {"prev_sets": prev[idx], "machine": machine[idx], "train_machine": machine[idx]}
    ctx_train, ctx_test = ctx(tr), ctx(te)
    ctx_val_fit, ctx_val = ctx(tr[:cut]), ctx(tr[cut:])
    print(f"\n== {label}: {len(df)} rounds, {df.date.min()} -> {df.date.max()}; "
          f"train {len(tr)} up to {split}, test {len(te)}")
    print("  held-out bits per draw, ball order (fair limit "
          f"{math.log2(math.perm(pool, PICK)):.2f}):")
    order_rows = evaluate(ORDER_MODELS, Uniform, drawn[tr], drawn[te], pool, ctx_train, ctx_test, ctx_val_fit, ctx_val)
    print(f"  held-out bits per draw, number sets (fair limit {math.log2(comb(pool, PICK)):.2f}):")
    set_rows = evaluate(SET_MODELS, SortedUniform, sets[tr], sets[te], pool, ctx_train, ctx_test, ctx_val_fit, ctx_val)
    rng = np.random.default_rng(seed)
    print("  compression vs fair streams:")
    comp = compression_test(drawn, pool, sims, rng)
    print("  most frequent rules vs fair histories:")
    rules = rules_test(drawn, pool, sims, rng)
    print("  echoes:")
    lags = lag_test(drawn, pool)
    return {"label": label, "pool": pool, "rounds": len(df), "first_date": str(df.date.min()),
            "last_date": str(df.date.max()), "split": split, "train": int(len(tr)), "test": int(len(te)),
            "order_limit_bits": math.log2(math.perm(pool, PICK)), "set_limit_bits": math.log2(comb(pool, PICK)),
            "order_models": order_rows, "set_models": set_rows, "compression": comp, "rules": rules, "lags": lags}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/uk_lotto_full.csv")
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results/grammar.json")
    a = ap.parse_args()
    t0 = time.time()
    df = pd.read_csv(a.csv)
    eras = [
        analyse_era(df[df.draw_no >= 2066], 59, "2022-12-31", "59-ball era (Oct 2015 - now)", a.sims, a.seed),
        analyse_era(df[df.draw_no < 2066], 49, "2010-12-31", "49-ball era (1994 - Oct 2015)", a.sims, a.seed + 1),
    ]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"eras": eras, "sims": a.sims}, fh, indent=1, default=float)
    print(f"\nwrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
