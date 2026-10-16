"""Walk-forward backtest of lotto "prediction" strategies against real UK Lotto draws.

Usage:
    python scripts/walk_forward_backtest.py --csv data/uk_lotto_full.csv \
        --split-date 2022-12-31 --tickets 20 --out results/

For every draw round after --split-date, each strategy sees only rounds drawn on
earlier dates, produces tickets, and is scored on how many of the six main
numbers each ticket matched. Scores are compared to the exact random-chance
(hypergeometric) distribution. Only the 1-59 era (draw 2066, 2015-10-10, onward)
is used so the ball pool is consistent. After the backtest, every strategy is
fitted on the full history to produce tickets for the next draw.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lotto_nn.backtest import chi2_sf  # noqa: E402
from lotto_nn.grammar import generate_batch as grammar_batch, grammar_from_state, new_grammar_state, update_grammar  # noqa: E402

COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]
DRAWN = ["d1", "d2", "d3", "d4", "d5", "d6"]
POOL, PICK = 59, 6
FIRST_59_DRAW = 2066


# ---------------------------------------------------------------- statistics
def hypergeom_pmf(k: int) -> float:
    return math.comb(PICK, k) * math.comb(POOL - PICK, PICK - k) / math.comb(POOL, PICK)


EXPECTED = np.array([hypergeom_pmf(k) for k in range(PICK + 1)])
EXP_MEAN = PICK * PICK / POOL
EXP_VAR = PICK * (PICK / POOL) * (1 - PICK / POOL) * (POOL - PICK) / (POOL - 1)


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


# ---------------------------------------------------------------- strategies
def freq(hist: np.ndarray, last: int | None = None) -> np.ndarray:
    h = hist if last is None else hist[-last:]
    return np.bincount(h.ravel(), minlength=POOL + 1)[1:].astype(float)


def weighted_tickets(weights: np.ndarray, n: int, rng) -> np.ndarray:
    p = weights / weights.sum()
    return np.array([np.sort(rng.choice(np.arange(1, POOL + 1), PICK, replace=False, p=p)) for _ in range(n)])


def top6(score: np.ndarray, rng) -> np.ndarray:
    jitter = rng.random(POOL) * 1e-6  # break ties randomly
    return np.sort(np.argsort(-(score + jitter))[:PICK] + 1)[None, :]


def gaps(hist: np.ndarray) -> np.ndarray:
    """Rounds since each number was last drawn (0 = in the last round; len(hist) if never)."""
    seen = np.zeros((len(hist), POOL), dtype=bool)
    np.put_along_axis(seen, np.asarray(hist) - 1, True, axis=1)
    back = seen[::-1]
    return np.where(back.any(0), back.argmax(0), len(hist)).astype(float)


def s_random(hist, n, rng, ctx):
    return weighted_tickets(np.ones(POOL), n, rng)


def s_hot_weighted(hist, n, rng, ctx):
    return weighted_tickets(freq(hist, 100) + 1, n, rng)


def s_cold_weighted(hist, n, rng, ctx):
    f = freq(hist, 100)
    return weighted_tickets(f.max() - f + 1, n, rng)


def s_hot_top6(hist, n, rng, ctx):
    return top6(freq(hist, 50), rng)


def s_cold_top6(hist, n, rng, ctx):
    return top6(-freq(hist, 50), rng)


def s_overdue_top6(hist, n, rng, ctx):
    return top6(gaps(hist), rng)


def s_repeat_last(hist, n, rng, ctx):
    return np.sort(hist[-1])[None, :]


def s_grammar(hist, n, rng, ctx):
    # the history only ever grows by whole rounds, so count just the new ones
    state = ctx.get("grammar_state")
    if state is None or state["rows"] > len(hist):
        state = ctx["grammar_state"] = new_grammar_state()
    update_grammar(state, hist[state["rows"]:])
    return grammar_batch(grammar_from_state(state), num_samples=n, rng=rng)


def s_transformer(hist, n, rng, ctx):
    """Repo LottoTransformer, retrained every `retrain` rounds (frozen in between)."""
    import torch
    from lotto_nn.model import LottoTransformer
    from lotto_nn.sample import generate_batch
    from lotto_nn.train import train_model

    if ctx.get("model") is None or len(hist) - ctx["trained_on"] >= ctx["retrain"]:
        torch.manual_seed(int(rng.integers(1 << 31)))
        model = LottoTransformer()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_model(model, torch.tensor(hist, dtype=torch.long), opt, epochs=ctx["epochs"], verbose=False)
        ctx.update(model=model, trained_on=len(hist))
    torch.manual_seed(int(rng.integers(1 << 31)))
    return generate_batch(ctx["model"], num_samples=n)


# ---------------------------------------------------------------- draw-order strategies
# These use ctx["aux"]: per-round arrays aligned with `hist` ("drawn": balls in the
# order the machine drew them, "bonus", "machine", "round") plus "target_round".
def s_early_balls(hist, n, rng, ctx):
    """Numbers that came out first or second in the last 200 rounds, weighted by count."""
    d = ctx["aux"]["drawn"][-200:, :2]
    return weighted_tickets(np.bincount(d.ravel(), minlength=POOL + 1)[1:] + 1.0, n, rng)


def order_markov(drawn: np.ndarray):
    """First-ball counts and ball-to-next-ball transition counts over drawn order (add-1 smoothed)."""
    start = np.bincount(drawn[:, 0], minlength=POOL + 1)[1:] + 1.0
    trans = np.ones((POOL + 1, POOL + 1))
    np.add.at(trans, (drawn[:, :-1].ravel(), drawn[:, 1:].ravel()), 1)
    return start, trans[:, 1:]


def s_order_markov(hist, n, rng, ctx):
    """Markov chain over the order balls were drawn: next ball depends on the previous ball."""
    start, trans = order_markov(ctx["aux"]["drawn"])
    out = []
    for _ in range(n):
        p = start.copy()
        seq = []
        for _ in range(PICK):
            p[[x - 1 for x in seq]] = 0
            nxt = int(rng.choice(np.arange(1, POOL + 1), p=p / p.sum()))
            seq.append(nxt)
            p = trans[nxt].copy()
        out.append(sorted(seq))
    return np.array(out)


def s_order_transformer(hist, n, rng, ctx):
    """Repo LottoTransformer trained on balls in drawn order (not sorted), sampled in any order."""
    import torch
    from lotto_nn.model import LottoTransformer
    from lotto_nn.sample import generate_batch
    from lotto_nn.train import train_model

    drawn = ctx["aux"]["drawn"]
    if ctx.get("model") is None or len(drawn) - ctx["trained_on"] >= ctx["retrain"]:
        torch.manual_seed(int(rng.integers(1 << 31)))
        model = LottoTransformer()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_model(model, torch.tensor(drawn, dtype=torch.long), opt, epochs=ctx["epochs"], verbose=False)
        ctx.update(model=model, trained_on=len(drawn))
    torch.manual_seed(int(rng.integers(1 << 31)))
    return np.sort(generate_batch(ctx["model"], num_samples=n, ascending=False), axis=1)


def s_machine_hot(hist, n, rng, ctx):
    """Hot numbers on the machine used last time for this round (the next machine isn't announced)."""
    aux = ctx["aux"]
    same_round = np.flatnonzero(aux["round"] == aux["target_round"])
    machine = aux["machine"][same_round[-1] if len(same_round) else -1]
    rows = np.flatnonzero(aux["machine"] == machine)[-300:]
    if len(rows) < 30:
        rows = np.arange(max(0, len(hist) - 300), len(hist))
    return weighted_tickets(np.bincount(hist[rows].ravel(), minlength=POOL + 1)[1:] + 1.0, n, rng)


def s_bonus_follow(hist, n, rng, ctx):
    """One ticket of the six most recent distinct bonus balls."""
    picks = []
    for b in ctx["aux"]["bonus"][::-1]:
        if b not in picks:
            picks.append(int(b))
        if len(picks) == PICK:
            break
    return np.sort(np.array(picks))[None, :]


STRATEGIES = {
    "random (baseline)": s_random,
    "hot-weighted (last 100)": s_hot_weighted,
    "cold-weighted (last 100)": s_cold_weighted,
    "hot top-6 (last 50)": s_hot_top6,
    "cold top-6 (last 50)": s_cold_top6,
    "most overdue top-6": s_overdue_top6,
    "repeat last draw": s_repeat_last,
    "n-gram grammar (repo)": s_grammar,
    "transformer NN (repo)": s_transformer,
    "early balls (1st/2nd out)": s_early_balls,
    "drawn-order Markov chain": s_order_markov,
    "drawn-order transformer": s_order_transformer,
    "machine-matched hot": s_machine_hot,
    "recent bonus balls": s_bonus_follow,
}
DRAW_ORDER_STRATEGIES = ["early balls (1st/2nd out)", "drawn-order Markov chain", "drawn-order transformer",
                         "machine-matched hot", "recent bonus balls"]


# ---------------------------------------------------------------- ensembles
def votes(tickets: np.ndarray) -> np.ndarray:
    """Share of a strategy's picks that went to each number 1..59 (sums to 1)."""
    return np.bincount(np.asarray(tickets).ravel(), minlength=POOL + 1)[1:] / np.asarray(tickets).size


def perf_weights(trailing: np.ndarray, tau: float) -> np.ndarray:
    """Softmax over each strategy's trailing mean matches (equal weights if no history)."""
    if trailing.size == 0 or np.isnan(trailing).all():
        return np.full(len(trailing), 1 / len(trailing)) if trailing.size else trailing
    z = (np.nan_to_num(trailing, nan=EXP_MEAN) - EXP_MEAN) / tau
    w = np.exp(z - z.max())
    return w / w.sum()


ENSEMBLES = [
    "ENSEMBLE equal-weight top-6",
    "ENSEMBLE equal-weight sampled",
    "ENSEMBLE perf-weighted top-6",
    "ENSEMBLE perf-weighted sampled",
]


def ensemble_tickets(vote_mat: np.ndarray, w: np.ndarray, n: int, rng) -> dict:
    eq = vote_mat.mean(0)
    pw = w @ vote_mat
    return {
        ENSEMBLES[0]: top6(eq, rng),
        ENSEMBLES[1]: weighted_tickets(eq + 1e-9, n, rng),
        ENSEMBLES[2]: top6(pw, rng),
        ENSEMBLES[3]: weighted_tickets(pw + 1e-9, n, rng),
    }


# ---------------------------------------------------------------- driver
def summarize(match_counts: list[int]) -> dict:
    m = np.array(match_counts)
    dist = np.bincount(m, minlength=PICK + 1)
    n = len(m)
    exp = EXPECTED * n
    # pool sparse tails (4,5,6) into one cell for the chi-square test
    obs_c = np.append(dist[:4], dist[4:].sum())
    exp_c = np.append(exp[:4], exp[4:].sum())
    chi2 = float(((obs_c - exp_c) ** 2 / exp_c).sum())
    z = (m.mean() - EXP_MEAN) / math.sqrt(EXP_VAR / n)
    return {
        "tickets": n,
        "mean_matches": float(m.mean()),
        "z_vs_random": float(z),
        "p_better_than_random": norm_sf(z),
        "dist": dist.tolist(),
        "expected_dist": exp.tolist(),
        "pct_3plus": float((m >= 3).mean() * 100),
        "chi2": chi2,
        "chi2_p": chi2_sf(chi2, len(obs_c) - 1),
    }


def aux_until(aux: dict, mask, target_round: int) -> dict:
    out = {k: v[mask] for k, v in aux.items()}
    out["target_round"] = target_round
    return out


def _init_worker(threads: int) -> None:
    try:
        import torch
        torch.set_num_threads(threads)
    except ImportError:
        pass


def _run_seed(args):
    t0 = time.time()
    out = run_backtest(*args)
    out["seconds"] = time.time() - t0
    return out


def run_backtest(draws, dates, test_idx, a, seed, aux):
    """One walk-forward pass over the test rounds with every strategy plus the ensembles."""
    rng = np.random.default_rng(seed)
    names = list(STRATEGIES)
    ctxs = {s: {"epochs": a.epochs, "retrain": a.retrain} for s in names}
    matches = {s: [] for s in names + ENSEMBLES}
    round_mean = {s: [] for s in names + ENSEMBLES}
    pick_corr = []
    for i in test_idx:
        mask = dates < dates[i]  # strictly earlier draw dates only
        hist = draws[mask]
        aux_hist = aux_until(aux, mask, int(aux["round"][i]))
        for c in ctxs.values():
            c["aux"] = aux_hist
        actual = np.zeros(POOL + 1, dtype=bool)
        actual[draws[i]] = True
        trailing = np.array([np.mean(round_mean[s][-a.window:]) if round_mean[s] else np.nan for s in names])
        w = perf_weights(trailing, a.tau)
        tickets = {s: STRATEGIES[s](hist, a.tickets, rng, ctxs[s]) for s in names}
        vote_mat = np.array([votes(tickets[s]) for s in names])
        pick_corr.append(np.corrcoef(vote_mat))
        tickets.update(ensemble_tickets(vote_mat, w, a.tickets, rng))
        for s, ts in tickets.items():
            ms = actual[np.asarray(ts, dtype=int)].sum(1)
            matches[s].extend(ms.tolist())
            round_mean[s].append(float(ms.mean()))
    perf = np.array([round_mean[s] for s in names + ENSEMBLES])
    return {
        "summary": {s: summarize(matches[s]) for s in names + ENSEMBLES},
        "perf_corr": np.corrcoef(perf),
        "pick_corr": np.nanmean(np.array(pick_corr), axis=0),
        "final_weights": dict(zip(names, perf_weights(
            np.array([np.mean(round_mean[s][-a.window:]) for s in names]), a.tau).tolist())),
    }


# ---------------------------------------------------------------- draw-order tests
def uniform_chi2(values: np.ndarray) -> tuple[float, float]:
    f = np.bincount(np.asarray(values).ravel(), minlength=POOL + 1)[1:].astype(float)
    chi2 = float(((f - f.mean()) ** 2 / f.mean()).sum())
    return chi2, chi2_sf(chi2, POOL - 1)


def homogeneity(groups: np.ndarray, sets: np.ndarray, min_rounds: int) -> dict:
    """Chi-square test that every group (machine or ball set) draws the 59 numbers equally often."""
    keep = [g for g in pd.unique(groups) if (groups == g).sum() >= min_rounds]
    table = np.array([np.bincount(sets[groups == g].ravel(), minlength=POOL + 1)[1:] for g in keep], dtype=float)
    exp = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / table.sum()
    chi2 = float(((table - exp) ** 2 / exp).sum())
    dof = (len(keep) - 1) * (POOL - 1)
    return {"groups": [str(g) for g in keep], "chi2": chi2, "dof": dof, "p": chi2_sf(chi2, dof)}


def order_stats(drawn: np.ndarray) -> dict:
    """Statistics of the order balls came out, for comparison against simulated fair draws."""
    x, y = drawn[:, :-1].ravel(), drawn[:, 1:].ravel()
    nxt = np.zeros((len(drawn) - 1, POOL + 1), dtype=bool)
    np.put_along_axis(nxt, drawn[1:], True, axis=1)
    carry = nxt[np.arange(len(drawn) - 1)[:, None], drawn[:-1]]  # [rounds-1, 6]: ball k reappears next round
    return {
        "serial_corr": float(np.corrcoef(x, y)[0, 1]),
        "rising_share": float((y > x).mean()),
        "mean_step": float(np.abs(y - x).mean()),
        "first_ball_carry": float(carry[:, 0].mean()),
        "last_ball_carry": float(carry[:, -1].mean()),
    }


def draw_order_tests(df: pd.DataFrame, sims: int, seed: int) -> dict:
    drawn = df[DRAWN].to_numpy(int)
    n = len(drawn)
    tests = []
    positions = []
    for k, col in enumerate(DRAWN + ["bonus"]):
        v = df[col].to_numpy(int)
        chi2, p = uniform_chi2(v)
        mean = float(v.mean())
        z = (mean - 30) / math.sqrt((POOL ** 2 - 1) / 12 / n)
        label = f"ball {k + 1} out" if col != "bonus" else "bonus ball"
        positions.append({"position": label, "mean": mean, "z_mean": z, "chi2": chi2, "p": p})
        tests.append({"test": f"{label}: every number equally likely", "stat": f"χ² = {chi2:.1f} (58 df)", "p": p})

    obs = order_stats(drawn)
    rng = np.random.default_rng(seed)
    null = {k: [] for k in obs}
    for _ in range(sims):
        fake = np.argsort(rng.random((n, POOL)), axis=1)[:, :PICK] + 1
        for k, v in order_stats(fake).items():
            null[k].append(v)
    labels = {
        "serial_corr": "consecutive balls are correlated",
        "rising_share": "the next ball is higher more or less than half the time",
        "mean_step": "consecutive balls are closer or further apart than chance",
        "first_ball_carry": "the first ball out reappears in the next round",
        "last_ball_carry": "the last ball out reappears in the next round",
    }
    sim = {}
    for k, v in obs.items():
        arr = np.array(null[k])
        p = float(((np.abs(arr - arr.mean()) >= abs(v - arr.mean())).sum() + 1) / (len(arr) + 1))
        sim[k] = {"observed": v, "expected": float(arr.mean()), "p": p}
        tests.append({"test": labels[k], "stat": f"{v:.4f} vs {arr.mean():.4f} by chance", "p": p})

    sets = df[COLS].to_numpy(int)
    mach = homogeneity(df.machine.astype(str).str.strip().to_numpy(), sets, 100)
    bset = homogeneity(df.ball_set.astype(str).str.strip().to_numpy(), sets, 80)
    tests.append({"test": f"machines favour different numbers ({len(mach['groups'])} machines)",
                  "stat": f"χ² = {mach['chi2']:.0f} ({mach['dof']} df)", "p": mach["p"]})
    tests.append({"test": f"ball sets favour different numbers ({len(bset['groups'])} sets)",
                  "stat": f"χ² = {bset['chi2']:.0f} ({bset['dof']} df)", "p": bset["p"]})
    for t in tests:
        print(f"  p={t['p']:.3f}  {t['test']}: {t['stat']}")
    return {"rounds": n, "sims": sims, "tests": tests, "positions": positions, "sequence": sim,
            "machines": mach, "ball_sets": bset, "bonferroni_alpha": 0.05 / len(tests)}


def next_draw_date(last: date) -> date:
    d = last + timedelta(days=1)
    while d.weekday() not in (2, 5):  # Wednesday, Saturday
        d += timedelta(days=1)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/uk_lotto_full.csv")
    ap.add_argument("--split-date", default="2022-12-31")
    ap.add_argument("--tickets", type=int, default=20, help="tickets per round for sampling strategies")
    ap.add_argument("--epochs", type=int, default=30, help="transformer training epochs")
    ap.add_argument("--retrain", type=int, default=26, help="retrain transformer every N rounds")
    ap.add_argument("--window", type=int, default=52, help="trailing rounds used for performance weights")
    ap.add_argument("--tau", type=float, default=0.05, help="softmax temperature for performance weights")
    ap.add_argument("--predictions", type=int, default=5)
    ap.add_argument("--sims", type=int, default=2000, help="simulated fair histories for draw-order p-values")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=0, help="seeds run in parallel (default: one per CPU core)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    df = df[df.draw_no >= FIRST_59_DRAW].sort_values(["draw_no", "round"]).reset_index(drop=True)
    draws = df[COLS].to_numpy(int)
    dates = pd.to_datetime(df.date).dt.date.to_numpy()
    aux = {"drawn": df[DRAWN].to_numpy(int), "bonus": df.bonus.to_numpy(int),
           "machine": df.machine.astype(str).str.strip().to_numpy(), "round": df["round"].to_numpy(int)}
    split = date.fromisoformat(a.split_date)
    test_idx = [i for i, d in enumerate(dates) if d > split]
    print(f"1-59 era: {len(df)} rounds {dates[0]} -> {dates[-1]}; backtesting {len(test_idx)} rounds after {split}")

    names = list(STRATEGIES) + ENSEMBLES
    # seeds are independent, so they run in parallel processes, sharing the CPU cores between them
    cores = os.cpu_count() or 1
    workers = max(1, min(len(a.seeds), a.workers or cores))
    threads = max(1, cores // workers)
    jobs = [(draws, dates, test_idx, a, seed, aux) for seed in a.seeds]
    t_all = time.time()
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(threads,)) as ex:
            runs = list(ex.map(_run_seed, jobs))
    else:
        _init_worker(threads)
        runs = [_run_seed(j) for j in jobs]
    print(f"{len(a.seeds)} seeds on {workers} worker(s) x {threads} thread(s): {time.time() - t_all:.0f}s")
    for seed, run in zip(a.seeds, runs):
        print(f"\nseed {seed} ({run['seconds']:.0f}s)")
        for s in names:
            r = run["summary"][s]
            print(f"  {s:32s} tickets={r['tickets']:6d} mean={r['mean_matches']:.4f} "
                  f"z={r['z_vs_random']:+.2f} 3+={r['pct_3plus']:.2f}% chi2_p={r['chi2_p']:.3f}")

    zs = {s: [r["summary"][s]["z_vs_random"] for r in runs] for s in names}
    means = {s: [r["summary"][s]["mean_matches"] for r in runs] for s in names}
    across = {s: {"z_mean": float(np.mean(zs[s])), "z_min": float(np.min(zs[s])), "z_max": float(np.max(zs[s])),
                  "mean_matches": float(np.mean(means[s])),
                  "pct_3plus": float(np.mean([r["summary"][s]["pct_3plus"] for r in runs]))} for s in names}
    perf_corr = np.mean([r["perf_corr"] for r in runs], axis=0)
    pick_corr = np.mean([r["pick_corr"] for r in runs], axis=0)
    print("\nacross seeds (z mean [min, max]):")
    for s in names:
        print(f"  {s:32s} {across[s]['z_mean']:+.2f} [{across[s]['z_min']:+.2f}, {across[s]['z_max']:+.2f}]")

    # fairness check: are numbers uniformly drawn across the 1-59 era?
    f = freq(draws)
    uni_chi2 = float(((f - f.mean()) ** 2 / f.mean()).sum())
    print("draw-order tests:")
    draw_order = draw_order_tests(df, a.sims, a.seeds[0])
    fairness = {"counts": f.astype(int).tolist(), "chi2": uni_chi2, "dof": POOL - 1, "p": chi2_sf(uni_chi2, POOL - 1),
                "sum_mean": float(draws.sum(1).mean()), "odd_mean": float((draws % 2).sum(1).mean())}
    print(f"uniformity chi2={uni_chi2:.1f} (dof 58) p={fairness['p']:.3f}")

    # next-draw predictions: every strategy fitted on all history, combined by both ensembles
    rng = np.random.default_rng(a.seeds[0])
    nd = next_draw_date(dates[-1])
    base = list(STRATEGIES)
    full_aux = aux_until(aux, np.ones(len(draws), dtype=bool), 1)
    tickets = {s: STRATEGIES[s](draws, max(a.predictions, a.tickets), rng,
                                {"epochs": a.epochs, "retrain": a.retrain, "aux": full_aux})
               for s in base}
    vote_mat = np.array([votes(tickets[s]) for s in base])
    w = np.array([runs[0]["final_weights"][s] for s in base])
    tickets.update(ensemble_tickets(vote_mat, w, a.predictions, rng))
    preds = {s: [sorted(int(x) for x in t) for t in tickets[s][:a.predictions]] for s in names}
    consensus = vote_mat.mean(0)
    perf_cons = w @ vote_mat
    print(f"\nnext draw {nd}")
    for s in ENSEMBLES:
        print(f"  {s:32s} {preds[s]}")

    os.makedirs(a.out, exist_ok=True)
    payload = {
        "source": "lottery.merseyworld.com full UK Lotto history",
        "era": {"first_draw": int(df.draw_no.min()), "first_date": str(dates[0]), "last_draw": int(df.draw_no.max()),
                "last_date": str(dates[-1]), "rounds": len(df)},
        "split_date": a.split_date, "test_rounds": len(test_idx), "tickets_per_round": a.tickets,
        "seeds": a.seeds, "window": a.window, "tau": a.tau,
        "expected": {"mean": EXP_MEAN, "dist": EXPECTED.tolist(), "pct_3plus": float(EXPECTED[3:].sum() * 100)},
        "strategies": names,
        "backtest": runs[0]["summary"], "across_seeds": across,
        "perf_corr": perf_corr.round(4).tolist(), "pick_corr": pick_corr.round(4).tolist(),
        "pick_corr_names": base,
        "final_weights": runs[0]["final_weights"],
        "fairness": fairness,
        "draw_order": draw_order,
        "draw_order_strategies": DRAW_ORDER_STRATEGIES,
        "latest": df.tail(2)[["draw_no", "date", "round", *COLS, "bonus"]].to_dict("records"),
        "next_draw_date": str(nd), "predictions": preds,
        "consensus_scores": {"equal": consensus.round(5).tolist(), "perf": perf_cons.round(5).tolist()},
    }
    with open(os.path.join(a.out, "walk_forward.json"), "w") as fh:
        json.dump(payload, fh, indent=1, default=int)
    print(f"wrote {a.out}/walk_forward.json")


if __name__ == "__main__":
    main()
