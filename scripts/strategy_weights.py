"""Every backtest strategy's number weights at every real draw night, for the nested Monte Carlo.

The outer loop of sim/nested_mc.cpp resets to the real history at each draw night. This script gives it,
for each of those nights, what every strategy of scripts/walk_forward_backtest.py backs from the real draws
before that night only: the share of its tickets that went to each number 1..59 (`--tickets` tickets for the
strategies that sample, the one ticket for the top-6 ones). Two more rows are added per night:
  "hindsight as reality"  each number's count over the whole 59-ball era so far, plus one;
  "consensus (equal)"     the mean of the strategies' shares (the backtest's equal-weight ensemble).
A last block, dated the next draw, is fitted on every real draw, so the simulator can name the next 15.

    python scripts/strategy_weights.py --start 2017-01-01 --out results/cache/strategy_weights.csv
"""
from __future__ import annotations

import argparse, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from walk_forward_backtest import COLS, DRAWN, FIRST_59_DRAW, POOL, STRATEGIES, aux_until, next_draw_date, votes  # noqa: E402

HINDSIGHT, CONSENSUS = "hindsight as reality", "consensus (equal)"


def load(csv: str):
    df = pd.read_csv(csv)
    df = df[df.draw_no >= FIRST_59_DRAW].sort_values(["draw_no", "round"]).reset_index(drop=True)
    draws = df[COLS].to_numpy(int)
    dates = pd.to_datetime(df.date).dt.date.to_numpy()
    aux = {"drawn": df[DRAWN].to_numpy(int), "bonus": df.bonus.to_numpy(int),
           "machine": df.machine.astype(str).str.strip().to_numpy(), "round": df["round"].to_numpy(int)}
    return draws, dates, aux


def reset_nights(dates, start: date) -> list[date]:
    nights = sorted({d for d in dates if d >= start})
    return nights + [next_draw_date(dates[-1])]


def run_strategy(args):
    """One strategy walked forward over every reset night (strategies keep state, e.g. a trained model)."""
    name, draws, dates, aux, nights, tickets, seed, epochs, retrain = args
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    t0 = time.time()
    rng = np.random.default_rng([seed, list(STRATEGIES).index(name)])
    ctx = {"epochs": epochs, "retrain": retrain}
    out = np.zeros((len(nights), POOL))
    for n, night in enumerate(nights):
        mask = dates < night
        ctx["aux"] = aux_until(aux, mask, 1)  # a reset is the night's first round
        out[n] = votes(STRATEGIES[name](draws[mask], tickets, rng, ctx))
    return name, out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/uk_lotto_full.csv")
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--tickets", type=int, default=200, help="tickets per night for the strategies that sample")
    ap.add_argument("--epochs", type=int, default=30, help="transformer training epochs")
    ap.add_argument("--retrain", type=int, default=26, help="retrain the transformers every N rounds")
    ap.add_argument("--strategies", nargs="*", default=list(STRATEGIES), help="subset of strategies (default: all)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="results/cache/strategy_weights.csv")
    a = ap.parse_args()

    draws, dates, aux = load(a.csv)
    nights = reset_nights(dates, date.fromisoformat(a.start))
    print(f"{len(nights) - 1} reset nights from {nights[0]} and the next draw {nights[-1]}; {len(a.strategies)} strategies")
    jobs = [(s, draws, dates, aux, nights, a.tickets, a.seed, a.epochs, a.retrain) for s in a.strategies]
    # the transformers are by far the slowest, so they start first
    jobs.sort(key=lambda j: "transformer" not in j[0])
    weights = {}
    with ProcessPoolExecutor(max(1, a.workers or os.cpu_count() or 1)) as ex:
        for name, w, secs in ex.map(run_strategy, jobs):
            weights[name] = w
            print(f"  {name:28s} {secs:6.0f}s")
    names = [s for s in STRATEGIES if s in weights]
    weights[CONSENSUS] = np.mean([weights[s] for s in names], axis=0)
    counts = np.ones((len(nights), POOL))
    for n, night in enumerate(nights):
        counts[n] += np.bincount(draws[dates < night].ravel(), minlength=POOL + 1)[1:]
    weights[HINDSIGHT] = counts / counts.sum(1, keepdims=True)

    rows = [[str(night), name, *np.round(weights[name][n], 6)] for n, night in enumerate(nights)
            for name in [HINDSIGHT, *names, CONSENSUS]]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pd.DataFrame(rows, columns=["date", "strategy", *[f"w{i}" for i in range(1, POOL + 1)]]).to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
