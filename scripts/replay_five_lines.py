"""Replay the five combined lines (and the five popularity-aware lines) for past draws.

For each draw since --since, every investigation's next-draw picks are rebuilt from the draws before
it only, exactly as the routine builds them after each draw (same functions, same seeds), then
combined by scripts/build_report.py and scored against the real rounds. The studies' tests and
backtests don't feed into the picks, so only the pick-making steps are run. Results go to
results/replay_five_lines.json.

    python scripts/replay_five_lines.py --since 2026-07-13 --workers 4
"""
import argparse, importlib.util, json, os, sys, time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SEED = 2026  # the first seed every study uses for its next-draw picks


def investigation_picks(hist, mods) -> dict:
    """The payload build_report.pick_sources reads, built from `hist` (all draws before the target)."""
    import numpy as np
    import pandas as pd
    wf, pr, sp, by, ml, rl = (mods[k] for k in ("wf", "pr", "sp", "by", "ml", "rl"))
    e59 = hist[hist.draw_no >= 2066].sort_values(["draw_no", "round"]).reset_index(drop=True)
    sets = e59[["n1", "n2", "n3", "n4", "n5", "n6"]].to_numpy(int)
    payload = {}

    # strategy backtest: every strategy fitted on all history (walk_forward_backtest.main)
    aux = {"drawn": e59[wf.DRAWN].to_numpy(int), "bonus": e59.bonus.to_numpy(int),
           "machine": e59.machine.astype(str).str.strip().to_numpy(), "round": e59["round"].to_numpy(int)}
    full_aux = wf.aux_until(aux, np.ones(len(e59), dtype=bool), 1)
    rng = np.random.default_rng(SEED)
    tickets = {s: fn(sets, 20, rng, {"epochs": 30, "retrain": 26, "aux": full_aux}) for s, fn in wf.STRATEGIES.items()}
    payload["predictions"] = {s: [sorted(int(x) for x in t) for t in tk[:5]] for s, tk in tickets.items()}

    # prime radix: conditional probability and copula per base (prime_radix_study.backtest)
    rng = np.random.default_rng(SEED)
    picks = {}
    for p in pr.FIRST_10_PRIMES:
        picks[f"conditional probability, base {p}"] = [sorted(int(x) for x in pr.cond_prob_tickets(sets, p, 1, rng)[0])]
        picks[f"conditional copula, base {p}"] = [sorted(int(x) for x in pr.copula_tickets(sets, p, 1, rng)[0])]
    payload["prime_radix"] = {"backtest": {"picks": picks}}

    # Fourier and wavelets (spectral_study.backtest)
    rng = np.random.default_rng(SEED)
    payload["spectral"] = {"backtest": {"picks": {s: [sorted(int(x) for x in t) for t in fn(sets, 3, rng)]
                                                  for s, fn in sp.STRATEGIES.items() if not s.startswith("random")}}}

    # Bayesian (bayesian_study.backtest)
    machine = e59.machine.astype(str).str.strip().to_numpy()
    rng = np.random.default_rng(SEED)
    payload["bayes"] = {"backtest": {"picks": {s: [sorted(int(x) for x in t) for t in ts[:3]] for s, ts in
                                               by.strategies(sets, machine, machine[-1], 59, 3, rng).items()
                                               if not s.startswith("random")}}}

    # machine learning: the three models fitted on every round, predicting the next (ml_ensemble_study.era_study)
    X, Y = ml.build_features(e59, 59)
    Xn, _ = ml.build_features(pd.concat([e59, e59.iloc[[-1]]], ignore_index=True), 59)
    tr = np.arange(ml.WARMUP, len(e59))
    rng = np.random.default_rng(SEED)
    probs = []
    for m in ml.make_models(SEED).values():
        m.fit(X[tr].reshape(-1, X.shape[-1]), Y[tr].ravel())
        probs.append(m.predict_proba(Xn[-1])[:, 1])
    p = np.mean(probs, axis=0)
    payload["ml"] = {"eras": [{"picks": {"top6": sorted(int(i) + 1 for i in np.argsort(-p)[:6]),
                                         "sampled": [sorted(int(x) for x in t) for t in ml.sample_tickets(p, 4, rng)]}}]}

    # reinforcement learning: agents learn online through every round, then pick (rl_study.run_era)
    with open(os.devnull, "w") as null:
        old, sys.stdout = sys.stdout, null
        try:
            era = rl.run_era(e59, 59, "2022-12-31", "replay", [SEED], 20, True)
        finally:
            sys.stdout = old
    payload["rl"] = {"backtest": {"picks": era["picks"]}}

    # every strategy family played on prime-base digits (radix_strategies_study.next_picks)
    if "rs" in mods:
        payload["radix_strat"] = {"backtest": {"picks": mods["rs"].next_picks(mods["rs"].History(e59), SEED)}}
    payload["latest"] = e59.tail(2)[["draw_no", "date", "round", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"]].to_dict("records")
    return payload


def wheel_entry(br, score, rounds, sl) -> dict:
    """The wheel option for a past draw: its 12 candidates, how many came up in each round, its lines' scores."""
    cand, lines = br.wheel_lines(score)
    return {"candidates": cand, "in_candidates": [len(set(cand) & set(r["numbers"])) for r in rounds],
            "lines": [sl.score_line(l, rounds) for l in lines]}


def replay_one(args):
    date, csv = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import pandas as pd
    import torch
    torch.set_num_threads(1)
    mods = {"wf": _load("walk_forward_backtest"), "pr": _load("prime_radix_study"), "sp": _load("spectral_study"),
            "by": _load("bayesian_study"), "ml": _load("ml_ensemble_study"), "rl": _load("rl_study"),
            "rs": _load("radix_strategies_study")}
    br, ps, sl = _load("build_report"), _load("popularity_study"), _load("score_live")
    t0 = time.time()
    df = pd.read_csv(csv)
    hist = df[df.date < date]
    payload = investigation_picks(hist, mods)
    seed = int(date.replace("-", ""))
    combined = br.combine(br.pick_sources(payload), seed)
    br.add_popularity_lines(combined, ps.fit(hist), payload["latest"], seed)
    rounds = sl.draw_rounds(df, date)
    out = {"date": date, "draw_no": rounds[0]["draw_no"], "rounds": rounds,
           "combined": [sl.score_line(l, rounds) for l in combined["lines"]],
           "popularity_lines": [sl.score_line(l, rounds) for l in combined["popularity_lines"]],
           "wheel": wheel_entry(br, combined["score"], rounds, sl),
           "popularity": combined["popularity"], "score": combined["score"],
           "window": hist[hist.draw_no >= 2066].sort_values(["draw_no", "round"])
                     [["n1", "n2", "n3", "n4", "n5", "n6"]].tail(104).to_numpy().tolist(),
           "seconds": round(time.time() - t0)}
    best = max(m for s in out["combined"] for m in s["matches"])
    print(f"{date}: best combined line {best} matches ({out['seconds']}s)", flush=True)
    return out


def line_seed(line) -> int:
    """A draw's six numbers as one integer, e.g. 3 8 11 38 42 44 -> 30811384244."""
    return int("".join(f"{x:02d}" for x in sorted(line)))


def seed_study(draws: list, n_random: int = 1000) -> dict:
    """Does the seed for lines 2-5 matter? Redraw them from each draw's stored support with other
    seeds (the date, the last draw, the first draw of a rolling window) and with many random seeds."""
    import numpy as np
    br = _load("build_report")

    def mean_matches(seed_of):
        m = [len(set(l) & set(r["numbers"])) for d in draws
             for l in br.lines_from_score(d["score"], seed_of(d))[1:] for r in d["rounds"]]
        return float(np.mean(m)), int(sum(x >= 3 for x in m)), len(m)

    variants = {"date (current)": lambda d: int(d["date"].replace("-", "")),
                "last draw's numbers": lambda d: line_seed(d["window"][-1])}
    for w in (10, 26, 52, 104):
        variants[f"first draw of the last {w} rounds"] = (lambda w: lambda d: line_seed(d["window"][-w]))(w)
    null = np.array([mean_matches(lambda d, s=s: s * 7919 + int(d["date"].replace("-", "")))[0]
                     for s in range(n_random)])
    rows = []
    for name, fn in variants.items():
        m, hits, n = mean_matches(fn)
        rows.append({"seed": name, "mean": m, "hits3": hits, "line_rounds": n, "percentile": float((null < m).mean() * 100)})
    line1 = [len(set(d["combined"][0]["line"]) & set(r["numbers"])) for d in draws for r in d["rounds"]]
    return {"variants": rows, "random": {"n": n_random, "mean": float(null.mean()),
                                         "p5": float(np.percentile(null, 5)), "p95": float(np.percentile(null, 95))},
            "line1": {"mean": float(np.mean(line1)), "line_rounds": len(line1)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/uk_lotto_full.csv")
    ap.add_argument("--since", default="2026-07-13")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/replay_five_lines.json")
    a = ap.parse_args()
    import pandas as pd
    sl = _load("score_live")
    dates = sorted(d for d in pd.read_csv(a.csv).date.unique() if d >= a.since)
    print(f"replaying {len(dates)} draws from {dates[0]} to {dates[-1]} with {a.workers} workers", flush=True)
    with Pool(a.workers) as pool:
        draws = pool.map(replay_one, [(d, a.csv) for d in dates], chunksize=1)
    summary = sl.summarise([{**d, "strategies": {}, "wheel_lines": d["wheel"]["lines"]} for d in draws])
    del summary["strategies"]
    seeds = seed_study(draws)
    for d in draws:
        del d["window"]  # only needed for the seed study
    with open(a.out, "w") as fh:
        json.dump({"since": a.since, "draws": draws, "summary": summary, "seed_study": seeds}, fh, indent=1)
    for r in seeds["variants"]:
        print(f"  lines 2-5 seeded by {r['seed']:32s} mean {r['mean']:.3f}, 3+ {r['hits3']}, "
              f"{r['percentile']:.0f}th percentile of {seeds['random']['n']} random seeds")
    s = summary["combined"]
    print(f"combined lines: {s['line_rounds']} line-rounds, counts {s['counts'][:4]}, mean {s['mean']:.3f} "
          f"(chance {summary['chance_mean']:.3f}), 3+ = {s['prize']}")
    s = summary["popularity_lines"]
    print(f"popularity-aware: {s['line_rounds']} line-rounds, counts {s['counts'][:4]}, mean {s['mean']:.3f}, 3+ = {s['prize']}")


if __name__ == "__main__":
    main()
