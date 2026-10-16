"""Inject results/walk_forward.json (plus results/grammar.json, results/prime_radix.json,
results/spectral.json, results/ml_ensemble.json, results/bayesian.json, results/rl.json and
results/live_scores.json, if present) into
results/report_template.html -> results/report.html, and combine every investigation's
next-draw picks into five lines (also written to results/combined_picks.json)."""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import popularity_study as ps  # noqa: E402

POOL, PICK = 59, 6
EXTRA = [("grammar", "results/grammar.json"), ("prime_radix", "results/prime_radix.json"),
         ("spectral", "results/spectral.json"), ("ml", "results/ml_ensemble.json"),
         ("bayes", "results/bayesian.json"),
         ("rl", "results/rl.json"), ("live", "results/live_scores.json"),
         ("popularity", "results/popularity.json"), ("replay", "results/replay_five_lines.json"),
         ("radix_strat", "results/radix_strategies.json")]


def pick_sources(payload: dict) -> list:
    """(investigation, [lines]) for every investigation that made next-draw picks.
    Random baselines and the backtest's own ensembles are left out (they would double count)."""
    out = []
    wf = [line for s, lines in payload.get("predictions", {}).items()
          if not s.startswith(("random", "ENSEMBLE")) for line in lines]
    if wf:
        out.append(("Strategy backtest", wf))
    for keys, label in [(("prime_radix", "radix_strat"), "Prime radix"), (("spectral",), "Fourier and wavelets"),
                        (("bayes",), "Bayesian"), (("rl",), "Reinforcement learning")]:
        # the classic strategies played on prime-base digits vote within the prime-radix investigation
        lines = [line for key in keys for s, ls in ((payload.get(key) or {}).get("backtest", {}).get("picks") or {}).items()
                 if not s.startswith("random") for line in ls]
        if lines:
            out.append((label, lines))
    ml = (payload.get("ml") or {}).get("eras", [{}])[0].get("picks")
    if ml:
        out.append(("Machine learning", [ml["top6"]] + ml["sampled"]))
    return out


def lines_from_score(score, seed: int, n_lines: int = 5) -> list:
    """Line 1 is the six numbers with the most support (the seed only breaks exact ties); the other
    lines are drawn in proportion to support, using the seed, from numbers no earlier line uses.

    Every line matches 36/59 numbers on average whatever it holds, but lines that share numbers win
    and lose together. With no number shared, the five lines cover 30 numbers and the chance that at
    least one of them matches 2+ (or 3+, a prize) in a round is as high as five lines can make it."""
    score = np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    top = np.argsort(-(score + rng.random(POOL) * 1e-9))[:PICK] + 1
    lines = [sorted(int(x) for x in top)]
    p = score + score.mean() * 0.2  # every number keeps some chance
    free = np.ones(POOL, dtype=bool)
    free[top - 1] = False
    while len(lines) < n_lines:
        q = p * free
        line = sorted(int(x) for x in rng.choice(np.arange(1, POOL + 1), PICK, replace=False, p=q / q.sum()))
        free[np.array(line) - 1] = False
        lines.append(line)
    return lines


def coverage(lines: list, n: int = 1_000_000, seed: int = 0, chunk: int = 100_000) -> dict:
    """Chance, per round, that at least one of the lines matches 2+ and 3+ numbers (simulated draws)."""
    rng = np.random.default_rng(seed)
    c2 = c3 = 0
    for _ in range(n // chunk):
        hits = np.zeros((chunk, POOL + 1), dtype=bool)
        np.put_along_axis(hits, np.argsort(rng.random((chunk, POOL)), axis=1)[:, :PICK] + 1, True, axis=1)
        best = np.max([hits[:, l].sum(1) for l in lines], axis=0)
        c2 += int((best >= 2).sum())
        c3 += int((best >= 3).sum())
    return {"distinct": len({x for l in lines for x in l}), "p2": c2 / n, "p3": c3 / n}


def combine(sources: list, seed: int, n_lines: int = 5) -> dict:
    """Equal vote per investigation, spread evenly over its lines. Line 1 is the six numbers
    with the most support; the other lines are drawn in proportion to support."""
    score = np.zeros(POOL)
    support = np.zeros(POOL, dtype=int)
    for _, lines in sources:
        v = np.zeros(POOL)
        for line in lines:
            v[np.array(line) - 1] += 1
        score += v / v.sum()
        support += (v > 0).astype(int)
    score /= len(sources)
    lines = lines_from_score(score, seed, n_lines)
    return {"lines": lines, "score": score.round(5).tolist(), "support": support.tolist(),
            "sources": [{"investigation": s, "lines": len(ls)} for s, ls in sources]}


def add_popularity_lines(combined: dict, model: dict, latest: list, seed: int) -> None:
    """Five more lines drawn from the same combined support, keeping the ones other players are least
    likely to have picked, and each line's popularity and jackpot value for both sets."""
    pop = ps.Popularity(model["beta"], model["lam"])
    prev = sorted({int(r[f"n{i}"]) for r in latest for i in range(1, 7)})
    lines = ps.choose_lines(combined["score"], pop, prev, seed=seed)
    rate = lambda ls: [{"popularity": round(pop.popularity(l, prev), 3), "value": round(pop.value(l, prev), 3)} for l in ls]
    combined["popularity_lines"] = lines
    combined["popularity"] = {"combined": rate(combined["lines"]), "aware": rate(lines),
                              "lam": model["lam"], "typical_share": pop.typical_share}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "results/walk_forward.json"
    payload = json.load(open(src))
    for key, path in EXTRA:
        if os.path.exists(path):
            payload[key] = json.load(open(path))
    sources = pick_sources(payload)
    if sources:
        seed = int(payload.get("next_draw_date", "2026-01-01").replace("-", ""))
        payload["combined"] = combine(sources, seed)
        payload["combined"]["next_draw_date"] = payload.get("next_draw_date")
        if payload.get("popularity"):
            add_popularity_lines(payload["combined"], payload["popularity"], payload.get("latest") or [], seed)
        C = payload["combined"]
        C["coverage"] = {"combined": coverage(C["lines"])}
        if C.get("popularity_lines"):
            C["coverage"]["popularity"] = coverage(C["popularity_lines"])
        with open("results/combined_picks.json", "w") as fh:
            json.dump(payload["combined"], fh, indent=1)
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = open("results/report_template.html").read().replace("/*DATA*/null", data)
    open("results/report.html", "w").write(html)
    print("wrote results/report.html" + (f"; five lines for {payload['combined']['next_draw_date']}: "
                                         f"{payload['combined']['lines']}" if sources else ""))


if __name__ == "__main__":
    main()
