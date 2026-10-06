"""Inject results/walk_forward.json (plus results/grammar.json, results/prime_radix.json,
results/spectral.json, results/ml_ensemble.json and results/bayesian.json, if present) into
results/report_template.html -> results/report.html, and combine every investigation's
next-draw picks into five lines (also written to results/combined_picks.json)."""
import json, os, sys

import numpy as np

POOL, PICK = 59, 6
EXTRA = [("grammar", "results/grammar.json"), ("prime_radix", "results/prime_radix.json"),
         ("spectral", "results/spectral.json"), ("ml", "results/ml_ensemble.json"),
         ("bayes", "results/bayesian.json")]


def pick_sources(payload: dict) -> list:
    """(investigation, [lines]) for every investigation that made next-draw picks.
    Random baselines and the backtest's own ensembles are left out (they would double count)."""
    out = []
    wf = [line for s, lines in payload.get("predictions", {}).items()
          if not s.startswith(("random", "ENSEMBLE")) for line in lines]
    if wf:
        out.append(("Strategy backtest", wf))
    for key, label in [("prime_radix", "Prime radix"), ("spectral", "Fourier and wavelets"), ("bayes", "Bayesian")]:
        picks = (payload.get(key) or {}).get("backtest", {}).get("picks") or {}
        lines = [line for s, ls in picks.items() if not s.startswith("random") for line in ls]
        if lines:
            out.append((label, lines))
    ml = (payload.get("ml") or {}).get("eras", [{}])[0].get("picks")
    if ml:
        out.append(("Machine learning", [ml["top6"]] + ml["sampled"]))
    return out


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
    rng = np.random.default_rng(seed)
    top = np.argsort(-(score + rng.random(POOL) * 1e-9))[:PICK] + 1
    lines = [sorted(int(x) for x in top)]
    p = score + score.mean() * 0.2  # every number keeps some chance
    p /= p.sum()
    while len(lines) < n_lines:
        line = sorted(int(x) for x in rng.choice(np.arange(1, POOL + 1), PICK, replace=False, p=p))
        if line not in lines:
            lines.append(line)
    return {"lines": lines, "score": score.round(5).tolist(), "support": support.tolist(),
            "sources": [{"investigation": s, "lines": len(ls)} for s, ls in sources]}


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
        with open("results/combined_picks.json", "w") as fh:
            json.dump(payload["combined"], fh, indent=1)
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = open("results/report_template.html").read().replace("/*DATA*/null", data)
    open("results/report.html", "w").write(html)
    print("wrote results/report.html" + (f"; five lines for {payload['combined']['next_draw_date']}: "
                                         f"{payload['combined']['lines']}" if sources else ""))


if __name__ == "__main__":
    main()
