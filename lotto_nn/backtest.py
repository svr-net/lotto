"""Statistical backtesting suite comparing generated draws to historical draws."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

MAX_NUM = 59
PICK = 6


def frequency_distribution(data_np: np.ndarray, max_num: int = MAX_NUM) -> np.ndarray:
    """Return counts of each number 1..max_num appearing in data_np ([N,6])."""
    flat = data_np.flatten()
    freq = np.bincount(flat, minlength=max_num + 1)
    return freq[1:]


def chi2_sf(x: float, dof: int) -> float:
    """Survival function (p-value) of the chi-square distribution with ``dof`` degrees of freedom."""
    a, x = dof / 2.0, x / 2.0
    if x <= 0:
        return 1.0
    if x < a + 1:  # series for the lower incomplete gamma, then complement
        term = total = 1.0 / a
        n = a
        for _ in range(500):
            n += 1
            term *= x / n
            total += term
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a)))
    b, c, d = x + 1 - a, 1e300, 1.0 / (x + 1 - a)  # continued fraction for the upper incomplete gamma
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1.0 / d if abs(d) > 1e-300 else 1e300
        c = b + an / c if abs(c) > 1e-300 else 1e-300
        h *= d * c
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi_square_test(hist_freq: np.ndarray, gen_freq: np.ndarray) -> dict:
    """Two-sample chi-square test that generated and historical number frequencies
    come from the same distribution.

    Both samples are compared against expected counts from the pooled
    frequencies, so samples of different sizes (e.g. 5000 generated tickets vs
    1181 historical draws) are scaled correctly. Numbers that never appear in
    either sample are dropped. Returns ``{"chi2", "dof", "p"}``.
    """
    table = np.vstack([np.asarray(hist_freq, dtype=float), np.asarray(gen_freq, dtype=float)])
    table = table[:, table.sum(axis=0) > 0]
    totals = table.sum(axis=1, keepdims=True)
    if table.shape[1] < 2 or (totals == 0).any():
        return {"chi2": 0.0, "dof": 0, "p": 1.0}
    expected = totals * table.sum(axis=0, keepdims=True) / table.sum()
    chi2 = float(((table - expected) ** 2 / expected).sum())
    dof = table.shape[1] - 1
    return {"chi2": chi2, "dof": dof, "p": chi2_sf(chi2, dof)}


def chi_square_stat(hist_freq: np.ndarray, gen_freq: np.ndarray) -> float:
    """Chi-square statistic of ``chi_square_test`` (sample sizes are normalized)."""
    return chi_square_test(hist_freq, gen_freq)["chi2"]


def pair_counts(data_np: np.ndarray) -> Counter:
    """Count co-occurrence of number pairs across draws."""
    c: Counter = Counter()
    for row in data_np:
        row_sorted = sorted(row)
        for i in range(len(row_sorted)):
            for j in range(i + 1, len(row_sorted)):
                pair = (row_sorted[i], row_sorted[j])
                c[pair] += 1
    return c


def top_k_pairs(counter: Counter, k: int = 20):
    return counter.most_common(k)


def sum_stats(data_np: np.ndarray) -> dict:
    sums = data_np.sum(axis=1)
    return {"mean": float(sums.mean()), "std": float(sums.std())}


def odd_even_stats(data_np: np.ndarray) -> dict:
    odd_count = (data_np % 2).sum(axis=1)
    return {"mean": float(odd_count.mean()), "std": float(odd_count.std())}


def spacing(row: np.ndarray) -> np.ndarray:
    row_sorted = np.sort(row)
    return np.diff(row_sorted)


def spacing_stats(data_np: np.ndarray) -> dict:
    spacings = np.vstack([spacing(r) for r in data_np])
    return {"mean": float(spacings.mean()), "std": float(spacings.std())}


def positional_frequency(data_np: np.ndarray, max_num: int = MAX_NUM) -> np.ndarray:
    """Per-position (column) frequency of each number 1..max_num.

    Returns array of shape [seq_len, max_num].
    """
    seq_len = data_np.shape[1]
    pos_freq = np.zeros((seq_len, max_num + 1), dtype=int)
    for row in data_np:
        for pos in range(seq_len):
            num = row[pos]
            pos_freq[pos, num] += 1
    return pos_freq[:, 1:]


def hits(ticket, draw) -> int:
    return len(set(ticket) & set(draw))


def match_matrix(tickets_np: np.ndarray, draws_np: np.ndarray, max_num: int = MAX_NUM) -> np.ndarray:
    """[n_tickets, n_draws] count of numbers each ticket shares with each draw."""

    def one_hot(a):
        a = np.asarray(a, dtype=int)
        m = np.zeros((len(a), max_num + 1), dtype=np.int32)
        np.put_along_axis(m, a, 1, axis=1)
        return m

    return one_hot(tickets_np) @ one_hot(draws_np).T


def holdout_hits(tickets_np: np.ndarray, test_draws_np: np.ndarray, max_num: int = MAX_NUM) -> dict:
    """Score tickets from a model fitted before the test period against the test draws.

    Every ticket is checked against every held-out draw. By chance a ticket
    matches PICK * PICK / max_num numbers on average (36/59 for Lotto), whatever
    numbers it holds. The per-draw average over all tickets is independent from
    draw to draw, so the z-score uses its spread across draws. A positive z
    means the tickets matched more than chance.

    Returns mean/expected matches, z, one-sided p, the distribution of match
    counts (0..PICK) over all ticket-draw pairs, the best single match, and the
    number of test draws.
    """
    m = match_matrix(tickets_np, test_draws_np, max_num)
    expected = PICK * PICK / max_num
    per_draw = m.mean(axis=0)
    n = len(per_draw)
    sd = per_draw.std(ddof=1) if n > 1 else 0.0
    z = float((per_draw.mean() - expected) / (sd / math.sqrt(n))) if sd > 0 else 0.0
    return {
        "mean": float(m.mean()),
        "expected": expected,
        "z": z,
        "p": 0.5 * math.erfc(z / math.sqrt(2)),
        "dist": np.bincount(m.ravel(), minlength=PICK + 1).tolist(),
        "max": int(m.max()),
        "test_draws": n,
    }


def format_holdout(h: dict | None, split_date: str) -> str:
    """One-line summary of ``holdout_hits`` for reports."""
    if h is None:
        return f"Hold-out hits: skipped (needs draws on both sides of {split_date})"
    return (
        f"Hold-out hits (fitted on draws up to {split_date}, scored on {h['test_draws']} later draws): "
        f"mean {h['mean']:.4f} vs chance {h['expected']:.4f}, z={h['z']:+.2f}, p={h['p']:.3f}, "
        f"best {h['max']}, match counts 0..6 {h['dist']}"
    )


def format_chi_square(c: dict) -> str:
    return f"Chi-square frequency statistic: {c['chi2']:.2f} (dof {c['dof']}, p={c['p']:.3f})"
