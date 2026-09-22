"""Statistical backtesting suite comparing generated draws to historical draws."""
from __future__ import annotations

from collections import Counter

import numpy as np

MAX_NUM = 59


def frequency_distribution(data_np: np.ndarray, max_num: int = MAX_NUM) -> np.ndarray:
    """Return counts of each number 1..max_num appearing in data_np ([N,6])."""
    flat = data_np.flatten()
    freq = np.bincount(flat, minlength=max_num + 1)
    return freq[1:]


def chi_square_stat(hist_freq: np.ndarray, gen_freq: np.ndarray) -> float:
    """Chi-square statistic comparing generated frequencies to historical (expected)."""
    expected = hist_freq + 1e-9
    observed = gen_freq
    return float(np.sum((observed - expected) ** 2 / expected))


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


def simulate_hits(generated_np: np.ndarray, draws_np: np.ndarray) -> np.ndarray:
    """For each generated ticket, find the max number of hits against any historical draw."""
    max_hits_per_ticket = []
    for ticket in generated_np:
        max_hits = 0
        for draw in draws_np:
            h = hits(ticket, draw)
            if h > max_hits:
                max_hits = h
        max_hits_per_ticket.append(max_hits)
    return np.array(max_hits_per_ticket)
