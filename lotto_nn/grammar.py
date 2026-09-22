"""Grammar-discovery module for UK Lotto draws.

This module treats each historical draw as a short "sentence" of six sorted
number "tokens" and fits a simple, well-understood statistical n-gram model
(a first-order Markov chain / bigram "grammar") over the position-to-position
transitions between consecutive sorted numbers. This is a legitimate way to
summarize co-occurrence structure in the historical data, but it is purely
descriptive: UK Lotto draws are independent random events, so no grammar,
pattern, or model can predict future outcomes.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

MIN_NUM = 1
MAX_NUM = 59


def extract_grammar(draws_np: np.ndarray) -> dict:
    """Fit a bigram "grammar" over sorted draw sequences.

    For each draw, numbers are sorted ascending and treated as a sequence of
    tokens. A start-token distribution (first number of each draw) and a
    transition table (probability of the next number given the current
    number) are estimated by counting frequencies across all historical
    draws.

    Returns a dict with:
        - "start_counts": Counter of first numbers in each draw
        - "transition_counts": dict[int, Counter] mapping a number to counts
          of numbers that followed it (in sorted order) in some draw
        - "vocab": sorted list of numbers observed in the data
    """
    start_counts: Counter = Counter()
    transition_counts: dict = defaultdict(Counter)
    vocab: set = set()

    for row in draws_np:
        seq = sorted(int(n) for n in row)
        if not seq:
            continue
        start_counts[seq[0]] += 1
        vocab.update(seq)
        for a, b in zip(seq, seq[1:]):
            transition_counts[a][b] += 1

    return {
        "start_counts": start_counts,
        "transition_counts": dict(transition_counts),
        "vocab": sorted(vocab),
    }


def grammar_summary(grammar: dict, top_k: int = 10) -> dict:
    """Summarize a fitted grammar: most common starting numbers and the most
    common transitions out of each of those starting numbers.
    """
    start_counts: Counter = grammar["start_counts"]
    transition_counts: dict = grammar["transition_counts"]

    top_starts = start_counts.most_common(top_k)
    top_transitions = {}
    for num, _ in top_starts:
        counter = transition_counts.get(num, Counter())
        top_transitions[num] = counter.most_common(top_k)

    return {
        "top_starts": top_starts,
        "top_transitions": top_transitions,
        "vocab_size": len(grammar["vocab"]),
    }


def _weighted_choice(rng: np.random.Generator, options: list, weights: list) -> int:
    total = float(sum(weights))
    if total <= 0:
        return int(rng.choice(options))
    probs = [w / total for w in weights]
    return int(rng.choice(options, p=probs))


def sample_draw(
    grammar: dict,
    seq_len: int = 6,
    min_num: int = MIN_NUM,
    max_num: int = MAX_NUM,
    rng: np.random.Generator | None = None,
) -> list:
    """Sample a single draw of `seq_len` unique numbers using the fitted
    grammar's start distribution and transition table.

    Falls back to uniform-random choices whenever the grammar has no learned
    signal for the current state (e.g. an unseen number or an exhausted
    transition table), so the sampler always terminates with a valid draw.
    """
    if rng is None:
        rng = np.random.default_rng()

    start_counts: Counter = grammar["start_counts"]
    transition_counts: dict = grammar["transition_counts"]
    full_range = list(range(min_num, max_num + 1))

    seq: list[int] = []

    if start_counts:
        options, weights = zip(*start_counts.items())
        first = _weighted_choice(rng, list(options), list(weights))
    else:
        first = int(rng.choice(full_range))
    seq.append(first)

    while len(seq) < seq_len:
        current = seq[-1]
        counter = transition_counts.get(current, Counter())
        candidates = [(n, c) for n, c in counter.items() if n not in seq]
        if candidates:
            options, weights = zip(*candidates)
            nxt = _weighted_choice(rng, list(options), list(weights))
        else:
            remaining = [n for n in full_range if n not in seq]
            if not remaining:
                break
            nxt = int(rng.choice(remaining))
        seq.append(nxt)

    return sorted(seq)


def generate_batch(
    grammar: dict,
    num_samples: int = 1000,
    seq_len: int = 6,
    min_num: int = MIN_NUM,
    max_num: int = MAX_NUM,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a batch of grammar-sampled draws. Returns an [num_samples, seq_len] int array."""
    if rng is None:
        rng = np.random.default_rng()
    samples = [
        sample_draw(grammar, seq_len=seq_len, min_num=min_num, max_num=max_num, rng=rng)
        for _ in range(num_samples)
    ]
    return np.array(samples, dtype=int)
