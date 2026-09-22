import numpy as np

from lotto_nn import backtest as bt


def test_frequency_distribution_counts_numbers():
    data = np.array([[1, 2, 3, 4, 5, 6], [1, 1, 2, 3, 4, 5]])
    freq = bt.frequency_distribution(data, max_num=6)
    # index 0 -> number 1, index 1 -> number 2, ...
    assert freq.shape == (6,)
    assert freq[0] == 3  # number 1 appears 3 times
    assert freq[5] == 1  # number 6 appears once


def test_chi_square_stat_zero_when_equal():
    hist = np.array([10.0, 20.0, 30.0])
    gen = np.array([10.0, 20.0, 30.0])
    assert bt.chi_square_stat(hist, gen) < 1e-9


def test_chi_square_stat_positive_when_different():
    hist = np.array([10.0, 20.0, 30.0])
    gen = np.array([5.0, 20.0, 30.0])
    assert bt.chi_square_stat(hist, gen) > 0.0


def test_pair_counts_and_top_k():
    data = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 7, 8, 9, 10]])
    pairs = bt.pair_counts(data)
    assert pairs[(1, 2)] == 2
    top = bt.top_k_pairs(pairs, k=1)
    assert top[0][0] == (1, 2)
    assert top[0][1] == 2


def test_sum_stats():
    data = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
    stats = bt.sum_stats(data)
    assert stats["mean"] == 21.0
    assert stats["std"] == 0.0


def test_odd_even_stats():
    data = np.array([[1, 2, 3, 4, 5, 6]])  # 3 odd numbers: 1,3,5
    stats = bt.odd_even_stats(data)
    assert stats["mean"] == 3.0


def test_spacing_stats():
    data = np.array([[1, 2, 3, 4, 5, 6]])
    stats = bt.spacing_stats(data)
    assert stats["mean"] == 1.0
    assert stats["std"] == 0.0


def test_positional_frequency():
    data = np.array([[1, 2, 3, 4, 5, 6], [1, 9, 3, 4, 5, 6]])
    pos_freq = bt.positional_frequency(data, max_num=9)
    # position 0 (first column) always has number 1 -> index 0 has count 2
    assert pos_freq[0, 0] == 2
    # position 1 has number 2 once and number 9 once
    assert pos_freq[1, 1] == 1  # number 2 -> index 1
    assert pos_freq[1, 8] == 1  # number 9 -> index 8


def test_hits_and_simulate_hits():
    ticket = [1, 2, 3, 4, 5, 6]
    draw = [1, 2, 3, 40, 41, 42]
    assert bt.hits(ticket, draw) == 3

    generated = np.array([[1, 2, 3, 4, 5, 6]])
    historical = np.array([[1, 2, 3, 40, 41, 42], [10, 11, 12, 13, 14, 15]])
    sim = bt.simulate_hits(generated, historical)
    assert sim.tolist() == [3]
