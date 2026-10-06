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


def test_hits_and_match_matrix():
    ticket = [1, 2, 3, 4, 5, 6]
    draw = [1, 2, 3, 40, 41, 42]
    assert bt.hits(ticket, draw) == 3

    tickets = np.array([[1, 2, 3, 4, 5, 6], [40, 41, 42, 43, 44, 45]])
    draws = np.array([[1, 2, 3, 40, 41, 42], [10, 11, 12, 13, 14, 15]])
    assert bt.match_matrix(tickets, draws).tolist() == [[3, 0], [3, 0]]


def test_chi_square_ignores_sample_size():
    # Same proportions, 5x more generated samples: the old statistic grew with
    # the sample size even though the distributions are identical.
    hist = np.array([10.0, 20.0, 30.0])
    res = bt.chi_square_test(hist, hist * 5)
    assert res["chi2"] < 1e-9
    assert res["dof"] == 2
    assert res["p"] > 0.999


def test_chi_square_p_value_detects_real_difference():
    hist = np.array([100.0] * 10)
    same = bt.chi_square_test(hist, np.array([500.0] * 10))
    skewed = bt.chi_square_test(hist, np.array([900.0] + [455.0] * 9))
    assert same["p"] > 0.5
    assert skewed["p"] < 1e-3


def test_chi2_sf_matches_known_critical_values():
    assert abs(bt.chi2_sf(3.841, 1) - 0.05) < 1e-3
    assert abs(bt.chi2_sf(76.15, 58) - 0.055) < 2e-3


def test_holdout_hits_random_tickets_match_chance():
    rng = np.random.default_rng(0)
    draw = lambda: np.sort(rng.choice(np.arange(1, 60), 6, replace=False))
    tickets = np.array([draw() for _ in range(200)])
    test_draws = np.array([draw() for _ in range(300)])
    res = bt.holdout_hits(tickets, test_draws)
    assert abs(res["expected"] - 36 / 59) < 1e-12
    assert abs(res["mean"] - res["expected"]) < 0.03
    assert abs(res["z"]) < 3.5
    assert sum(res["dist"]) == 200 * 300
    assert res["test_draws"] == 300


def test_holdout_hits_flags_tickets_that_match_test_draws():
    rng = np.random.default_rng(1)
    test_draws = np.array([np.sort(rng.choice(np.arange(1, 60), 6, replace=False)) for _ in range(50)])
    res = bt.holdout_hits(test_draws, test_draws)  # tickets that "know" the answers
    assert res["max"] == 6
    assert res["z"] > 5
