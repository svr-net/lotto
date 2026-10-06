import numpy as np

from lotto_nn.grammar import extract_grammar, generate_batch, grammar_summary, sample_draw


def test_extract_grammar_counts_starts_and_transitions():
    data = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 7, 8, 9],
        ]
    )
    grammar = extract_grammar(data)
    assert grammar["start_counts"][1] == 2
    assert grammar["transition_counts"][1][2] == 2
    assert grammar["transition_counts"][3][4] == 1
    assert grammar["transition_counts"][3][7] == 1
    assert grammar["vocab"] == sorted(set(data.flatten().tolist()))


def test_grammar_summary_top_starts_and_transitions():
    data = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 7, 8, 9],
            [10, 11, 12, 13, 14, 15],
        ]
    )
    grammar = extract_grammar(data)
    summary = grammar_summary(grammar, top_k=1)
    assert summary["top_starts"][0][0] == 1
    assert summary["top_starts"][0][1] == 2
    assert 1 in summary["top_transitions"]
    assert summary["vocab_size"] == len(set(data.flatten().tolist()))


def test_sample_draw_returns_unique_sorted_numbers():
    data = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 7, 8, 9]])
    grammar = extract_grammar(data)
    rng = np.random.default_rng(0)
    draw = sample_draw(grammar, rng=rng)
    assert len(draw) == 6
    assert len(set(draw)) == 6
    assert draw == sorted(draw)
    assert all(1 <= n <= 59 for n in draw)


def test_sample_draw_handles_empty_grammar():
    grammar = {"start_counts": {}, "transition_counts": {}, "vocab": []}
    rng = np.random.default_rng(0)
    draw = sample_draw(grammar, rng=rng)
    assert len(draw) == 6
    assert len(set(draw)) == 6


def test_generate_batch_shape():
    data = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 7, 8, 9]])
    grammar = extract_grammar(data)
    rng = np.random.default_rng(1)
    batch = generate_batch(grammar, num_samples=10, rng=rng)
    assert batch.shape == (10, 6)
    for row in batch:
        assert len(set(row.tolist())) == 6


def test_extract_grammar_counts_transitions_per_position():
    data = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 7, 8, 9, 10]])
    grammar = extract_grammar(data)
    assert grammar["positional_transition_counts"][(1, 2)][3] == 1
    assert grammar["positional_transition_counts"][(0, 2)][3] == 1
    assert (0, 3) not in grammar["positional_transition_counts"]


def test_sampled_draws_do_not_pile_up_on_max_number():
    # Draws spread over the whole range: a position-blind chain drifts upward and
    # over-samples 59; the positional, room-aware sampler should not.
    rng = np.random.default_rng(0)
    data = np.array([sorted(rng.choice(np.arange(1, 60), 6, replace=False)) for _ in range(2000)])
    batch = generate_batch(extract_grammar(data), num_samples=3000, rng=np.random.default_rng(1))
    freq = np.bincount(batch.ravel(), minlength=60)[1:] / batch.size
    assert freq[58] < 1.5 / 59  # position-blind chain: ~2x; fixed sampler: ~1.1x
    assert all(row.tolist() == sorted(row.tolist()) and len(set(row.tolist())) == 6 for row in batch)
