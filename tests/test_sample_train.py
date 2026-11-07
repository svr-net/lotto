import numpy as np
import torch

from lotto_nn.model import LottoTransformer
from lotto_nn.sample import generate_batch, sample_draw, MIN_NUM, MAX_NUM
from lotto_nn.train import train_model


def _tiny_model():
    return LottoTransformer(vocab_size=60, seq_len=6, embed_dim=16, nhead=2, ff_dim=32, num_layers=1)


def test_sample_draw_returns_six_unique_valid_numbers():
    model = _tiny_model()
    draw = sample_draw(model, temperature=1.0, device=torch.device("cpu"))
    assert len(draw) == 6
    assert len(set(draw)) == 6
    assert all(MIN_NUM <= n <= MAX_NUM for n in draw)


def test_generate_batch_shape_and_validity():
    model = _tiny_model()
    batch = generate_batch(model, num_samples=5, temperature=1.0, device=torch.device("cpu"))
    assert batch.shape == (5, 6)
    for row in batch:
        assert len(set(row.tolist())) == 6
        assert all(MIN_NUM <= n <= MAX_NUM for n in row)


def test_train_model_runs_and_returns_finite_loss():
    model = _tiny_model()
    tokens = torch.randint(1, 60, (20, 6), dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = train_model(model, tokens, optimizer, epochs=2, batch_size=8, verbose=False)
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in history)


def test_model_is_causal():
    # Logits at position i must not depend on tokens after i, otherwise the
    # model can learn to copy its input instead of predicting the next number.
    model = _tiny_model().eval()
    a = torch.tensor([[0, 5, 10, 20, 30, 40]])
    b = torch.tensor([[0, 5, 10, 55, 56, 57]])
    with torch.no_grad():
        assert torch.allclose(model(a)[0, :3], model(b)[0, :3], atol=1e-6)


def test_trained_model_does_not_copy_input():
    # Uniformly random sorted draws carry no learnable signal beyond ordering, so
    # a next-number model cannot drive the loss near zero; a copying model can.
    torch.manual_seed(0)
    draws = torch.stack([torch.randperm(59)[:6].sort().values + 1 for _ in range(300)])
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    history = train_model(model, draws, optimizer, epochs=30, batch_size=64, verbose=False)
    assert history[-1] > 2.0


def test_sampled_draws_are_sorted():
    model = _tiny_model()
    batch = generate_batch(model, num_samples=20, temperature=1.0, device=torch.device("cpu"))
    assert all(row.tolist() == sorted(row.tolist()) for row in batch)


def test_unordered_sampling_allows_any_order():
    torch.manual_seed(0)
    model = _tiny_model()
    batch = generate_batch(model, num_samples=30, temperature=1.0, device=torch.device("cpu"), ascending=False)
    for row in batch:
        assert len(set(row.tolist())) == 6
        assert all(MIN_NUM <= n <= MAX_NUM for n in row)
    assert any(row.tolist() != sorted(row.tolist()) for row in batch)


def _next_number_probs(model, prefix, temperature=0.9, ascending=True):
    """Exact distribution of the next number after ``prefix``, as sample_draw computes it."""
    import torch.nn.functional as F
    pos = len(prefix)
    inp = torch.tensor([([0] + prefix + [0] * model.seq_len)[: model.seq_len]])
    with torch.no_grad():
        logits = model(inp)[0, pos]
    mask = torch.ones_like(logits, dtype=torch.bool)
    if ascending:
        mask[(prefix[-1] + 1 if prefix else 1) : 59 - (model.seq_len - pos - 1) + 1] = False
    else:
        mask[1:60] = False
        mask[prefix] = True
    return F.softmax(logits.masked_fill(mask, float("-inf")) / temperature, dim=0)


def test_batched_sampling_follows_the_model_like_sample_draw():
    # the batched sampler's first numbers follow the exact first-step distribution of sample_draw
    torch.manual_seed(0)
    model = LottoTransformer()
    model.eval()
    for ascending in (True, False):
        p = _next_number_probs(model, [], ascending=ascending).numpy()
        draws = generate_batch(model, num_samples=20000, ascending=ascending)
        assert all(len(set(d)) == 6 and 1 <= min(d) and max(d) <= 59 for d in draws[:500])
        freq = np.bincount(draws[:, 0], minlength=60) / len(draws)
        assert np.abs(freq - p).max() < 0.02
        if ascending:
            assert (np.diff(draws, axis=1) > 0).all()
