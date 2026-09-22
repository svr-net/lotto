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
