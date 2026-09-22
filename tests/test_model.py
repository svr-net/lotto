import torch

from lotto_nn.model import LottoTransformer


def test_forward_pass_shape():
    model = LottoTransformer(vocab_size=60, seq_len=6, embed_dim=16, nhead=2, ff_dim=32, num_layers=1)
    batch = torch.randint(1, 60, (4, 6), dtype=torch.long)
    logits = model(batch)
    assert logits.shape == (4, 6, 60)


def test_forward_pass_handles_padding_token_zero():
    model = LottoTransformer(vocab_size=60, seq_len=6, embed_dim=16, nhead=2, ff_dim=32, num_layers=1)
    batch = torch.zeros((2, 6), dtype=torch.long)
    logits = model(batch)
    assert logits.shape == (2, 6, 60)
    assert torch.isfinite(logits).all()
