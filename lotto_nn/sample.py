"""Sampling engine for generating lottery-like draws from a trained model."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .model import LottoTransformer

MIN_NUM = 1
MAX_NUM = 59


def sample_draw(
    model: LottoTransformer,
    temperature: float = 0.9,
    device: torch.device | None = None,
) -> list:
    """Sample a single draw of 6 unique numbers in [1, 59] from the model.

    Invalid tokens (0, out-of-range, or already-chosen numbers) are masked out
    of the softmax distribution before sampling, rather than rejection-sampled
    in an unbounded loop.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    seq: list[int] = []
    with torch.no_grad():
        for pos in range(model.seq_len):
            padded = seq + [0] * (model.seq_len - len(seq))
            inp = torch.tensor([padded], dtype=torch.long, device=device)
            logits = model(inp)[0, pos].clone()  # [vocab_size]

            # Mask out padding token 0, numbers > MAX_NUM, and already-picked numbers.
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[0] = True
            if MAX_NUM + 1 < mask.shape[0]:
                mask[MAX_NUM + 1 :] = True
            for n in seq:
                mask[n] = True
            logits = logits.masked_fill(mask, float("-inf"))

            probs = F.softmax(logits / temperature, dim=0)
            num = torch.multinomial(probs, 1).item()
            seq.append(num)
    return seq


def generate_batch(
    model: LottoTransformer,
    num_samples: int = 1000,
    temperature: float = 0.9,
    device: torch.device | None = None,
) -> np.ndarray:
    """Generate a batch of NN-style draws. Returns an [num_samples, 6] int array."""
    samples = [sample_draw(model, temperature, device) for _ in range(num_samples)]
    return np.array(samples, dtype=int)
