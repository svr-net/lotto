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

    The model is fed the start token plus the numbers drawn so far and the
    logits at the next position are sampled. Draws are generated in ascending
    order (matching ``tokenize``), so tokens that are 0, out of range, not
    above the previous number, or too large to leave room for the remaining
    numbers are masked out rather than rejection-sampled.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    seq: list[int] = []
    with torch.no_grad():
        for pos in range(model.seq_len):
            padded = ([0] + seq + [0] * model.seq_len)[: model.seq_len]
            inp = torch.tensor([padded], dtype=torch.long, device=device)
            logits = model(inp)[0, pos].clone()  # [vocab_size]

            # Keep only numbers above the previous pick that leave room for the rest.
            lowest = seq[-1] + 1 if seq else MIN_NUM
            highest = MAX_NUM - (model.seq_len - pos - 1)
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[lowest : highest + 1] = False
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
