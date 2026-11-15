"""Sampling engine for generating lottery-like draws from a trained model."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .model import LottoTransformer

MIN_NUM = 1
MAX_NUM = 59


def generate_batch(
    model: LottoTransformer,
    num_samples: int = 1000,
    temperature: float = 0.9,
    device: torch.device | None = None,
    ascending: bool = True,
    batch_size: int = 8192,
) -> np.ndarray:
    """Generate a batch of NN-style draws. Returns an [num_samples, 6] int array.

    Every draw in a chunk of ``batch_size`` is generated together: one forward pass per position for
    the whole chunk instead of one per draw (batched autoregressive decoding), with the same masks and
    the same distribution as ``sample_draw``, so the draws follow the model exactly as before, only
    many times faster.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    L = model.seq_len
    vocab = torch.arange(model.vocab_size, device=device)
    out = []
    with torch.inference_mode():
        for start in range(0, num_samples, batch_size):
            n = min(batch_size, num_samples - start)
            seqs = torch.zeros(n, L, dtype=torch.long, device=device)
            for pos in range(L):
                inp = torch.zeros(n, L, dtype=torch.long, device=device)  # start token, the draw so far, padding
                inp[:, 1 : pos + 1] = seqs[:, :pos]
                logits = model(inp)[:, pos]  # [n, vocab_size]
                if ascending:
                    lowest = seqs[:, pos - 1] + 1 if pos else torch.full((n,), MIN_NUM, device=device)
                    highest = MAX_NUM - (L - pos - 1)
                    mask = (vocab[None, :] < lowest[:, None]) | (vocab[None, :] > highest)
                else:
                    mask = ((vocab < MIN_NUM) | (vocab > MAX_NUM)).expand(n, -1)
                    if pos:
                        mask = mask.scatter(1, seqs[:, :pos], True)
                probs = F.softmax(logits.masked_fill(mask, float("-inf")) / temperature, dim=1)
                seqs[:, pos] = torch.multinomial(probs, 1).squeeze(1)
            out.append(seqs.cpu().numpy())
    return np.concatenate(out).astype(int) if out else np.zeros((0, L), dtype=int)
