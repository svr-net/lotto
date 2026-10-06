"""Training loop for the LottoTransformer generator."""
from __future__ import annotations

import torch
import torch.nn as nn

from .model import LottoTransformer, shift_right


def train_model(
    model: LottoTransformer,
    tokens: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module = None,
    epochs: int = 100,
    batch_size: int = 256,
    device: torch.device | None = None,
    verbose: bool = True,
) -> list:
    """Train `model` as a next-number predictor over historical draws.

    Each position sees only the start token and the numbers before it, so the
    model learns the conditional distribution of the next number.

    Returns the list of per-epoch average losses.
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    if device is None:
        device = tokens.device

    model.to(device)
    tokens = tokens.to(device)
    model.train()

    n = tokens.shape[0]
    if n == 0:
        raise ValueError("tokens must contain at least one draw")

    history = []
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        tokens_shuffled = tokens[perm]
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            batch = tokens_shuffled[i : i + batch_size]
            optimizer.zero_grad()
            logits = model(shift_right(batch))
            loss = loss_fn(logits.view(-1, model.vocab_size), batch.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        avg_loss = epoch_loss / n
        history.append(avg_loss)
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}, loss={avg_loss:.4f}")
    return history
