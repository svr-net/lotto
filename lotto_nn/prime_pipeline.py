"""Prime-radix pipeline: re-encode draws in the first 10 prime bases and run the same NN on each.

Every number in a draw is written as a fixed-width digit string in base ``b`` (for
``b`` in 2, 3, 5, 7, 11, 13, 17, 19, 23, 29). The six numbers are concatenated into
one digit sequence, the same ``LottoTransformer`` architecture is trained on it
(vocab_size = b, seq_len = 6 * width), and sampled draws are decoded back to decimal
so that the usual backtests can be reported for each base.

Usage:
    python -m lotto_nn.prime_pipeline --csv sample_data/uk_lotto_sample.csv --epochs 50
"""
from __future__ import annotations


import numpy as np
import torch
import torch.nn.functional as F

from . import backtest as bt
from .data import draws_to_numpy, load_draws, temporal_split
from .model import LottoTransformer
from .sample import MAX_NUM, MIN_NUM
from .train import train_model

FIRST_10_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
NUMS_PER_DRAW = 6


def digit_width(base: int, max_value: int = MAX_NUM) -> int:
    """Number of base-``base`` digits needed to represent ``max_value``."""
    width = 1
    while base**width <= max_value:
        width += 1
    return width


def to_base(n: int, base: int, width: int) -> list:
    """Fixed-width, most-significant-first digits of ``n`` in ``base``."""
    digits = []
    for _ in range(width):
        digits.append(n % base)
        n //= base
    return digits[::-1]


def encode_draws(draws_np: np.ndarray, base: int) -> torch.Tensor:
    """[N, 6] decimal draws -> [N, 6 * width] digit tokens in ``base``."""
    width = digit_width(base)
    rows = [[d for n in draw for d in to_base(int(n), base, width)] for draw in draws_np]
    return torch.tensor(rows, dtype=torch.long)


def generate_batch_base(model: LottoTransformer, base: int, num_samples: int, temperature: float = 0.9,
                        batch_size: int = 8192) -> np.ndarray:
    """[num_samples, 6] draws sampled digit by digit as ``sample_draw_base`` does, but for a whole chunk
    of draws at once (one forward pass per digit position for the chunk, not per draw): each row keeps
    the numbers it can still complete to, and only their digits are allowed at each step."""
    model.eval()
    device = next(model.parameters()).device
    width = digit_width(base)
    digits = torch.tensor([[0] * width] + [to_base(n, base, width) for n in range(MIN_NUM, MAX_NUM + 1)],
                          dtype=torch.long, device=device)  # [60, width]; row 0 is never a candidate
    onehot = F.one_hot(digits, model.vocab_size).float()  # [60, width, vocab]
    out = []
    with torch.inference_mode():
        for start in range(0, num_samples, batch_size):
            n = min(batch_size, num_samples - start)
            seq = torch.zeros(n, model.seq_len, dtype=torch.long, device=device)
            used = torch.zeros(n, MAX_NUM + 1, dtype=torch.bool, device=device)
            used[:, 0] = True
            chosen = torch.zeros(n, NUMS_PER_DRAW, dtype=torch.long, device=device)
            pos = 0
            for i in range(NUMS_PER_DRAW):
                cand = ~used
                for k in range(width):
                    inp = torch.zeros_like(seq)
                    inp[:, 1 : pos + 1] = seq[:, :pos]
                    logits = model(inp)[:, pos]
                    allowed = (cand.float() @ onehot[:, k]) > 0  # digits some candidate has at place k
                    probs = F.softmax(logits.masked_fill(~allowed, float("-inf")) / temperature, dim=1)
                    d = torch.multinomial(probs, 1).squeeze(1)
                    seq[:, pos] = d
                    pos += 1
                    cand &= digits[:, k][None, :] == d[:, None]
                num = cand.float().argmax(1)
                chosen[:, i] = num
                used[torch.arange(n, device=device), num] = True
            out.append(chosen.cpu().numpy())
    return np.concatenate(out).astype(int) if out else np.zeros((0, NUMS_PER_DRAW), dtype=int)


def run_base(
    draws_np: np.ndarray,
    base: int,
    epochs: int = 50,
    batch_size: int = 256,
    num_samples: int = 1000,
    temperature: float = 0.9,
    lr: float = 1e-3,
    verbose: bool = True,
    test_draws_np: np.ndarray | None = None,
) -> dict:
    """Train the NN on draws re-encoded in ``base`` and return backtest results.

    If ``test_draws_np`` (later draws not in ``draws_np``) is given, the sampled
    draws are also scored against it (``result["holdout"]``).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    width = digit_width(base)
    tokens = encode_draws(draws_np, base).to(device)
    model = LottoTransformer(vocab_size=base, seq_len=NUMS_PER_DRAW * width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = train_model(model, tokens, optimizer, epochs=epochs, batch_size=batch_size, verbose=False)

    generated = generate_batch_base(model, base, num_samples, temperature)

    chi = bt.chi_square_test(bt.frequency_distribution(draws_np), bt.frequency_distribution(generated))
    holdout = bt.holdout_hits(generated, test_draws_np) if test_draws_np is not None and len(test_draws_np) else None
    result = {
        "base": base,
        "digit_width": width,
        "final_loss": history[-1] if history else float("nan"),
        "chi_square": chi["chi2"],
        "chi_square_p": chi["p"],
        "hist_sum": bt.sum_stats(draws_np),
        "gen_sum": bt.sum_stats(generated),
        "hist_odd_even": bt.odd_even_stats(draws_np),
        "gen_odd_even": bt.odd_even_stats(generated),
        "hist_spacing": bt.spacing_stats(draws_np),
        "gen_spacing": bt.spacing_stats(generated),
        "holdout": holdout,
        "generated_np": generated,
    }
    if verbose:
        print(f"=== Base {base} (digits per number: {width}) ===")
        print("Final training loss:", result["final_loss"])
        print(bt.format_chi_square(chi))
        print("Historical/Generated sum mean/std:", result["hist_sum"], result["gen_sum"])
        print("Historical/Generated odd mean/std:", result["hist_odd_even"], result["gen_odd_even"])
        print("Historical/Generated spacing mean/std:", result["hist_spacing"], result["gen_spacing"])
        if holdout is not None:
            print(bt.format_holdout(holdout, "the split date"))
    return result


def run_prime_pipeline(
    csv_path: str,
    bases: list | None = None,
    epochs: int = 50,
    batch_size: int = 256,
    num_samples: int = 1000,
    temperature: float = 0.9,
    lr: float = 1e-3,
    verbose: bool = True,
    split_date: str = "2022-12-31",
) -> dict:
    """Run the NN once per prime base. Returns ``{base: result_dict}``.

    When the CSV has draws on both sides of ``split_date``, each base is trained
    only on the earlier draws and scored against the later ones (hold-out);
    otherwise it is trained on every draw and no hold-out is reported.
    """
    df = load_draws(csv_path)
    draws_np, test_np = draws_to_numpy(df), None
    if "date" in df.columns:
        train_np, later_np = temporal_split(df, split_date=split_date)
        if len(train_np) and len(later_np):
            draws_np, test_np = train_np, later_np
    results = {}
    for base in bases or FIRST_10_PRIMES:
        results[base] = run_base(
            draws_np, base, epochs, batch_size, num_samples, temperature, lr, verbose, test_np
        )
    return results

