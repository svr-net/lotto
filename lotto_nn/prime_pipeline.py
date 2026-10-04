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

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from . import backtest as bt
from .data import draws_to_numpy, load_draws
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


def from_base(digits, base: int) -> int:
    value = 0
    for d in digits:
        value = value * base + int(d)
    return value


def encode_draws(draws_np: np.ndarray, base: int) -> torch.Tensor:
    """[N, 6] decimal draws -> [N, 6 * width] digit tokens in ``base``."""
    width = digit_width(base)
    rows = [[d for n in draw for d in to_base(int(n), base, width)] for draw in draws_np]
    return torch.tensor(rows, dtype=torch.long)


def decode_draws(tokens: np.ndarray, base: int) -> np.ndarray:
    """[N, 6 * width] digit tokens -> [N, 6] decimal draws."""
    width = digit_width(base)
    tokens = np.asarray(tokens)
    out = np.zeros((tokens.shape[0], NUMS_PER_DRAW), dtype=int)
    for i in range(tokens.shape[0]):
        for j in range(NUMS_PER_DRAW):
            out[i, j] = from_base(tokens[i, j * width : (j + 1) * width], base)
    return out


def sample_draw_base(model: LottoTransformer, base: int, temperature: float = 0.9) -> list:
    """Sample one draw of 6 unique numbers in [1, 59], digit by digit.

    At each digit, only digits that can still complete to an unused number in range
    are allowed, so decoded draws are always valid.
    """
    model.eval()
    device = next(model.parameters()).device
    width = digit_width(base)
    digits_of = {n: to_base(n, base, width) for n in range(MIN_NUM, MAX_NUM + 1)}
    chosen: list[int] = []
    seq: list[int] = []
    with torch.no_grad():
        for _ in range(NUMS_PER_DRAW):
            candidates = [n for n in digits_of if n not in chosen]
            for k in range(width):
                pos = len(seq)
                padded = seq + [0] * (model.seq_len - len(seq))
                inp = torch.tensor([padded], dtype=torch.long, device=device)
                logits = model(inp)[0, pos].clone()
                allowed = {digits_of[n][k] for n in candidates}
                mask = torch.ones_like(logits, dtype=torch.bool)
                mask[list(allowed)] = False
                logits = logits.masked_fill(mask, float("-inf"))
                probs = F.softmax(logits / temperature, dim=0)
                d = torch.multinomial(probs, 1).item()
                seq.append(d)
                candidates = [n for n in candidates if digits_of[n][k] == d]
            chosen.append(candidates[0])
    return chosen


def run_base(
    draws_np: np.ndarray,
    base: int,
    epochs: int = 50,
    batch_size: int = 256,
    num_samples: int = 1000,
    temperature: float = 0.9,
    lr: float = 1e-3,
    verbose: bool = True,
) -> dict:
    """Train the NN on draws re-encoded in ``base`` and return backtest results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    width = digit_width(base)
    tokens = encode_draws(draws_np, base).to(device)
    model = LottoTransformer(vocab_size=base, seq_len=NUMS_PER_DRAW * width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = train_model(model, tokens, optimizer, epochs=epochs, batch_size=batch_size, verbose=False)

    generated = np.array(
        [sample_draw_base(model, base, temperature) for _ in range(num_samples)], dtype=int
    ).reshape(num_samples, NUMS_PER_DRAW)

    sim_hits = bt.simulate_hits(generated, draws_np)
    result = {
        "base": base,
        "digit_width": width,
        "final_loss": history[-1] if history else float("nan"),
        "chi_square": bt.chi_square_stat(
            bt.frequency_distribution(draws_np), bt.frequency_distribution(generated)
        ),
        "hist_sum": bt.sum_stats(draws_np),
        "gen_sum": bt.sum_stats(generated),
        "hist_odd_even": bt.odd_even_stats(draws_np),
        "gen_odd_even": bt.odd_even_stats(generated),
        "hist_spacing": bt.spacing_stats(draws_np),
        "gen_spacing": bt.spacing_stats(generated),
        "sim_hits_mean": float(sim_hits.mean()),
        "sim_hits_std": float(sim_hits.std()),
        "sim_hits_max": int(sim_hits.max()),
        "generated_np": generated,
    }
    if verbose:
        print(f"=== Base {base} (digits per number: {width}) ===")
        print("Final training loss:", result["final_loss"])
        print("Chi-square frequency statistic:", result["chi_square"])
        print("Historical/Generated sum mean/std:", result["hist_sum"], result["gen_sum"])
        print("Historical/Generated odd mean/std:", result["hist_odd_even"], result["gen_odd_even"])
        print("Historical/Generated spacing mean/std:", result["hist_spacing"], result["gen_spacing"])
        print(
            "Simulation hits - mean:", result["sim_hits_mean"],
            "std:", result["sim_hits_std"], "max:", result["sim_hits_max"],
        )
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
    out: str | None = None,
) -> dict:
    """Run the NN once per prime base. Returns ``{base: result_dict}``."""
    draws_np = draws_to_numpy(load_draws(csv_path))
    results = {}
    for base in bases or FIRST_10_PRIMES:
        results[base] = run_base(
            draws_np, base, epochs, batch_size, num_samples, temperature, lr, verbose
        )
    if out:
        with open(out, "w") as f:
            for base, r in results.items():
                f.write(
                    f"base={base} width={r['digit_width']} final_loss={r['final_loss']:.4f} "
                    f"chi_square={r['chi_square']} gen_sum={r['gen_sum']} "
                    f"hist_sum={r['hist_sum']} sim_hits_mean={r['sim_hits_mean']:.4f} "
                    f"sim_hits_max={r['sim_hits_max']}\n"
                )
    return results


def main():
    parser = argparse.ArgumentParser(description="Run the lotto NN over the first 10 prime radix bases.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default=None, help="Path to write per-base summary.")
    args = parser.parse_args()
    run_prime_pipeline(
        args.csv, epochs=args.epochs, batch_size=args.batch_size, num_samples=args.num_samples,
        temperature=args.temperature, lr=args.lr, out=args.out,
    )


if __name__ == "__main__":
    main()
