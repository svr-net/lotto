"""End-to-end UK Lotto NN pipeline: ingest, train, sample, backtest.

Usage:
    python -m lotto_nn.pipeline --csv sample_data/uk_lotto_sample.csv --epochs 50 --num-samples 5000
"""
from __future__ import annotations

import argparse

import torch

from . import backtest as bt
from .data import draws_to_numpy, load_draws, temporal_split, tokenize
from .model import LottoTransformer
from .sample import generate_batch
from .train import train_model


def run_pipeline(
    csv_path: str,
    epochs: int = 50,
    batch_size: int = 256,
    num_samples: int = 1000,
    temperature: float = 0.9,
    lr: float = 1e-3,
    split_date: str = "2022-12-31",
    verbose: bool = True,
) -> dict:
    """Run the full ingestion -> training -> sampling -> backtesting pipeline.

    Returns a dictionary of computed backtest results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1-2. Data ingestion + tokenization
    df = load_draws(csv_path)
    draws_np = draws_to_numpy(df)
    tokens = tokenize(draws_np).to(device)

    # 3-4. Model + training
    model = LottoTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_model(model, tokens, optimizer, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # 5. Sampling
    generated_np = generate_batch(model, num_samples=num_samples, temperature=temperature)

    # 6. Backtests
    results: dict = {}

    hist_freq = bt.frequency_distribution(draws_np)
    gen_freq = bt.frequency_distribution(generated_np)
    results["chi_square"] = bt.chi_square_stat(hist_freq, gen_freq)
    results["hist_freq"] = hist_freq
    results["gen_freq"] = gen_freq

    hist_pairs = bt.pair_counts(draws_np)
    gen_pairs = bt.pair_counts(generated_np)
    results["top_hist_pairs"] = bt.top_k_pairs(hist_pairs, 20)
    results["top_gen_pairs"] = bt.top_k_pairs(gen_pairs, 20)

    results["hist_sum"] = bt.sum_stats(draws_np)
    results["gen_sum"] = bt.sum_stats(generated_np)

    results["hist_odd_even"] = bt.odd_even_stats(draws_np)
    results["gen_odd_even"] = bt.odd_even_stats(generated_np)

    results["hist_spacing"] = bt.spacing_stats(draws_np)
    results["gen_spacing"] = bt.spacing_stats(generated_np)

    if "date" in df.columns:
        train_draws_np, test_draws_np = temporal_split(df, split_date=split_date)
        results["walk_forward_train_size"] = len(train_draws_np)
        results["walk_forward_test_size"] = len(test_draws_np)

    hist_pos_freq = bt.positional_frequency(draws_np)
    gen_pos_freq = bt.positional_frequency(generated_np)
    results["hist_pos_freq"] = hist_pos_freq
    results["gen_pos_freq"] = gen_pos_freq

    sim_hits = bt.simulate_hits(generated_np, draws_np)
    results["sim_hits_mean"] = float(sim_hits.mean())
    results["sim_hits_std"] = float(sim_hits.std())
    results["sim_hits_max"] = int(sim_hits.max())

    results["generated_np"] = generated_np

    if verbose:
        print("Chi-square frequency statistic:", results["chi_square"])
        print("Top 20 historical pairs:", results["top_hist_pairs"])
        print("Top 20 generated pairs:", results["top_gen_pairs"])
        print("Historical sum mean/std:", results["hist_sum"])
        print("Generated sum mean/std:", results["gen_sum"])
        print("Historical odd count mean/std:", results["hist_odd_even"])
        print("Generated odd count mean/std:", results["gen_odd_even"])
        print("Historical spacing mean/std:", results["hist_spacing"])
        print("Generated spacing mean/std:", results["gen_spacing"])
        print(
            "Simulation hits - mean:",
            results["sim_hits_mean"],
            "std:",
            results["sim_hits_std"],
            "max:",
            results["sim_hits_max"],
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Run the UK Lotto NN pipeline.")
    parser.add_argument("--csv", required=True, help="Path to UK Lotto CSV file.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--split-date", type=str, default="2022-12-31")
    args = parser.parse_args()

    run_pipeline(
        csv_path=args.csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        temperature=args.temperature,
        lr=args.lr,
        split_date=args.split_date,
        verbose=True,
    )


if __name__ == "__main__":
    main()
