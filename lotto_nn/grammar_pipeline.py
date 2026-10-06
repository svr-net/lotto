"""End-to-end UK Lotto grammar-discovery pipeline: ingest, extract grammar,
sample, backtest.

This pipeline fits a simple statistical n-gram "grammar" (start distribution
+ positional transition table) over historical draws instead of training a
neural generator. It reuses the same data ingestion and backtesting utilities
as the neural pipeline (`lotto_nn.pipeline`).

Usage:
    python -m lotto_nn.grammar_pipeline --csv sample_data/uk_lotto_sample.csv --num-samples 5000

Note: UK Lotto draws are random and independent; the extracted "grammar" is a
descriptive summary of historical co-occurrence structure only, and cannot
predict future draws.
"""
from __future__ import annotations

import argparse

import numpy as np

from . import backtest as bt
from .data import draws_to_numpy, load_draws, temporal_split
from .grammar import extract_grammar, generate_batch, grammar_summary


def run_grammar_pipeline(
    csv_path: str,
    num_samples: int = 1000,
    split_date: str = "2022-12-31",
    verbose: bool = True,
    num_predictions: int = 0,
    seed: int | None = None,
    actual_out: str | None = None,
    predictions_out: str | None = None,
    backtest_out: str | None = None,
) -> dict:
    """Run the full ingestion -> grammar extraction -> sampling -> backtesting pipeline.

    Returns a dictionary of computed backtest results and the fitted grammar summary.

    The distribution backtests use a grammar fitted on every draw. The hold-out
    test fits a separate grammar only on draws up to ``split_date`` and scores
    its samples against the later draws (``results["holdout"]``, None when
    either side of the split is empty).

    If ``actual_out``/``predictions_out``/``backtest_out`` are provided, the latest
    historical draw, ``num_predictions`` freshly sampled draws, and a text summary of
    the backtest results are written to those files respectively.
    """
    rng = np.random.default_rng(seed)

    # 1-2. Data ingestion
    df = load_draws(csv_path)
    draws_np = draws_to_numpy(df)

    # 3. Grammar extraction
    grammar = extract_grammar(draws_np)
    summary = grammar_summary(grammar)

    # 4. Sampling
    generated_np = generate_batch(grammar, num_samples=num_samples, rng=rng)

    # 5. Backtests (reuses the same statistical suite as the neural pipeline)
    results: dict = {"grammar_summary": summary}

    hist_freq = bt.frequency_distribution(draws_np)
    gen_freq = bt.frequency_distribution(generated_np)
    chi = bt.chi_square_test(hist_freq, gen_freq)
    results["chi_square"] = chi["chi2"]
    results["chi_square_dof"] = chi["dof"]
    results["chi_square_p"] = chi["p"]
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

    results["holdout"] = None
    if "date" in df.columns:
        train_draws_np, test_draws_np = temporal_split(df, split_date=split_date)
        results["walk_forward_train_size"] = len(train_draws_np)
        results["walk_forward_test_size"] = len(test_draws_np)
        if len(train_draws_np) and len(test_draws_np):
            holdout_tickets = generate_batch(extract_grammar(train_draws_np), num_samples=num_samples, rng=rng)
            results["holdout"] = bt.holdout_hits(holdout_tickets, test_draws_np)

    results["generated_np"] = generated_np

    if actual_out:
        latest_draw = draws_np[-1]
        with open(actual_out, "w") as f:
            f.write(" ".join(str(n) for n in latest_draw) + "\n")

    if predictions_out:
        n_pred = num_predictions or 5
        prediction_draws = generate_batch(grammar, num_samples=n_pred, rng=rng)
        results["prediction_draws"] = prediction_draws
        with open(predictions_out, "w") as f:
            for draw in prediction_draws:
                f.write(" ".join(str(n) for n in sorted(draw)) + "\n")

    if backtest_out:
        with open(backtest_out, "w") as f:
            f.write(f"Grammar summary (top starts): {summary['top_starts']}\n")
            f.write(f"Grammar vocab size: {summary['vocab_size']}\n")
            f.write(bt.format_chi_square(chi) + "\n")
            f.write(f"Top 20 historical pairs: {results['top_hist_pairs']}\n")
            f.write(f"Top 20 generated pairs: {results['top_gen_pairs']}\n")
            f.write(f"Historical sum mean/std: {results['hist_sum']}\n")
            f.write(f"Generated sum mean/std: {results['gen_sum']}\n")
            f.write(f"Historical odd count mean/std: {results['hist_odd_even']}\n")
            f.write(f"Generated odd count mean/std: {results['gen_odd_even']}\n")
            f.write(f"Historical spacing mean/std: {results['hist_spacing']}\n")
            f.write(f"Generated spacing mean/std: {results['gen_spacing']}\n")
            f.write(bt.format_holdout(results["holdout"], split_date) + "\n")

    if verbose:
        print("Grammar summary (top starts):", summary["top_starts"])
        print("Grammar vocab size:", summary["vocab_size"])
        print(bt.format_chi_square(chi))
        print("Top 20 historical pairs:", results["top_hist_pairs"])
        print("Top 20 generated pairs:", results["top_gen_pairs"])
        print("Historical sum mean/std:", results["hist_sum"])
        print("Generated sum mean/std:", results["gen_sum"])
        print("Historical odd count mean/std:", results["hist_odd_even"])
        print("Generated odd count mean/std:", results["gen_odd_even"])
        print("Historical spacing mean/std:", results["hist_spacing"])
        print("Generated spacing mean/std:", results["gen_spacing"])
        print(bt.format_holdout(results["holdout"], split_date))

    return results


def main():
    parser = argparse.ArgumentParser(description="Run the UK Lotto grammar-discovery pipeline.")
    parser.add_argument("--csv", required=True, help="Path to UK Lotto CSV file.")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--split-date", type=str, default="2022-12-31")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--num-predictions",
        type=int,
        default=5,
        help="Number of prediction draws to write to --predictions-out.",
    )
    parser.add_argument("--actual-out", type=str, default=None, help="Path to write the latest actual draw.")
    parser.add_argument(
        "--predictions-out", type=str, default=None, help="Path to write generated prediction draws."
    )
    parser.add_argument("--backtest-out", type=str, default=None, help="Path to write backtest results summary.")
    args = parser.parse_args()

    run_grammar_pipeline(
        csv_path=args.csv,
        num_samples=args.num_samples,
        split_date=args.split_date,
        verbose=True,
        num_predictions=args.num_predictions,
        seed=args.seed,
        actual_out=args.actual_out,
        predictions_out=args.predictions_out,
        backtest_out=args.backtest_out,
    )


if __name__ == "__main__":
    main()
