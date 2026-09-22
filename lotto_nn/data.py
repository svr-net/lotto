"""Data ingestion and tokenization utilities for UK Lotto draws."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

DRAW_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]


def load_draws(csv_path: str) -> pd.DataFrame:
    """Load historical UK Lotto draws from a CSV file.

    Expected columns: date, n1, n2, n3, n4, n5, n6, bonus
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in DRAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df


def draws_to_numpy(df: pd.DataFrame) -> np.ndarray:
    """Extract the six main numbers as an [N, 6] int array."""
    return df[DRAW_COLUMNS].values.astype(int)


def tokenize(draws_np: np.ndarray) -> torch.Tensor:
    """Convert a [N, 6] int array of lotto numbers into an integer token tensor."""
    return torch.tensor(draws_np, dtype=torch.long)


def temporal_split(df: pd.DataFrame, split_date: str = "2022-12-31"):
    """Split draws into an early "train" set and a later "test" set by date.

    Returns (train_draws_np, test_draws_np) as [N, 6] int arrays.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    split_ts = pd.to_datetime(split_date)
    train_df = df[df["date"] <= split_ts]
    test_df = df[df["date"] > split_ts]
    train_draws = draws_to_numpy(train_df)
    test_draws = draws_to_numpy(test_df)
    return train_draws, test_draws
