"""Data ingestion and tokenization utilities for UK Lotto draws."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

DRAW_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]

# Maps normalized (lowercased, non-alphanumeric-stripped) header names from
# common UK National Lottery CSV exports to the column names used internally.
_COLUMN_ALIASES = {
    "date": "date",
    "drawdate": "date",
    "ball1": "n1",
    "ball2": "n2",
    "ball3": "n3",
    "ball4": "n4",
    "ball5": "n5",
    "ball6": "n6",
    "n1": "n1",
    "n2": "n2",
    "n3": "n3",
    "n4": "n4",
    "n5": "n5",
    "n6": "n6",
    "bonus": "bonus",
    "bonusball": "bonus",
}


def _normalize_column_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _rename_known_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename recognized alternate headers (e.g. "Ball 1", "Draw Date") to the
    internal column names ("n1", "date"), leaving unrecognized columns as-is.
    """
    rename_map = {}
    for col in df.columns:
        normalized = _normalize_column_name(col)
        if normalized in _COLUMN_ALIASES:
            rename_map[col] = _COLUMN_ALIASES[normalized]
    return df.rename(columns=rename_map)


def load_draws(csv_path: str) -> pd.DataFrame:
    """Load historical UK Lotto draws from a CSV file.

    Expected columns: date, n1, n2, n3, n4, n5, n6, bonus

    Common alternate headers (as used by some UK National Lottery CSV
    exports), such as "Draw Date" and "Ball 1".."Ball 6", are automatically
    recognized and renamed to the expected column names.
    """
    df = pd.read_csv(csv_path)
    df = _rename_known_columns(df)
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
