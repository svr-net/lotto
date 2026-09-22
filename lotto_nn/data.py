"""Data ingestion and tokenization utilities for UK Lotto draws."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import torch

DRAW_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]

# Maps normalized (lowercased, non-alphanumeric-stripped) header names from
# common UK National Lottery CSV exports to the column names used internally.
_COLUMN_ALIASES = {
    "date": "date",
    "drawdate": "date",
    "bonus": "bonus",
    "bonusball": "bonus",
}
# Common prefixes used by various UK National Lottery CSV exports to label the
# six main numbers (e.g. "Ball 1", "Number 1", "No. 1", "N1").
_BALL_COLUMN_PREFIXES = ["ball", "number", "num", "no", "n"]
for _i in range(1, 7):
    for _prefix in _BALL_COLUMN_PREFIXES:
        _COLUMN_ALIASES[f"{_prefix}{_i}"] = f"n{_i}"
del _i, _prefix


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


def _find_header_row(csv_path: str) -> int:
    """Find the index of the real header row in a CSV that may be preceded by
    metadata/preamble lines with a different number of fields.

    Some UK National Lottery CSV feeds (e.g. the merseyworld.com export)
    prepend a few lines of free-text metadata before the actual
    comma-separated header and data rows. This inspects the comma count of
    each non-blank line and returns the index of the first line whose field
    count matches the most common field count in the file (i.e. the shape
    the real header/data rows share).
    """
    with open(csv_path, newline="") as f:
        lines = f.readlines()
    field_counts = [line.count(",") + 1 for line in lines if line.strip()]
    if not field_counts:
        return 0
    mode_count = Counter(field_counts).most_common(1)[0][0]
    for i, line in enumerate(lines):
        if line.strip() and line.count(",") + 1 == mode_count:
            return i
    return 0


def load_draws(csv_path: str) -> pd.DataFrame:
    """Load historical UK Lotto draws from a CSV file.

    Expected columns: date, n1, n2, n3, n4, n5, n6, bonus

    Common alternate headers (as used by some UK National Lottery CSV
    exports), such as "Draw Date" and "Ball 1".."Ball 6", are automatically
    recognized and renamed to the expected column names.

    Some feeds prepend a few lines of metadata/preamble before the real
    header row, which would otherwise cause pandas to raise a
    ``ParserError`` ("Expected N fields ... saw M"). If the initial parse
    fails this way, the real header row is located automatically and parsing
    is retried starting from it.
    """
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError:
        header_row = _find_header_row(csv_path)
        df = pd.read_csv(csv_path, skiprows=header_row)
    df = _rename_known_columns(df)
    missing = [c for c in DRAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Columns found in {csv_path}: {list(df.columns)}"
        )
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
