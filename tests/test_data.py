import textwrap

import numpy as np
import pandas as pd

from lotto_nn.data import draws_to_numpy, load_draws, temporal_split, tokenize


SAMPLE_CSV = textwrap.dedent(
    """\
    date,n1,n2,n3,n4,n5,n6,bonus
    2022-01-01,1,2,3,4,5,6,7
    2022-06-01,10,11,12,13,14,15,16
    2023-01-01,20,21,22,23,24,25,26
    """
)


def test_load_draws_reads_required_columns(tmp_path):
    csv_path = tmp_path / "draws.csv"
    csv_path.write_text(SAMPLE_CSV)
    df = load_draws(str(csv_path))
    assert list(df["n1"]) == [1, 10, 20]
    assert len(df) == 3


def test_load_draws_missing_columns_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("date,n1\n2022-01-01,1\n")
    try:
        load_draws(str(csv_path))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_draws_to_numpy_shape(tmp_path):
    csv_path = tmp_path / "draws.csv"
    csv_path.write_text(SAMPLE_CSV)
    df = load_draws(str(csv_path))
    draws_np = draws_to_numpy(df)
    assert draws_np.shape == (3, 6)
    assert draws_np.dtype == np.int64 or draws_np.dtype == int


def test_tokenize_returns_long_tensor(tmp_path):
    csv_path = tmp_path / "draws.csv"
    csv_path.write_text(SAMPLE_CSV)
    df = load_draws(str(csv_path))
    draws_np = draws_to_numpy(df)
    tokens = tokenize(draws_np)
    assert tokens.shape == (3, 6)
    assert tokens.dtype.is_floating_point is False


def test_temporal_split(tmp_path):
    csv_path = tmp_path / "draws.csv"
    csv_path.write_text(SAMPLE_CSV)
    df = load_draws(str(csv_path))
    train, test = temporal_split(df, split_date="2022-12-31")
    assert train.shape == (2, 6)
    assert test.shape == (1, 6)
