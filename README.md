# lotto

## UK Lotto NN Pipeline

An end-to-end pipeline that ingests historical UK Lotto draws, tokenizes them,
trains a small transformer-based generator, samples lottery-like number
combinations, and runs a suite of statistical backtests comparing generated
draws against historical draws.

### Package layout

- `lotto_nn/data.py` — CSV ingestion, tokenization, and temporal train/test split
- `lotto_nn/model.py` — `LottoTransformer`, a small transformer encoder over 6-number sequences
- `lotto_nn/train.py` — training loop
- `lotto_nn/sample.py` — autoregressive sampling engine (masks invalid/duplicate tokens)
- `lotto_nn/backtest.py` — frequency, pairing, sum, odd/even, spacing, positional bias,
  and simulation backtests
- `lotto_nn/pipeline.py` — orchestrates the full pipeline end-to-end and exposes a CLI
- `lotto_nn/grammar.py` — fits a simple statistical n-gram "grammar" (start distribution +
  positional transition table) over historical draws and samples draws from it
- `lotto_nn/grammar_pipeline.py` — orchestrates the grammar-discovery pipeline end-to-end
  (extraction, sampling, backtesting) and exposes a CLI

### Setup

```bash
pip install -r requirements.txt
```

### Usage

Run the full pipeline against a UK Lotto CSV (columns: `date, n1, n2, n3, n4, n5, n6, bonus`):

```bash
python -m lotto_nn.pipeline --csv sample_data/uk_lotto_sample.csv --epochs 50 --num-samples 5000
```

A small synthetic sample dataset is provided at `sample_data/uk_lotto_sample.csv` for
demonstration and testing purposes; replace it with real historical UK Lotto draws for
meaningful backtest results.

### Full automated run

`run_lotto_pipeline.sh` automates the entire workflow: it downloads the latest UK Lotto
historical feed to `uk_lotto.csv`, runs the pipeline (writing `actual.txt`,
`predictions.txt`, and `backtest.txt`), and prints a PREDICTED | ACTUAL | MATCHES matrix:

```bash
./run_lotto_pipeline.sh
```

Environment variables `EPOCHS`, `NUM_SAMPLES`, `NUM_PREDICTIONS`, and `LOTTO_CSV_URL` can
be set to override the defaults.

### Grammar-discovery pipeline

As an alternative to the neural pipeline, `lotto_nn.grammar_pipeline` fits a simple
statistical n-gram "grammar" (a start-number distribution plus a positional
number-to-number transition table) over historical draws, instead of training a neural
network:

```bash
python -m lotto_nn.grammar_pipeline --csv sample_data/uk_lotto_sample.csv --num-samples 5000
```

`run_lotto_grammar_pipeline.sh` automates the same workflow as `run_lotto_pipeline.sh`
(download `uk_lotto.csv`, run the grammar pipeline, print the PREDICTED | ACTUAL | MATCHES
matrix), but without requiring PyTorch:

```bash
./run_lotto_grammar_pipeline.sh
```

### Tests

```bash
pip install -r requirements.txt
pytest -q
```

**Note:** This pipeline is for statistical/educational exploration only. Lottery draws are
random and independent; no model can predict future outcomes.
