# lotto

## UK Lotto NN Pipeline

An end-to-end pipeline that ingests historical UK Lotto draws, tokenizes them,
trains a small transformer-based generator, samples lottery-like number
combinations, and runs a suite of statistical backtests comparing generated
draws against historical draws.

### Package layout

- `lotto_nn/data.py` — CSV ingestion, tokenization, and temporal train/test split
- `lotto_nn/model.py` — `LottoTransformer`, a small causal transformer that predicts each number of a sorted draw from the ones before it
- `lotto_nn/train.py` — training loop
- `lotto_nn/sample.py` — autoregressive sampling engine (masks invalid/duplicate tokens)
- `lotto_nn/backtest.py` — frequency, pairing, sum, odd/even, spacing, positional bias,
  and hold-out backtests (a two-sample chi-square on number frequencies, and tickets from a model
  fitted on draws up to `--split-date` scored against the later draws vs the 36/59 chance rate)
- `lotto_nn/pipeline.py` — orchestrates the full pipeline end-to-end and exposes a CLI
- `lotto_nn/grammar.py` — fits a simple statistical n-gram "grammar" (start distribution +
  positional transition table) over historical draws and samples draws from it
- `lotto_nn/grammar_pipeline.py` — orchestrates the grammar-discovery pipeline end-to-end
  (extraction, sampling, backtesting) and exposes a CLI

- `lotto_nn/prime_pipeline.py` — re-encodes draws as fixed-width digits in each of the first 10
  prime radix bases (2, 3, 5, 7, 11, 13, 17, 19, 23, 29), trains the same transformer per base,
  decodes samples back to decimal, and reports backtest outcomes per base
  (`python -m lotto_nn.prime_pipeline --csv uk_lotto.csv --epochs 50 --out prime.txt`)

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

### Full-history feed, walk-forward backtest and ensemble

`scripts/fetch_merseyworld.py` downloads every UK Lotto draw since 1994 from
lottery.merseyworld.com (no API key) into `data/uk_lotto_full.csv`
(`draw_no,date,round,n1..n6,bonus,...`, chronological, main numbers sorted; two-round
draws since June 2026 appear as rounds 1 and 2). `data/uk_lotto_59.csv` is the 1-59 era
subset (draw 2066, 2015-10-10, onward).

`scripts/walk_forward_backtest.py` replays nine strategies (random baseline, hot/cold
weighted, hot/cold/overdue top-6, repeat-last, the repo n-gram grammar and transformer)
over every round after `--split-date`, each seeing only earlier draws, and scores matches
against the exact hypergeometric chance distribution. It also correlates the strategies
(round-by-round hits and number picks), combines all nine into equal-weight and
trailing-performance-weighted ensembles, repeats everything over several seeds, and
writes next-draw picks to `results/walk_forward.json`:

```bash
python scripts/fetch_merseyworld.py
python scripts/walk_forward_backtest.py --seeds 2026 1 2 3 4 --out results
python scripts/build_report.py   # results/report.html
```

### Tests

```bash
pip install -r requirements.txt
pytest -q
```

**Note:** This pipeline is for statistical/educational exploration only. Lottery draws are
random and independent; no model can predict future outcomes.
