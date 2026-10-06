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
(`draw_no,date,round,n1..n6,bonus,...,d1..d6`, chronological; `n1..n6` are the main
numbers sorted and `d1..d6` the same balls in the order the machine drew them; two-round
draws since June 2026 appear as rounds 1 and 2). `data/uk_lotto_59.csv` is the 1-59 era
subset (draw 2066, 2015-10-10, onward).

`scripts/walk_forward_backtest.py` replays fourteen strategies (random baseline, hot/cold
weighted, hot/cold/overdue top-6, repeat-last, the repo n-gram grammar and transformer, plus
five built on draw order and machines: early-ball frequency, a drawn-order Markov chain, the
transformer trained on drawn order, machine-matched hot numbers and recent bonus balls)
over every round after `--split-date`, each seeing only earlier draws, and scores matches
against the exact hypergeometric chance distribution. It also correlates the strategies
(round-by-round hits and number picks), combines all nine into equal-weight and
trailing-performance-weighted ensembles, tests the drawn order for structure (per-position
uniformity, serial correlation, rises/falls, carry-over, machine and ball-set effects, with
p-values from simulated fair histories), repeats everything over several seeds, and
writes next-draw picks to `results/walk_forward.json`:

```bash
python scripts/fetch_merseyworld.py
python scripts/walk_forward_backtest.py --seeds 2026 1 2 3 4 --out results
python scripts/grammar_discovery.py   # results/grammar.json
python scripts/prime_radix_study.py   # results/prime_radix.json
python scripts/spectral_study.py      # results/spectral.json
python scripts/ml_ensemble_study.py   # results/ml_ensemble.json (needs scikit-learn)
python scripts/bayesian_study.py      # results/bayesian.json
python scripts/build_report.py        # results/report.html
```

`scripts/grammar_discovery.py` asks whether the draws follow any "grammar". For each era
(49 and 59 balls) it fits models on earlier draws and measures how many bits they need to
encode later, unseen draws, against the fair-machine limit (log2 of the number of possible
draws). The models are number frequency, frequency by draw position, drawn-order bigram and
trigram, echo of the previous round, per-machine frequency, a repo-style sorted grammar, and
transformers on drawn and sorted order. It also compresses the draw stream (lzma, bz2, zlib),
mines the most frequent pairs, triples and ordered pairs, and checks overlap with draws 1-10
rounds back. Each result is compared with simulated fair histories. Any real grammar would
show up as a model beating the limit, a stream compressing better than fair noise, or a rule
more frequent than chance allows.

`scripts/prime_radix_study.py` writes every ball in the first ten prime bases (as
`lotto_nn.prime_pipeline` does) and applies conditional probabilities and copulas to the
digits. It tests, per base and against simulated fair histories, whether a digit depends on
the previous ball's digit or on its count in the previous round, and whether digits are
rank-correlated (copula) within a draw or between consecutive rounds. It also backtests two
strategies per base, a conditional-probability (digit lift) strategy and a conditional Gaussian
copula over consecutive rounds' digits, alongside the repo prime-radix transformer.

### Tests

```bash
pip install -r requirements.txt
pytest -q
```

**Note:** This pipeline is for statistical/educational exploration only. Lottery draws are
random and independent; no model can predict future outcomes.

`scripts/spectral_study.py` treats each era as a time series of rounds and applies Fourier
and wavelet analysis. Periodograms (Fisher's g) look for regular rhythms in each ball's on/off
series, in summary series (sum, odd count, low-half count, smallest and largest ball, first
ball out, bonus) and in the stream of balls in drawn order. A Haar decomposition checks how
variance splits across time scales and looks for localized bursts, and a Morlet scalogram of
the draw sum looks for rhythms that switch on and off. Every statistic is compared with 1,000
simulated fair histories. It also backtests Fourier extrapolation (least-squares sinusoids at
zero-padded periodogram peaks, projected one round ahead) and wavelet-smoothed hotness.

`scripts/ml_ensemble_study.py` frames each round x number as "will this number be drawn?",
with 20 features computed only from earlier rounds (windowed frequencies, gaps, last-round
and bonus flags, usual drawing position, last machine's rate, parity/decade/residues, previous
sum). Gradient boosting (scikit-learn HistGradientBoosting), a random forest, logistic
regression and their soft-voting average are retrained walk-forward every 26 rounds in each
era. They are scored by held-out log-loss against the fair rate, per-round AUC, matches for
top-6 and probability-sampled tickets, and permutation importance.

`scripts/bayesian_study.py` compares a fair machine with one whose ball probabilities follow a
Dirichlet prior with unknown concentration (biases of roughly 3% to 100%), using
Dirichlet-multinomial marginal likelihoods. The Bayes factor and posterior probability of
fairness are calibrated against simulated fair histories. It also asks whether machines, ball
sets or calendar years favour different balls, gives each ball's 95% credible interval, tracks
the Bayes factor draw by draw, and backtests posterior-mean and Thompson-sampling strategies.

`scripts/build_report.py` also combines every investigation's next-draw picks into five lines
(written to `results/combined_picks.json` and shown at the top of the report). Each
investigation gets one equal vote, spread over its own lines; line 1 is the six numbers with the
most combined support, and lines 2-5 are drawn in proportion to that support.
