# lotto

Research on UK Lotto (6 from 59) draws: can any model beat chance? Every study so far says no. The
results are published as a report at https://svr-net.github.io/lotto/.

## What's here

- `data/uk_lotto_full.csv`: every draw since 1994 (`scripts/fetch_merseyworld.py` updates it).
- `scripts/`: one script per study, each writing `results/<study>.json`:
  - `walk_forward_backtest.py`: 14 strategies and 6 ensembles, walk-forward on real draws;
  - `grammar_discovery.py`, `prime_radix_study.py`, `radix_strategies_study.py`, `spectral_study.py`,
    `ml_ensemble_study.py`, `bayesian_study.py`, `rl_study.py`, `popularity_study.py`: the studies;
  - `sequence_study.py`: the draws as one sequence (memory, timing, waits);
  - `wheel_study.py`, `monte_carlo_study.py`: the 59-number wheel on real and simulated draws;
  - `consensus_wheels.py`: the consensus 15 numbers in a "3 if 4" and a "4 if 4" wheel;
  - `replay_wheel.py`: the wheel rebuilt and scored for each of the last three months' draws;
  - `score_live.py`: scores the committed wheels against each new draw;
  - `freeze_wheel.py`: freezes a wheel's lines (`results/frozen_wheel.json`) and exports them;
  - `build_report.py`: builds `results/report.html` from `results/report_template.html`.
- `sim/nested_mc.cpp`: nested Monte Carlo in C++, reset to the real draws at every step.
- `lotto_nn/`: the small transformer and n-gram models the studies use.
- `tests/`: `python -m pytest`.

## Running

```sh
pip install -r requirements.txt
docker build --network=host --output type=local,dest=sim/bin sim   # builds sim/bin/nested_mc
python -I scripts/fetch_merseyworld.py && python scripts/score_live.py
python scripts/walk_forward_backtest.py --out results               # then each study script
./sim/bin/nested_mc --first-draw 2026-11-07
python scripts/build_report.py
```

Merging to `main` publishes the report to GitHub Pages (`.github/workflows/pages.yml`).

Every line of six numbers has the same chance of winning. Nothing here changes that.
