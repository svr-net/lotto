# lotto

One 15-number wheel for the UK Lotto (6 from 59), chosen by radix digits, a reverse-engineered machine and a neural
network over a particle filter, learned from every real draw and scored walk-forward against chance. The report is published at
https://svr-net.github.io/lotto/.

## What's here

- `data/uk_lotto_full.csv`: every draw since 1994, with the order the balls came out (`scripts/fetch_merseyworld.py`
  updates it).
- `sim/`: everything is computed in C++, built on Alpine by `sim/Dockerfile`.
  - `assimilate.cpp`: the driver. Before every real draw night of the 59-ball era it makes five forecasts from the
    draws before it, and every round that night resets them:
    - a Rao-Blackwellised particle filter (`assimilate.hpp`): the machine as a hidden state, its bias scale and drift
      learned, updated on the exact probability of each draw's order;
    - a family of 18 small neural networks (`nn.hpp`, MLP + Adam; two widths, three learning rates, three penalties)
      nudging the filter's forecast per number from its uncertainty, gaps, recent and first-ball counts, the radix
      digits and the machine forecast, weighted by their real predictions;
    - 80 radix strategies (`radix.hpp`): every number's digits in the first ten prime bases, hot, cold, overdue and
      early-ball rules, weighted by their real predictions;
    - the reverse-engineered machine (`reverse.hpp`): physical hypotheses (a bias per machine, per ball set, per
      pairing; repeats, neighbours and final digits of the last round), each a Bayesian model updated on every draw,
      with a forecast of which machine and ball set run next learned from the schedule;
    - fair.
    The mixture of the five, at equal prior weight and reweighted by every real draw, chooses the 15 numbers most
    likely to hold 4+ of the six; they are laid out as a "3 if 4" wheel (15 lines, a 3-match line whenever 4 of the
    six are among the 15). The same pipeline runs on simulated machines with a planted bias, to show what it finds
    when there is something to find (on every machine, or on one machine only). Every hypothesis, ball position
    included, is also tested on its own against 100 fair histories with the real machines and ball sets.
    Nothing is tuned by hand.
  - `kernel.hpp`, `threads.hpp`, `models.hpp`, `data.hpp`: the fused Monte Carlo kernel (C++ and WGSL), the lock-free
    multi-core backend, the wheel and exact probabilities, the data reader.
  - `web.cpp`: a thin WebAssembly wrapper the report calls to simulate the wheel on the viewer's device, on the CPU or
    on WebGPU; both give the same histograms.
- `results/assimilation.json`: the results, including the wheel's round-by-round record over the last six months
  (the report's "Last 6 months" tab); `results/exports/`: `wheel_app.txt` and `wheel.csv` (the wheel's lines)
  and `lotto_sim.wasm`.
- `scripts/build_report.py` puts the results into `results/report_template.html` -> `results/report.html`.
- `tests/`: `python -m pytest` (runs the C++ binaries).

## Running

```sh
docker build --network=host --target native --output type=local,dest=sim/bin sim       # sim/bin/assimilate (static)
docker build --network=host --target web --output type=local,dest=results/exports sim  # lotto_sim.wasm
python -I scripts/fetch_merseyworld.py
./sim/bin/assimilate                    # results/assimilation.json and the wheel's exports, ~5 min
python scripts/build_report.py
```

Merging to `main` publishes the report to GitHub Pages (`.github/workflows/pages.yml`).

Every line of six numbers has the same chance of winning. Nothing here changes that.
