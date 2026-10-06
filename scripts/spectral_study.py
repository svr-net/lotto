"""Fourier and wavelet analysis of the UK Lotto draw sequence.

Usage:
    python scripts/spectral_study.py --csv data/uk_lotto_full.csv --out results/spectral.json

Each era (49 balls, 1994-2015; 59 balls, since 2015-10-10) is treated as a time
series of rounds. A fair machine makes every series below white noise, so any
periodicity, slow cycle, scale-specific variability or localized burst would be
structure. Every statistic is compared with the same statistic on simulated fair
histories of the same length.

Fourier (periodogram):
  - per-number appearance series (1 if the number was drawn that round): the
    strongest single periodicity across all numbers (Fisher's g)
  - per-number low-frequency power: slow "hot/cold" cycles
  - summary series (sum, odd count, low-half count, smallest and largest ball,
    first ball out, bonus): the strongest periodicity in each
  - the stream of balls in drawn order: strongest periodicity, and power at
    period 6 (the draw structure)
Wavelets:
  - Haar decomposition of the per-number series: how variance splits across
    time scales, compared with fair
  - largest standardized Haar detail coefficient: localized bursts or level shifts
  - Morlet scalogram of the draw-sum series: the strongest localized periodicity
Strategies (59-ball era, walk-forward, each round predicted from earlier rounds):
  - Fourier extrapolation: each number's appearance series over the last 256
    rounds is reduced to its strongest K harmonics and extended one round ahead
  - wavelet-smoothed hotness: each number's series is Haar-smoothed (fine detail
    levels removed) and the latest smoothed value used as its weight
"""
from __future__ import annotations

import argparse, json, math, os, time

import numpy as np
import pandas as pd

PICK = 6
DRAWN = ["d1", "d2", "d3", "d4", "d5", "d6"]
SETS = ["n1", "n2", "n3", "n4", "n5", "n6"]


# ---------------------------------------------------------------- transforms
def periodogram(x: np.ndarray) -> np.ndarray:
    """Periodogram ordinates for frequencies k/n, k = 1..n//2 (mean removed), along the last axis."""
    x = x - x.mean(axis=-1, keepdims=True)
    return np.abs(np.fft.rfft(x, axis=-1)[..., 1:]) ** 2 / x.shape[-1]


def fisher_g(I: np.ndarray) -> np.ndarray:
    """Largest periodogram ordinate as a share of the total (Fisher's g)."""
    return I.max(axis=-1) / I.sum(axis=-1)


def haar_dwt(x: np.ndarray, levels: int) -> list:
    """Orthonormal Haar DWT along the last axis (length must be divisible by 2**levels).
    Returns [detail level 1 (finest), ..., detail level `levels`, approximation]."""
    out, a = [], x.astype(float)
    for _ in range(levels):
        even, odd = a[..., 0::2], a[..., 1::2]
        out.append((even - odd) / math.sqrt(2))
        a = (even + odd) / math.sqrt(2)
    return out + [a]


def haar_idwt(coeffs: list) -> np.ndarray:
    a = coeffs[-1]
    for d in reversed(coeffs[:-1]):
        x = np.empty(a.shape[:-1] + (a.shape[-1] * 2,))
        x[..., 0::2] = (a + d) / math.sqrt(2)
        x[..., 1::2] = (a - d) / math.sqrt(2)
        a = x
    return a


def morlet_power(x: np.ndarray, scales: np.ndarray, w0: float = 6.0) -> np.ndarray:
    """Morlet continuous wavelet power |W(s, t)|^2 / var(x) (Torrence & Compo 1998), via FFT."""
    n = len(x)
    x = (x - x.mean()) / (x.std() + 1e-12)
    pad = 1 << int(math.ceil(math.log2(2 * n)))
    X = np.fft.fft(x, pad)
    w = 2 * math.pi * np.fft.fftfreq(pad)
    out = np.empty((len(scales), n))
    for i, s in enumerate(scales):
        psi = math.pi ** -0.25 * np.exp(-((s * w - w0) ** 2) / 2) * (w > 0) * math.sqrt(2 * math.pi * s)
        out[i] = np.abs(np.fft.ifft(X * psi)[:n]) ** 2
    return out


# ---------------------------------------------------------------- statistics of one history
def appearance(sets: np.ndarray, pool: int) -> np.ndarray:
    """[pool, rounds] 0/1 series: was number k+1 drawn in each round."""
    a = np.zeros((len(sets), pool + 1), dtype=np.int8)
    np.put_along_axis(a, sets, 1, axis=1)
    return a[:, 1:].T.astype(float)


def summary_series(sets: np.ndarray, drawn: np.ndarray, bonus: np.ndarray, pool: int) -> dict:
    return {
        "sum of the six balls": sets.sum(1),
        "number of odd balls": (sets % 2).sum(1),
        "balls in the low half": (sets <= pool // 2).sum(1),
        "smallest ball": sets.min(1),
        "largest ball": sets.max(1),
        "first ball out": drawn[:, 0],
        "bonus ball": bonus,
    }


def history_stats(sets, drawn, bonus, pool, levels, scales) -> dict:
    n = len(sets)
    A = appearance(sets, pool)
    I = periodogram(A)  # [pool, n//2]
    g = fisher_g(I)
    low = I[:, : max(1, I.shape[1] // 20)].mean(1) / I.mean(1)  # power in the lowest 5% of frequencies, relative
    m = 1 << int(math.log2(n))  # most recent power-of-two stretch for the Haar transform
    coeffs = haar_dwt(A[:, -m:] - A[:, -m:].mean(1, keepdims=True), levels)
    energy = np.array([(c ** 2).sum(1).mean() for c in coeffs[:-1]])  # mean energy per detail level
    energy = energy / energy.sum()
    sd = [c.std() + 1e-12 for c in coeffs[:-1]]
    burst = max(float(np.abs(c / s).max()) for c, s in zip(coeffs[:-1], sd))
    stats = {
        "number_period_g": float(g.max()), "number_period_which": int(g.argmax()) + 1,
        "number_period_freq": int(I[g.argmax()].argmax()) + 1,
        "number_lowfreq": float(low.mean()),
        "wavelet_energy": energy, "wavelet_burst": burst,
    }
    for name, s in summary_series(sets, drawn, bonus, pool).items():
        Is = periodogram(s.astype(float))
        stats["series_g:" + name] = float(fisher_g(Is))
        stats["series_freq:" + name] = int(Is.argmax()) + 1
    stream = drawn.ravel().astype(float)
    Ist = periodogram(stream)
    stats["stream_g"] = float(fisher_g(Ist))
    stats["stream_period6"] = float(Ist[len(stream) // 6 - 1] / Ist.mean())
    P = morlet_power(sets.sum(1).astype(float), scales)
    t = np.arange(n)
    coi = np.sqrt(2) * scales[:, None]
    inside = (t[None, :] > coi) & (t[None, :] < n - 1 - coi)
    Pm = np.where(inside, P, 0)
    i, j = np.unravel_index(Pm.argmax(), Pm.shape)
    stats["morlet_max"] = float(Pm.max())
    stats["morlet_where"] = [float(scales[i] * 1.033), int(j)]  # Fourier period ~1.033 x scale for w0=6
    return stats


def fair_history(rng, n, pool):
    d = np.argsort(rng.random((n, pool)), axis=1)[:, : PICK + 1] + 1
    return np.sort(d[:, :PICK], 1), d[:, :PICK], d[:, PICK]


TESTS = {
    "number_period_g": "a number reappears with a regular period",
    "number_lowfreq": "numbers run hot and cold in slow cycles",
    "wavelet_energy": "variability concentrates at some time scale (Haar)",
    "wavelet_burst": "a localized burst or level shift (Haar)",
    "stream_g": "the drawn-ball stream has a periodicity",
    "stream_period6": "the drawn-ball stream has a period-6 rhythm",
    "morlet_max": "the draw sum has a localized periodicity (Morlet)",
}


def era_tests(df: pd.DataFrame, pool: int, label: str, sims: int, seed: int) -> dict:
    df = df.sort_values(["draw_no", "round"]).reset_index(drop=True)
    sets, drawn, bonus = df[SETS].to_numpy(int), df[DRAWN].to_numpy(int), df.bonus.to_numpy(int)
    n = len(df)
    levels = 7
    scales = np.geomspace(2, n / 4, 24)
    obs = history_stats(sets, drawn, bonus, pool, levels, scales)
    rng = np.random.default_rng(seed)
    null = [history_stats(*fair_history(rng, n, pool), pool, levels, scales) for _ in range(sims)]
    series_names = list(summary_series(sets, drawn, bonus, pool))

    def pval(obs_v, null_v):
        null_v = np.array(null_v)
        return float(((null_v >= obs_v).sum() + 1) / (len(null_v) + 1))

    rows = []
    # wavelet energy profile: distance from the fair mean profile in fair-sd units
    E = np.array([s["wavelet_energy"] for s in null])
    mu, sd = E.mean(0), E.std(0) + 1e-12
    dist = lambda e: float((((e - mu) / sd) ** 2).sum())
    for key, label_t in TESTS.items():
        if key == "wavelet_energy":
            o, nv = dist(obs[key]), [dist(e) for e in E]
            detail = "scale shares " + ", ".join(f"{x:.3f}" for x in obs[key]) + " (fair " + ", ".join(f"{x:.3f}" for x in mu) + ")"
        else:
            o, nv = obs[key], [s[key] for s in null]
            detail = ""
            if key == "number_period_g":
                detail = f"strongest: ball {obs['number_period_which']}, every {n / obs['number_period_freq']:.1f} rounds"
            if key == "morlet_max":
                detail = f"period ~{obs['morlet_where'][0]:.0f} rounds around round {obs['morlet_where'][1]}"
        rows.append({"test": label_t, "key": key, "observed": o, "fair_mean": float(np.mean(nv)), "p": pval(o, nv), "detail": detail})
    for name in series_names:
        k = "series_g:" + name
        rows.append({"test": f"{name}: periodicity", "key": k, "observed": obs[k],
                     "fair_mean": float(np.mean([s[k] for s in null])), "p": pval(obs[k], [s[k] for s in null]),
                     "detail": f"strongest period {n / obs['series_freq:' + name]:.1f} rounds"})
    for r in rows:
        print(f"  p={r['p']:.3f}  {r['test']}  {r['detail']}")
    return {"label": label, "pool": pool, "rounds": n, "first_date": str(df.date.min()), "last_date": str(df.date.max()),
            "tests": rows, "sims": sims}


# ---------------------------------------------------------------- strategies
POOL = 59
EXP_MEAN = PICK * PICK / POOL
EXP_VAR = PICK * (PICK / POOL) * (1 - PICK / POOL) * (POOL - PICK) / (POOL - 1)


def weighted_tickets(weights, n, rng):
    p = np.clip(weights, 1e-6, None)
    p = p / p.sum()
    return np.array([np.sort(rng.choice(np.arange(1, POOL + 1), PICK, replace=False, p=p)) for _ in range(n)])


def fourier_forecast(A: np.ndarray, k: int, pad: int = 8) -> np.ndarray:
    """Extend each row one step past its end with its k strongest sinusoids.

    Frequencies are the k highest local peaks of a zero-padded periodogram, so
    periods need not divide the window. (Using the DFT's own frequencies would just
    wrap the window round: the "forecast" would be a smoothed copy of its first
    value.) Amplitudes and phases are then fitted by least squares and the fit is
    evaluated at t = n."""
    n = A.shape[1]
    t = np.arange(n)
    X = A - A.mean(1, keepdims=True)
    P = np.abs(np.fft.rfft(X, n * pad, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n * pad)
    out = np.empty(len(A))
    for r in range(len(A)):
        p = P[r]
        peaks = np.flatnonzero((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:])) + 1
        peaks = peaks[freqs[peaks] >= 1.0 / n]  # ignore drifts slower than the window
        f = freqs[peaks[np.argsort(-p[peaks])[:k]]]
        D = np.column_stack([np.ones(n)] + [g(2 * np.pi * fi * t) for fi in f for g in (np.cos, np.sin)])
        beta, *_ = np.linalg.lstsq(D, A[r], rcond=None)
        d_next = np.array([1.0] + [g(2 * np.pi * fi * n) for fi in f for g in (np.cos, np.sin)])
        out[r] = d_next @ beta
    return out


def fourier_strategy(k):
    def s(hist, n, rng):
        A = appearance(hist[-256:], POOL)
        return weighted_tickets(fourier_forecast(A, k) + 0.02, n, rng)
    return s


def wavelet_strategy(drop):
    def s(hist, n, rng):
        A = appearance(hist[-256:], POOL)
        c = haar_dwt(A, 8)
        c = [np.zeros_like(x) if i < drop else x for i, x in enumerate(c)]
        smooth = haar_idwt(c)[:, -1]
        return weighted_tickets(smooth + 0.02, n, rng)
    return s


STRATEGIES = {
    "random (baseline)": lambda hist, n, rng: weighted_tickets(np.ones(POOL), n, rng),
    "Fourier extrapolation, 3 strongest harmonics": fourier_strategy(3),
    "Fourier extrapolation, 10 strongest harmonics": fourier_strategy(10),
    "wavelet-smoothed hotness, ~8-round scale": wavelet_strategy(3),
    "wavelet-smoothed hotness, ~32-round scale": wavelet_strategy(5),
}


def backtest(df: pd.DataFrame, split: str, seeds: list, tickets: int) -> dict:
    sets = df[SETS].to_numpy(int)
    dates = pd.to_datetime(df.date).dt.date.to_numpy()
    split = pd.to_datetime(split).date()
    test_idx = [i for i, d in enumerate(dates) if d > split and (dates < d).sum() >= 256]
    res = {s: [] for s in STRATEGIES}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        per = {s: [] for s in STRATEGIES}
        for i in test_idx:
            hist = sets[dates < dates[i]]
            actual = set(sets[i].tolist())
            for s, fn in STRATEGIES.items():
                per[s].extend(len(actual & set(t.tolist())) for t in fn(hist, tickets, rng))
        for s in STRATEGIES:
            m = np.array(per[s])
            res[s].append(float((m.mean() - EXP_MEAN) / math.sqrt(EXP_VAR / len(m))))
    out = {s: {"z_mean": float(np.mean(z)), "z_min": float(np.min(z)), "z_max": float(np.max(z))} for s, z in res.items()}
    for s, r in out.items():
        print(f"    {s:48s} z {r['z_mean']:+.2f} [{r['z_min']:+.2f}, {r['z_max']:+.2f}]")
    rng = np.random.default_rng(seeds[0])
    picks = {s: [sorted(int(x) for x in t) for t in fn(sets, 3, rng)] for s, fn in STRATEGIES.items() if not s.startswith("random")}
    return {"strategies": out, "test_rounds": len(test_idx), "seeds": seeds, "tickets_per_round": tickets, "picks": picks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/uk_lotto_full.csv")
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--split-date", default="2022-12-31")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 1, 2, 3, 4])
    ap.add_argument("--tickets", type=int, default=20)
    ap.add_argument("--out", default="results/spectral.json")
    a = ap.parse_args()
    t0 = time.time()
    df = pd.read_csv(a.csv)
    e59 = df[df.draw_no >= 2066].sort_values(["draw_no", "round"]).reset_index(drop=True)
    e49 = df[df.draw_no < 2066]
    eras = []
    for sub, pool, label, seed in [(e59, 59, "59-ball era (Oct 2015 - now)", a.seed), (e49, 49, "49-ball era (1994 - Oct 2015)", a.seed + 1)]:
        print(f"== {label}")
        eras.append(era_tests(sub, pool, label, a.sims, seed))
    print("== strategies (59-ball era, walk-forward)")
    bt = backtest(e59, a.split_date, a.seeds, a.tickets)
    n_tests = sum(len(e["tests"]) for e in eras)
    with open(a.out, "w") as fh:
        json.dump({"eras": eras, "backtest": bt, "n_tests": n_tests, "split": a.split_date}, fh, indent=1, default=float)
    print(f"wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
