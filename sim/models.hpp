// The model-side calculations: the 3-if-4 wheel, the exact probability of a drawn set under a weighted machine, and
// the statistics that score forecasts against the real draws.
#pragma once
#include <cmath>
#include <string>

#include "kernel.hpp"

constexpr int SET = 15;
// 15 lines on 15 points in which any 4 points share 3 with some line ("3 if 4"); as in scripts/consensus_wheels.py
constexpr int W34[15][6] = {{0, 1, 6, 10, 12, 14}, {0, 2, 4, 9, 10, 11}, {3, 8, 9, 10, 13, 14}, {1, 2, 3, 4, 6, 8},
    {2, 5, 8, 11, 12, 14}, {0, 4, 5, 6, 8, 13}, {5, 6, 7, 8, 9, 12}, {1, 2, 5, 7, 10, 13}, {3, 5, 6, 7, 10, 11},
    {3, 4, 7, 10, 12, 14}, {1, 4, 9, 11, 12, 13}, {0, 1, 7, 8, 11, 14}, {0, 1, 3, 5, 9, 14}, {0, 2, 3, 7, 12, 13},
    {2, 6, 9, 11, 13, 14}};

inline Wheel wheel_of(Mask set) {  // the 3-if-4 design on the 15 numbers of `set`
    int nums[SET], c = 0;
    for (int i = 0; i < POOL && c < SET; i++) if (set >> i & 1) nums[c++] = i;
    Wheel w;
    for (auto& row : W34) { Mask m = 0; for (int i : row) m |= Mask(1) << nums[i]; w.lines.push_back(m); }
    w.set = set;
    return w;
}

// ---------------------------------------------------------------- statistics
inline double mean(const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return v.empty() ? 0 : s / v.size(); }
inline double choose(int n, int k) { double r = 1; for (int i = 0; i < k; i++) r = r * (n - i) / (i + 1); return r; }
inline double fair_k(int set, int k) { return choose(set, k) * choose(POOL - set, PICK - k) / choose(POOL, PICK); }  // k of the six among `set` numbers
inline double zstat(const std::vector<double>& v) {  // mean over its standard error
    double m = mean(v), s = 0; for (double x : v) s += (x - m) * (x - m);
    return v.size() > 1 && s > 0 ? m / std::sqrt(s / (v.size() - 1) / v.size()) : 0;
}
// ---------------------------------------------------------------- weighted machines
using Probs = std::array<double, POOL>;
inline double set_logprob(const Probs& q, Mask draw) {  // exact: summed over the 720 orders the six could come out in
    int idx[PICK], c = 0;
    for (int i = 0; i < POOL; i++) if (draw >> i & 1) idx[c++] = i;
    double f[1 << PICK] = {1}, used[1 << PICK] = {0};
    for (int m = 1; m < (1 << PICK); m++) {
        used[m] = used[m & (m - 1)] + q[idx[std::countr_zero(unsigned(m))]];
        for (int x = 0; x < PICK; x++) if (m >> x & 1) { int prev = m ^ (1 << x); f[m] += f[prev] * q[idx[x]] / std::max(1e-12, 1 - used[prev]); }
    }
    return std::log(f[(1 << PICK) - 1]);
}
