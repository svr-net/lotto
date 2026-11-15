// The fused Monte Carlo kernel shared by the native driver (nested_mc.cpp) and the browser build (web.cpp).
//
// One pass per simulated round draws six of 59 balls (uniformly, or in proportion to per-path weights that
// can learn from the path's own draws) and scores every line with AND + popcount on 64-bit masks. Backends
// implement Backend::run; CpuBackend is the single-threaded reference.
#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <vector>

constexpr int POOL = 59, PICK = 6;
constexpr int PAY[7] = {0, 0, 2, 30, 140, 1750, 0};  // fixed prizes per line by matches; 2 = a free Lucky Dip
using Mask = uint64_t;  // bit n-1 set for number n
struct Wheel { std::vector<Mask> lines; Mask set = 0; };
struct Job {  // paths futures of `horizon` rounds from `weights`, totals recorded at each checkpoint
    std::array<float, POOL> weights; bool weighted; int paths, horizon; const Wheel* wheel;
    std::vector<int> checkpoints; uint64_t seed;
    bool learn = true;  // weighted paths add each of their own draws to the weights (false: weights stay fixed)
};
struct Result {  // path-major: [path * checkpoints + c]
    std::vector<int> prize, money, hits4, k1;
};

struct Rng {  // splitmix64
    uint64_t s;
    uint64_t next() { uint64_t z = (s += 0x9E3779B97F4A7C15ull); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
                      z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
    uint32_t below(uint32_t n) { return uint32_t(((next() >> 32) * uint64_t(n)) >> 32); }
    float unit() { return float(next() >> 40) * (1.0f / 16777216.0f); }
};

inline void run_paths(const Job& j, Result& r, int from, int to) {
    const size_t nck = j.checkpoints.size();
    std::array<float, POOL> w;
    for (int p = from; p < to; p++) {
        Rng rng{j.seed ^ (uint64_t(p) * 0xD1B54A32D192ED03ull)};
        if (j.weighted) w = j.weights;
        int prize = 0, money = 0, hits4 = 0; size_t ck = 0;
        for (int h = 1; h <= j.horizon; h++) {
            Mask draw = 0;
            if (!j.weighted) {
                for (int n = 0; n < PICK;) { Mask b = Mask(1) << rng.below(POOL); if (!(draw & b)) { draw |= b; n++; } }
            } else {
                float total = 0;
                for (float v : w) total += v;
                for (int n = 0; n < PICK; n++) {
                    float u = rng.unit() * total; int i = 0;
                    for (; i < POOL - 1; i++) { if (draw >> i & 1) continue; u -= w[i]; if (u < 0) break; }
                    while (draw >> i & 1) i--;  // float round-off at the end of the scan
                    draw |= Mask(1) << i; total -= w[i];
                }
                if (j.learn) for (int i = 0; i < POOL; i++) if (draw >> i & 1) w[i] += 1;  // the path learns from its own draws
            }
            int best = 0;
            for (Mask l : j.wheel->lines) { int m = std::popcount(draw & l); money += PAY[m]; best = std::max(best, m); }
            int k = std::popcount(draw & j.wheel->set);
            prize += best >= 3; hits4 += k >= 4;
            if (h == 1) r.k1[p] = k;
            while (ck < nck && j.checkpoints[ck] == h) {
                r.prize[p * nck + ck] = prize; r.money[p * nck + ck] = money; r.hits4[p * nck + ck] = hits4; ck++;
            }
        }
    }
}

struct Backend {
    virtual ~Backend() = default;
    virtual const char* name() const = 0;
    virtual void run(const Job& j, Result& r) = 0;
    Result simulate(const Job& j) {
        size_t n = size_t(j.paths) * j.checkpoints.size();
        Result r{std::vector<int>(n), std::vector<int>(n), std::vector<int>(n), std::vector<int>(j.paths)};
        run(j, r);
        return r;
    }
};
struct CpuBackend : Backend {
    const char* name() const override { return "cpu"; }
    void run(const Job& j, Result& r) override { run_paths(j, r, 0, j.paths); }
};

inline double quantile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    double pos = q * (v.size() - 1); size_t i = size_t(pos);
    return i + 1 < v.size() ? v[i] + (pos - i) * (v[i + 1] - v[i]) : v.back();
}
