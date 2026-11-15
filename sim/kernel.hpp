// The fused Monte Carlo kernel, shared by the native driver (assimilate.cpp) and the browser build (web.cpp).
//
// One pass per simulated path draws its rounds (six of 59 balls, uniformly or in proportion to weights that can
// learn from the path's own draws), scores every line with AND + popcount on 64-bit masks and, at each checkpoint,
// adds the path's totals straight into one flat histogram (Layout). Nothing per path is stored, so a job of any
// size returns a few kilobytes, and every statistic the drivers need (means, quantiles, ranks, shares) is read
// from the histograms (Hist).
//
// The same kernel is written twice, once in C++ (run_paths) and once in WGSL (WGSL below, one invocation per
// path), with the same splitmix64 stream (64-bit arithmetic emulated on 32-bit lanes in WGSL) and the same float
// operations in the same order, so the CPU and WebGPU backends give identical histograms. Backends are injected
// through Backend: CpuBackend here, ThreadsBackend (threads.hpp) natively, and WebGPU in the browser (web.cpp
// packs gpu_plan() for the page, which only runs the compute pass).
#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <functional>
#include <vector>

constexpr int POOL = 59, PICK = 6, MAX_CHECKPOINTS = 4;
constexpr int PAY[7] = {0, 0, 2, 30, 140, 1750, 0};  // fixed prizes per line by matches; 2 = a free Lucky Dip
constexpr uint32_t MONEY_BINS = 8192;                // money histogram: 2 a bin (every prize is even); the last bin holds the rest
using Mask = uint64_t;                               // bit n-1 set for number n
struct Wheel { std::vector<Mask> lines; Mask set = 0; };
struct Job {  // paths futures of `horizon` rounds from `weights`, totals recorded at each checkpoint (ascending, <= horizon)
    std::array<float, POOL> weights; bool weighted; int paths, horizon; const Wheel* wheel;
    std::vector<int> checkpoints; uint64_t seed;
    bool learn = true;  // weighted paths add each of their own draws to the weights (false: weights stay fixed)
};

struct Rng {  // splitmix64
    uint64_t s;
    uint64_t next() { uint64_t z = (s += 0x9E3779B97F4A7C15ull); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
                      z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
    uint32_t below(uint32_t n) { return uint32_t(((next() >> 32) * uint64_t(n)) >> 32); }
    float unit() { return float(next() >> 40) * (1.0f / 16777216.0f); }
};
constexpr uint64_t PATH_SEED = 0xD1B54A32D192ED03ull;  // path p starts from seed ^ (p * PATH_SEED)

// ---------------------------------------------------------------- histograms
struct Layout {  // k1 [7], then per checkpoint: prize rounds [horizon + 1], rounds with 4+ of the set [horizon + 1], money [MONEY_BINS]
    uint32_t nck = 0, horizon = 0;
    uint32_t stride() const { return 2 * (horizon + 1) + MONEY_BINS; }
    uint32_t k1() const { return 0; }
    uint32_t prize(uint32_t c) const { return PICK + 1 + c * stride(); }
    uint32_t hits4(uint32_t c) const { return prize(c) + horizon + 1; }
    uint32_t money(uint32_t c) const { return hits4(c) + horizon + 1; }
    uint32_t size() const { return PICK + 1 + nck * stride(); }
};
struct Hist {  // counts of the values 0, scale, 2 scale, ...
    const uint32_t* c; uint32_t bins, scale;
    double total() const { double n = 0; for (uint32_t i = 0; i < bins; i++) n += c[i]; return n; }
    double count(double v) const { uint32_t i = uint32_t(v / scale); return v >= 0 && i < bins && i * double(scale) == v ? c[i] : 0; }
    double below(double v) const { double n = 0; for (uint32_t i = 0; i < bins && i * double(scale) < v; i++) n += c[i]; return n; }
    double mean() const { double s = 0, n = 0; for (uint32_t i = 0; i < bins; i++) { s += double(c[i]) * i * scale; n += c[i]; } return n ? s / n : 0; }
    double at(double k) const {  // the k-th smallest value, from 0
        double n = 0; for (uint32_t i = 0; i < bins; i++) if ((n += c[i]) > k) return double(i) * scale;
        return double(bins - 1) * scale;
    }
    double quantile(double q) const {  // as sorting every path and interpolating between neighbours
        double n = total(), pos = q * (n - 1), i = double(uint64_t(pos)), v = at(i);
        return i + 1 < n ? v + (pos - i) * (at(i + 1) - v) : at(n - 1);
    }
};
struct Result {
    Layout L; std::vector<uint32_t> h; int paths;
    Hist k1() const { return {h.data() + L.k1(), PICK + 1, 1}; }
    Hist prize(uint32_t c) const { return {h.data() + L.prize(c), L.horizon + 1, 1}; }
    Hist hits4(uint32_t c) const { return {h.data() + L.hits4(c), L.horizon + 1, 1}; }
    Hist money(uint32_t c) const { return {h.data() + L.money(c), MONEY_BINS, 2}; }
};

// ---------------------------------------------------------------- the fused kernel, CPU
// Paths [from, to) of job j, added into histogram h; top[c] is raised to one past the highest money bin touched at
// checkpoint c, so a caller merging h only needs that much of each money histogram.
inline void run_paths(const Job& j, const Layout& L, int from, int to, uint32_t* h, uint32_t* top) {
    const Mask* lines = j.wheel->lines.data(); const size_t nl = j.wheel->lines.size(); const Mask set = j.wheel->set;
    const uint32_t nck = L.nck; const int* cks = j.checkpoints.data();
    std::array<float, POOL> w;
    for (int p = from; p < to; p++) {
        Rng rng{j.seed ^ (uint64_t(p) * PATH_SEED)};
        if (j.weighted) w = j.weights;
        uint32_t prize = 0, money = 0, hits4 = 0, ck = 0;
        for (int round = 1; round <= j.horizon; round++) {
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
            for (size_t l = 0; l < nl; l++) { int m = std::popcount(draw & lines[l]); money += PAY[m]; best = std::max(best, m); }
            int k = std::popcount(draw & set);
            prize += best >= 3; hits4 += k >= 4;
            if (round == 1) h[L.k1() + k]++;
            for (; ck < nck && cks[ck] == round; ck++) {
                uint32_t b = std::min(money / 2, MONEY_BINS - 1);
                h[L.prize(ck) + prize]++; h[L.hits4(ck) + hits4]++; h[L.money(ck) + b]++;
                top[ck] = std::max(top[ck], b + 1);
            }
        }
    }
}

struct Backend {
    virtual ~Backend() = default;
    virtual const char* name() const = 0;
    virtual void run(const Job& j, Result& r) = 0;  // adds j's paths into r.h (zeroed, sized r.L.size())
    // other kernels: body(from, to) over chunks of [0, n), in parallel where the backend can
    virtual void parallel(int n, const std::function<void(int, int)>& body) { if (n > 0) body(0, n); }
    static Result empty(const Job& j) { Layout L{uint32_t(j.checkpoints.size()), uint32_t(j.horizon)}; return {L, std::vector<uint32_t>(L.size()), j.paths}; }
    Result simulate(const Job& j) { Result r = empty(j); run(j, r); return r; }
};
struct CpuBackend final : Backend {
    const char* name() const override { return "cpu"; }
    void run(const Job& j, Result& r) override { uint32_t top[MAX_CHECKPOINTS] = {}; run_paths(j, r.L, 0, j.paths, r.h.data(), top); }
};

// The forward summary both drivers report, per checkpoint: 5%, 50%, 95% of the prize rounds and of the net result
// (money less cost_per_round a round), and the share of paths with no prize.
inline void summarise(const Result& r, const std::vector<int>& checkpoints, double cost_per_round, float* out) {
    for (uint32_t c = 0; c < r.L.nck; c++) {
        Hist pr = r.prize(c), m = r.money(c); const double cost = cost_per_round * checkpoints[c];
        float* o = out + c * 7;
        o[0] = float(pr.quantile(0.05)); o[1] = float(pr.quantile(0.5)); o[2] = float(pr.quantile(0.95));
        o[3] = float(m.quantile(0.05) - cost); o[4] = float(m.quantile(0.5) - cost); o[5] = float(m.quantile(0.95) - cost);
        o[6] = float(pr.count(0) / r.paths);
    }
}

// ---------------------------------------------------------------- the fused kernel, WebGPU
// The same kernel as a WGSL compute shader: one invocation per path, histogram bins added with atomicAdd, the
// GPU's lock-free read-modify-write. splitmix64 runs on (lo, hi) pairs of u32.
inline constexpr const char* WGSL = R"(
struct P { paths: u32, horizon: u32, nlines: u32, weighted: u32, learn: u32, nck: u32, seed_lo: u32, seed_hi: u32,
           set_lo: u32, set_hi: u32, money_bins: u32, pad: u32, cks: vec4<u32> };
@group(0) @binding(0) var<uniform> prm: P;
@group(0) @binding(1) var<storage, read> lines: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> weights: array<f32, 59>;
@group(0) @binding(3) var<storage, read_write> hist: array<atomic<u32>>;
const PAY = array<u32, 7>(0u, 0u, 2u, 30u, 140u, 1750u, 0u);
fn mul32(a: u32, b: u32) -> vec2<u32> {  // the full 64-bit product of two u32
  let a0 = a & 0xFFFFu; let a1 = a >> 16u; let b0 = b & 0xFFFFu; let b1 = b >> 16u;
  let p00 = a0 * b0; let p01 = a0 * b1; let p10 = a1 * b0;
  let mid = (p00 >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
  return vec2<u32>((p00 & 0xFFFFu) | (mid << 16u), a1 * b1 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u));
}
fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { var r = mul32(a.x, b.x); r.y += a.x * b.y + a.y * b.x; return r; }
fn mix(z: vec2<u32>, s: u32) -> vec2<u32> { return z ^ vec2<u32>((z.x >> s) | (z.y << (32u - s)), z.y >> s); }
var<private> st: vec2<u32>;
fn next() -> vec2<u32> {
  let lo = st.x + 0x7F4A7C15u; st = vec2<u32>(lo, st.y + 0x9E3779B9u + select(0u, 1u, lo < st.x));
  var z = mul64(mix(st, 30u), vec2<u32>(0x1CE4E5B9u, 0xBF58476Du));
  z = mul64(mix(z, 27u), vec2<u32>(0x133111EBu, 0x94D049BBu));
  return mix(z, 31u);
}
fn below(n: u32) -> u32 { return mul32(next().y, n).y; }
fn unit() -> f32 { return f32(next().y >> 8u) * (1.0 / 16777216.0); }
fn has(lo: u32, hi: u32, i: u32) -> bool { if (i < 32u) { return ((lo >> i) & 1u) == 1u; } return ((hi >> (i - 32u)) & 1u) == 1u; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= prm.paths) { return; }
  st = vec2<u32>(prm.seed_lo, prm.seed_hi) ^ mul64(vec2<u32>(p, 0u), vec2<u32>(0xD192ED03u, 0xD1B54A32u));
  let stride = 2u * (prm.horizon + 1u) + prm.money_bins;
  var w: array<f32, 59>;
  if (prm.weighted == 1u) { for (var i = 0u; i < 59u; i++) { w[i] = weights[i]; } }
  var prize = 0u; var money = 0u; var hits4 = 0u; var ck = 0u;
  for (var h = 1u; h <= prm.horizon; h++) {
    var lo = 0u; var hi = 0u;
    if (prm.weighted == 0u) {
      var n = 0u;
      loop { if (n >= 6u) { break; } let i = below(59u);
        if (!has(lo, hi, i)) { if (i < 32u) { lo |= 1u << i; } else { hi |= 1u << (i - 32u); } n++; } }
    } else {
      var total = 0.0;
      for (var i = 0u; i < 59u; i++) { total += w[i]; }
      for (var n = 0u; n < 6u; n++) {
        var u = unit() * total; var i = 0u;
        loop { if (i >= 58u) { break; } if (!has(lo, hi, i)) { u -= w[i]; if (u < 0.0) { break; } } i++; }
        loop { if (!has(lo, hi, i)) { break; } i--; }
        if (i < 32u) { lo |= 1u << i; } else { hi |= 1u << (i - 32u); }
        total -= w[i];
      }
      if (prm.learn == 1u) { for (var i = 0u; i < 59u; i++) { if (has(lo, hi, i)) { w[i] += 1.0; } } }
    }
    var best = 0u;
    for (var l = 0u; l < prm.nlines; l++) {
      let m = countOneBits(lines[l].x & lo) + countOneBits(lines[l].y & hi);
      money += PAY[m]; best = max(best, m);
    }
    let k = countOneBits(prm.set_lo & lo) + countOneBits(prm.set_hi & hi);
    if (best >= 3u) { prize++; }
    if (k >= 4u) { hits4++; }
    if (h == 1u) { atomicAdd(&hist[k], 1u); }
    loop { if (ck >= prm.nck || prm.cks[ck] != h) { break; }
      let base = 7u + ck * stride;
      atomicAdd(&hist[base + prize], 1u);
      atomicAdd(&hist[base + prm.horizon + 1u + hits4], 1u);
      atomicAdd(&hist[base + 2u * (prm.horizon + 1u) + min(money / 2u, prm.money_bins - 1u)], 1u);
      ck++; }
  }
}
)";

struct GpuPlan {  // the bytes of one compute pass: bindings 0..3 of WGSL, and the workgroups to dispatch
    std::array<uint32_t, 16> params; std::vector<uint32_t> lines; std::array<float, POOL> weights; uint32_t workgroups, hist_words;
};
inline GpuPlan gpu_plan(const Job& j) {
    const Wheel& w = *j.wheel; Layout L{uint32_t(j.checkpoints.size()), uint32_t(j.horizon)};
    GpuPlan g{};
    g.params = {uint32_t(j.paths), uint32_t(j.horizon), uint32_t(w.lines.size()), j.weighted, j.learn, L.nck, uint32_t(j.seed), uint32_t(j.seed >> 32),
                uint32_t(w.set), uint32_t(w.set >> 32), MONEY_BINS, 0, 0, 0, 0, 0};
    for (uint32_t c = 0; c < L.nck; c++) g.params[12 + c] = uint32_t(j.checkpoints[c]);
    for (Mask m : w.lines) { g.lines.push_back(uint32_t(m)); g.lines.push_back(uint32_t(m >> 32)); }
    g.weights = j.weights; g.workgroups = (uint32_t(j.paths) + 63) / 64; g.hist_words = L.size();
    return g;
}
