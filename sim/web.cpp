// The browser build of the wheel simulator: kernel.hpp compiled to WebAssembly (clang, wasm32-wasi), so a
// phone can run the forward forecast itself. Two backends, chosen by the page:
//   cpu     run_cpu(): the fused kernel in WebAssembly;
//   webgpu  the same kernel as a WGSL compute shader, one invocation per simulated path. C++ owns the
//           shader and packs its inputs (gpu_prepare); the page runs the compute pass, the one step only
//           JavaScript can do in a browser, and hands the raw output back (gpu_finish).
// Either way summarise() writes, for each checkpoint (26, 52, 104 rounds): the 5%, 50% and 95% quantiles of
// the rounds with a 3+ line and of the net result, and the share of futures with no prize (7 floats each).
#include <cstring>

#include "kernel.hpp"

#define EXPORT(name) __attribute__((export_name(#name))) name

constexpr int MAX_LINES = 256, HORIZON = 104, NCK = 3;
static const std::vector<int> CHECKPOINTS = {26, 52, 104};
static uint32_t lines_in[MAX_LINES * 2];  // each line's 64-bit mask as (low, high)
static float weights_in[POOL];
static uint32_t gpu_params[8], gpu_cks[4] = {26, 52, 104, 0};
static std::vector<int32_t> gpu_out;      // per path and checkpoint: (prize rounds, money)
static Result result;
static float summary[NCK * 7];

static const char* WGSL = R"(
struct P { paths: u32, horizon: u32, nlines: u32, weighted: u32, set_lo: u32, set_hi: u32, nck: u32, seed: u32 };
@group(0) @binding(0) var<uniform> prm: P;
@group(0) @binding(1) var<storage, read> lines: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> weights: array<f32, 59>;
@group(0) @binding(3) var<storage, read> cks: array<u32, 4>;
@group(0) @binding(4) var<storage, read_write> out: array<i32>;
const PAY = array<i32, 7>(0, 0, 2, 30, 140, 1750, 0);
var<private> st: u32;
fn rnd() -> f32 { st ^= st << 13u; st ^= st >> 17u; st ^= st << 5u; return f32(st >> 8u) / 16777216.0; }
fn has(lo: u32, hi: u32, i: u32) -> bool { if (i < 32u) { return ((lo >> i) & 1u) == 1u; } return ((hi >> (i - 32u)) & 1u) == 1u; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= prm.paths) { return; }
  st = prm.seed ^ (p * 0x9E3779B9u + 0x7F4A7C15u); if (st == 0u) { st = 1u; }
  for (var i = 0u; i < 4u; i++) { _ = rnd(); }
  var w: array<f32, 59>;
  if (prm.weighted == 1u) { for (var i = 0u; i < 59u; i++) { w[i] = weights[i]; } }
  var prize = 0i; var money = 0i; var ck = 0u;
  for (var h = 1u; h <= prm.horizon; h++) {
    var lo = 0u; var hi = 0u;
    if (prm.weighted == 0u) {
      var n = 0u;
      loop { if (n >= 6u) { break; } let i = min(u32(rnd() * 59.0), 58u);
        if (!has(lo, hi, i)) { if (i < 32u) { lo |= 1u << i; } else { hi |= 1u << (i - 32u); } n++; } }
    } else {
      var total = 0.0;
      for (var i = 0u; i < 59u; i++) { total += w[i]; }
      for (var n = 0u; n < 6u; n++) {
        var u = rnd() * total; var i = 0u;
        loop { if (i >= 58u) { break; } if (!has(lo, hi, i)) { u -= w[i]; if (u < 0.0) { break; } } i++; }
        loop { if (!has(lo, hi, i)) { break; } i--; }
        if (i < 32u) { lo |= 1u << i; } else { hi |= 1u << (i - 32u); }
        total -= w[i];
      }
      for (var i = 0u; i < 59u; i++) { if (has(lo, hi, i)) { w[i] += 1.0; } }
    }
    var best = 0u;
    for (var l = 0u; l < prm.nlines; l++) {
      let m = countOneBits(lines[l].x & lo) + countOneBits(lines[l].y & hi);
      money += PAY[m]; best = max(best, m);
    }
    if (best >= 3u) { prize++; }
    loop { if (ck >= prm.nck || cks[ck] != h) { break; }
      let b = (p * prm.nck + ck) * 2u; out[b] = prize; out[b + 1u] = money; ck++; }
  }
})";

static Wheel wheel(int nlines) {
    Wheel w;
    for (int i = 0; i < nlines; i++) { Mask m = Mask(lines_in[i * 2]) | Mask(lines_in[i * 2 + 1]) << 32; w.lines.push_back(m); w.set |= m; }
    return w;
}

extern "C" {
uint32_t* EXPORT(lines_ptr)() { return lines_in; }
float* EXPORT(weights_ptr)() { return weights_in; }
float* EXPORT(summary_ptr)() { return summary; }
const char* EXPORT(gpu_shader)() { return WGSL; }
uint32_t* EXPORT(gpu_params_ptr)() { return gpu_params; }
uint32_t* EXPORT(gpu_checkpoints_ptr)() { return gpu_cks; }

void EXPORT(run_cpu)(int nlines, int weighted, int paths, uint32_t seed) {
    Wheel w = wheel(nlines);
    Job job{{}, weighted != 0, paths, HORIZON, &w, CHECKPOINTS, seed};
    std::memcpy(job.weights.data(), weights_in, sizeof weights_in);
    result = CpuBackend().simulate(job);
}

// Packs the shader's uniform block and returns where the page must write its output (paths * 3 * 2 i32).
int32_t* EXPORT(gpu_prepare)(int nlines, int weighted, int paths, uint32_t seed) {
    Wheel w = wheel(nlines);
    const uint32_t prm[8] = {uint32_t(paths), HORIZON, uint32_t(nlines), weighted ? 1u : 0u, uint32_t(w.set), uint32_t(w.set >> 32), NCK, seed | 1u};
    std::memcpy(gpu_params, prm, sizeof prm);
    gpu_out.assign(size_t(paths) * NCK * 2, 0);
    return gpu_out.data();
}

void EXPORT(gpu_finish)(int paths) {
    size_t n = size_t(paths) * NCK;
    result = {std::vector<int>(n), std::vector<int>(n), {}, {}};
    for (size_t i = 0; i < n; i++) { result.prize[i] = gpu_out[i * 2]; result.money[i] = gpu_out[i * 2 + 1]; }
}

void EXPORT(summarise)(int paths, float cost_per_round) {
    for (int c = 0; c < NCK; c++) {
        std::vector<double> prize(paths), net(paths); double none = 0;
        for (int p = 0; p < paths; p++) { prize[p] = result.prize[p * NCK + c]; net[p] = result.money[p * NCK + c] - double(cost_per_round) * CHECKPOINTS[c]; none += prize[p] == 0; }
        float* o = summary + c * 7;
        o[0] = quantile(prize, 0.05); o[1] = quantile(prize, 0.5); o[2] = quantile(prize, 0.95);
        o[3] = quantile(net, 0.05); o[4] = quantile(net, 0.5); o[5] = quantile(net, 0.95); o[6] = none / paths;
    }
}
}
