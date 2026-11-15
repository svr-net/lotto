// The browser build: a thin set of exports over kernel.hpp, compiled to WebAssembly (clang, wasm32-wasi). Nothing
// is calculated here. The page writes the wheel and a forecast's ball probabilities into the staging buffers, picks
// a backend, and reads back 7 floats per checkpoint (26, 52, 104 rounds; see summarise in kernel.hpp).
//   cpu     run_cpu(): the fused kernel in WebAssembly.
//   webgpu  gpu_prepare() lays out the kernel's WGSL and its four bindings (gpu_plan) as a descriptor the page
//           turns into one compute pass without knowing what is in it; the page copies binding 3, the
//           histograms, back where the descriptor says and calls gpu_finish().
// Both backends fill the same histograms from the same random stream, so they give the same answer.
#include "kernel.hpp"

#define EXPORT(name) __attribute__((export_name(#name))) name

constexpr int MAX_LINES = 256, HORIZON = 104;
static const std::vector<int> CHECKPOINTS = {26, 52, 104};
static uint32_t lines_in[MAX_LINES * 2];  // each line's 64-bit mask as (low, high)
static float weights_in[POOL];
static Wheel wheel;
static Job job;
static Result result;
static GpuPlan plan;
static uint32_t descriptor[16];  // shader ptr, length, workgroups, 0, then (ptr, bytes, kind) per binding: 0 uniform, 1 read, 2 read-write
static float summary[MAX_CHECKPOINTS * 7];

static void stage(int nlines, int weighted, int paths, uint32_t seed) {
    wheel = {};
    for (int i = 0; i < nlines && i < MAX_LINES; i++) { Mask m = Mask(lines_in[i * 2]) | Mask(lines_in[i * 2 + 1]) << 32; wheel.lines.push_back(m); wheel.set |= m; }
    job = {{}, weighted != 0, paths, HORIZON, &wheel, CHECKPOINTS, seed};
    job.learn = false;  // the weights are a forecast's ball probabilities: they stay as given
    for (int i = 0; i < POOL; i++) job.weights[i] = weights_in[i];
    result = Backend::empty(job);
}
static float* finish(float cost_per_round) { summarise(result, CHECKPOINTS, cost_per_round, summary); return summary; }

extern "C" {
uint32_t* EXPORT(lines_ptr)() { return lines_in; }
float* EXPORT(weights_ptr)() { return weights_in; }
uint32_t* EXPORT(hist_ptr)() { return result.h.data(); }
uint32_t EXPORT(hist_words)() { return uint32_t(result.h.size()); }

float* EXPORT(run_cpu)(int nlines, int weighted, int paths, uint32_t seed, float cost_per_round) {
    stage(nlines, weighted, paths, seed);
    CpuBackend().run(job, result);
    return finish(cost_per_round);
}
uint32_t* EXPORT(gpu_prepare)(int nlines, int weighted, int paths, uint32_t seed) {
    stage(nlines, weighted, paths, seed);
    plan = gpu_plan(job);
    auto ptr = [](const void* p) { return uint32_t(reinterpret_cast<uintptr_t>(p)); };
    uint32_t d[16] = {ptr(WGSL), uint32_t(__builtin_strlen(WGSL)), plan.workgroups, 0,
                      ptr(plan.params.data()), uint32_t(sizeof plan.params), 0,
                      ptr(plan.lines.data()), uint32_t(plan.lines.size() * 4), 1,
                      ptr(plan.weights.data()), uint32_t(sizeof plan.weights), 1,
                      ptr(result.h.data()), plan.hist_words * 4, 2};
    for (int i = 0; i < 16; i++) descriptor[i] = d[i];
    return descriptor;
}
float* EXPORT(gpu_finish)(float cost_per_round) { return finish(cost_per_round); }
}
