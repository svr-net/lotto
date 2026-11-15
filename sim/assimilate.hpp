// Fused kernels for the particle filter (sequential Monte Carlo) that resets to every real draw.
//
// The hidden state is the machine: 59 log-weights theta, balls drawn one at a time in proportion to e^theta
// (Plackett-Luce), drifting every round (AR(1): theta <- rho theta + sqrt(1 - rho^2) tau eps) so the filter can
// follow a machine or ball set that changes; tau and rho are learned per particle. Each real round updates every
// particle on the exact probability of the order the balls came out in, adds the filter's evidence, and, when the
// weights have collapsed, resamples and moves the particles. Every kernel works one
// particle or one sample per index through Backend::parallel, with its own random stream, so results never
// depend on the backend or the number of threads.
#pragma once
#include <cmath>
#include <numeric>

#include "data.hpp"
#include "models.hpp"

inline double u53(Rng& r) { return double(r.next() >> 11) * (1.0 / 9007199254740992.0); }
inline double gauss(Rng& r) { double u = std::max(1e-300, u53(r)), v = u53(r); return std::sqrt(-2 * std::log(u)) * std::cos(6.283185307179586 * v); }
inline uint64_t stream(uint64_t a, uint64_t b, uint64_t c = 0) { Rng r{a ^ (b * 0x9E3779B97F4A7C15ull) ^ (c * PATH_SEED)}; r.next(); return r.next(); }

// log P(order | logits z): the six balls drawn one at a time, each in proportion to e^z among those left.
// grad (optional) receives d log P / d z.
inline double pl_logprob(const double* z, const Order& o, double* grad = nullptr) {
    double mx = z[0]; for (int i = 1; i < POOL; i++) mx = std::max(mx, z[i]);
    double e[POOL], total = 0;
    for (int i = 0; i < POOL; i++) total += e[i] = std::exp(z[i] - mx);
    if (grad) std::fill(grad, grad + POOL, 0.0);
    double lp = 0; Mask used = 0;
    for (int k = 0; k < PICK; k++) {
        int i = o[k];
        lp += std::log(e[i] / total);
        if (grad) { for (int j = 0; j < POOL; j++) if (!(used >> j & 1)) grad[j] -= e[j] / total; grad[i] += 1; }
        used |= Mask(1) << i; total -= e[i];
    }
    return lp;
}
inline void softmax(const double* z, double* q) {
    double mx = z[0]; for (int i = 1; i < POOL; i++) mx = std::max(mx, z[i]);
    double t = 0; for (int i = 0; i < POOL; i++) t += q[i] = std::exp(z[i] - mx);
    for (int i = 0; i < POOL; i++) q[i] /= t;
}
inline double logsumexp(const std::vector<double>& v) {
    double mx = *std::max_element(v.begin(), v.end()), s = 0;
    for (double x : v) s += std::exp(x - mx);
    return mx + std::log(s);
}

// ---------------------------------------------------------------- the filter
// Rao-Blackwellised: each particle carries only the two parameters of how the machine drifts, both learned from the
// draws (log-uniform priors): the bias scale tau (0.001..0.5) and the drift memory L rounds (rho = 1 - 1/L,
// 20..20000). Given those, the 59 log-weights are tracked per particle as a Gaussian (mean m, variance v per
// number): predicted forward by the drift, then updated by each real draw with a Laplace step on its exact
// ordered likelihood (one Newton step from the predicted mean, diagonal curvature). The particle is weighted by
// the Laplace approximation of that draw's marginal likelihood, so the bias scale the draws support wins: a
// particle with too large a tau is not penalised for guessing 59 numbers at random, only for its spread.
constexpr int LOG_TAU = 0, LOG_L = 1;
struct Filter {
    int P;
    std::vector<double> par;   // P x 2: log tau, log L
    std::vector<double> m, v;  // P x POOL: the Gaussian over the log-weights
    std::vector<double> logw;  // normalised log weights
    double log_evidence = 0;   // log P(every round so far | the filter), from the ordered draws
    int resamples = 0;
    Filter(int P_, uint64_t seed, Backend& be) : P(P_), par(size_t(P_) * 2), m(size_t(P_) * POOL, 0.0), v(size_t(P_) * POOL), logw(P_, -std::log(double(P_))) {
        be.parallel(P, [&](int a, int b) {
            for (int p = a; p < b; p++) {
                Rng r{stream(seed, uint64_t(p))};
                par[p * 2 + LOG_TAU] = std::log(1e-3) + u53(r) * std::log(500.0);
                par[p * 2 + LOG_L] = std::log(20.0) + u53(r) * std::log(1000.0);
                double t = tau(p); for (int i = 0; i < POOL; i++) v[size_t(p) * POOL + i] = t * t;
            }
        });
    }
    const double* theta(int p) const { return &m[size_t(p) * POOL]; }
    const double* var(int p) const { return &v[size_t(p) * POOL]; }
    double weight(int p) const { return std::exp(logw[p]); }
    double tau(int p) const { return std::exp(par[p * 2 + LOG_TAU]); }
    double memory(int p) const { return std::exp(par[p * 2 + LOG_L]); }
};

// One real round, fused per particle: drift the Gaussian, Laplace-update it on the exact ordered likelihood and
// score the draw. Then (serially, so every backend gives the same result) the evidence, the new weights and, when the
// effective sample size falls below half, systematic resampling and a Liu-West move of (log tau, log L) whose
// bandwidth follows the effective sample size (Silverman's rule per coordinate, h = 1.06 ESS^-1/5).
inline void assimilate(Filter& f, const Order& order, uint64_t seed, Backend& be) {
    const int P = f.P;
    std::vector<double> ll(P);
    be.parallel(P, [&](int a, int b) {
        for (int p = a; p < b; p++) {
            double* mm = &f.m[size_t(p) * POOL]; double* vv = &f.v[size_t(p) * POOL];
            const double rho = 1 - 1 / f.memory(p), t = f.tau(p), q = (1 - rho * rho) * t * t;
            for (int i = 0; i < POOL; i++) { mm[i] *= rho; vv[i] = rho * rho * vv[i] + q; }
            // curvature of the ordered likelihood at the predicted mean (diagonal), gradient, one Newton step
            double g[POOL], h[POOL] = {}, e[POOL], total = 0, mx = mm[0];
            for (int i = 1; i < POOL; i++) mx = std::max(mx, mm[i]);
            for (int i = 0; i < POOL; i++) total += e[i] = std::exp(mm[i] - mx);
            Mask used = 0;
            for (int k = 0; k < PICK; k++) { for (int j = 0; j < POOL; j++) if (!(used >> j & 1)) { double pj = e[j] / total; h[j] += pj * (1 - pj); } used |= Mask(1) << order[k]; total -= e[order[k]]; }
            pl_logprob(mm, order, g);
            double z[POOL], lap = 0;
            for (int i = 0; i < POOL; i++) {
                double vn = 1 / (1 / vv[i] + h[i]), step = vn * g[i];
                z[i] = mm[i] + step;
                lap += -0.5 * step * step / vv[i] + 0.5 * std::log(vn / vv[i]);
                vv[i] = vn;
            }
            ll[p] = pl_logprob(z, order) + lap;  // Laplace: likelihood at the updated mean, the prior's pull, the volume
            std::copy(z, z + POOL, mm);
        }
    });
    std::vector<double> nw(P);
    for (int p = 0; p < P; p++) nw[p] = f.logw[p] + ll[p];
    double z = logsumexp(nw);
    f.log_evidence += z;
    double ess_inv = 0;
    for (int p = 0; p < P; p++) { f.logw[p] = nw[p] - z; ess_inv += std::exp(2 * f.logw[p]); }
    const double ess = 1 / ess_inv;
    if (ess >= P / 2.0) return;
    double mean[2] = {}, var[2] = {};
    for (int p = 0; p < P; p++) { double w = f.weight(p); for (int i = 0; i < 2; i++) mean[i] += w * f.par[p * 2 + i]; }
    for (int p = 0; p < P; p++) { double w = f.weight(p); for (int i = 0; i < 2; i++) { double d = f.par[p * 2 + i] - mean[i]; var[i] += w * d * d; } }
    Rng r{stream(seed, 0xA11, 2)};
    std::vector<int> pick(P); double step = 1.0 / P, u = u53(r) * step, c = 0; int j = 0;
    for (int p = 0; p < P; p++) { while (j < P - 1 && c + f.weight(j) < u) c += f.weight(j++); pick[p] = j; u += step; }
    const std::vector<double> op = f.par, om = f.m, ov = f.v;
    const double h = std::min(0.9, 1.06 * std::pow(ess, -0.2)), a = std::sqrt(1 - h * h);
    be.parallel(P, [&](int lo, int hi) {
        for (int p = lo; p < hi; p++) {
            Rng g{stream(seed, uint64_t(p), 3)}; const int s = pick[p];
            for (int i = 0; i < 2; i++) f.par[p * 2 + i] = a * op[s * 2 + i] + (1 - a) * mean[i] + h * std::sqrt(var[i]) * gauss(g);
            f.par[p * 2 + LOG_TAU] = std::clamp(f.par[p * 2 + LOG_TAU], std::log(1e-4), std::log(2.0));
            f.par[p * 2 + LOG_L] = std::clamp(f.par[p * 2 + LOG_L], std::log(2.0), std::log(1e6));
            std::copy(&om[size_t(s) * POOL], &om[size_t(s) * POOL] + POOL, &f.m[size_t(p) * POOL]);
            std::copy(&ov[size_t(s) * POOL], &ov[size_t(s) * POOL] + POOL, &f.v[size_t(p) * POOL]);
        }
    });
    std::fill(f.logw.begin(), f.logw.end(), -std::log(double(P)));
    f.resamples++;
}
inline double weighted_quantile(std::vector<std::pair<double, double>> vw, double q) {  // (value, weight)
    std::sort(vw.begin(), vw.end());
    double t = 0; for (auto& [v, w] : vw) t += w;
    double c = 0; for (auto& [v, w] : vw) if ((c += w) >= q * t) return v;
    return vw.back().first;
}

// ---------------------------------------------------------------- predictive mixtures
// A forecast as a mixture of Plackett-Luce machines: component c has weight w[c] and ball probabilities q[c].
struct Mixture {
    std::vector<double> w; std::vector<double> q;  // q: components x POOL
    int size() const { return int(w.size()); }
    const double* at(int c) const { return &q[size_t(c) * POOL]; }
    static Mixture single(const double* p) { Mixture m; m.w = {1}; m.q.assign(p, p + POOL); return m; }
    static Mixture fair() { double u[POOL]; std::fill(u, u + POOL, 1.0 / POOL); return single(u); }
    void add(const Mixture& o, double scale) { for (int c = 0; c < o.size(); c++) w.push_back(scale * o.w[c]); q.insert(q.end(), o.q.begin(), o.q.end()); }
    std::array<double, POOL> marginal() const {  // weighted mean ball probabilities
        std::array<double, POOL> m{}; for (int c = 0; c < size(); c++) for (int i = 0; i < POOL; i++) m[i] += w[c] * q[size_t(c) * POOL + i];
        return m;
    }
};
// The filter's particles as one mixture of machines
inline Mixture filter_mixture(const Filter& f, Backend& be) {
    Mixture m; m.w.resize(f.P); m.q.resize(size_t(f.P) * POOL);
    be.parallel(f.P, [&](int a, int b) { for (int p = a; p < b; p++) { m.w[p] = f.weight(p); softmax(f.theta(p), &m.q[size_t(p) * POOL]); } });
    return m;
}
// log P(set) under a mixture, exactly: each component's sum over the 720 orders, mixed
inline double mixture_set_logprob(const Mixture& m, Mask draw, Backend& be) {
    const double fair = -std::log(choose(POOL, PICK));
    std::vector<double> r(m.size());
    be.parallel(m.size(), [&](int a, int b) {
        for (int c = a; c < b; c++) { Probs q; std::copy(m.at(c), m.at(c) + POOL, q.begin()); r[c] = std::exp(set_logprob(q, draw) - fair); }
    });
    double s = 0; for (int c = 0; c < m.size(); c++) s += m.w[c] * r[c];
    return fair + std::log(s);
}
// M draws of six from the mixture (component by weight, then balls one at a time), one stream per sample
inline std::vector<Mask> sample_sets(const Mixture& m, int M, uint64_t seed, Backend& be) {
    std::vector<double> cum(m.size()); double t = 0;
    for (int c = 0; c < m.size(); c++) cum[c] = t += m.w[c];
    std::vector<Mask> out(M);
    be.parallel(M, [&](int a, int b) {
        for (int s = a; s < b; s++) {
            Rng r{stream(seed, uint64_t(s), 4)};
            int c = int(std::upper_bound(cum.begin(), cum.end(), u53(r) * t) - cum.begin());
            const double* q = m.at(std::min(c, m.size() - 1));
            Mask d = 0; double tot = 0; for (int i = 0; i < POOL; i++) tot += q[i];
            for (int n = 0; n < PICK; n++) {
                double u = u53(r) * tot; int i = 0;
                for (; i < POOL - 1; i++) { if (d >> i & 1) continue; u -= q[i]; if (u < 0) break; }
                while (d >> i & 1) i--;
                d |= Mask(1) << i; tot -= q[i];
            }
            out[s] = d;
        }
    });
    return out;
}

// ---------------------------------------------------------------- choosing the 15
// The 3-if-4 wheel has a 3+ line whenever 4 of the six are among its 15, so the 15 to play are those most likely to
// hold 4+. Hill-climb by single swaps on simulated draws: the gain of swapping a (in) for b (out) counts the draws
// with 3 of the 15 that hold b but not a (they rise to 4) less those with exactly 4 that hold a but not b (they
// fall to 3), so each pass is exact on the samples and costs one sweep.
inline int count4(const std::vector<Mask>& draws, Mask set) { int n = 0; for (Mask d : draws) n += std::popcount(d & set) >= 4; return n; }
inline Mask best_fifteen(const std::vector<Mask>& draws, Mask set, int max_swaps = 60) {
    for (int it = 0; it < max_swaps; it++) {
        static thread_local int gain[POOL][POOL];
        for (auto& row : gain) std::fill(row, row + POOL, 0);
        for (Mask d : draws) {
            int k = std::popcount(d & set);
            if (k == 3) {
                for (Mask in = d & ~set; in; in &= in - 1) for (Mask out = set & ~d; out; out &= out - 1) gain[std::countr_zero(out)][std::countr_zero(in)]++;
            } else if (k == 4) {
                Mask others = ~set & ~d & ((Mask(1) << POOL) - 1);
                for (Mask out = d & set; out; out &= out - 1) for (Mask in = others; in; in &= in - 1) gain[std::countr_zero(out)][std::countr_zero(in)]--;
            }
        }
        int ba = -1, bb = -1, bg = 0;
        for (Mask out = set; out; out &= out - 1) for (int b = 0; b < POOL; b++) {
            int a = std::countr_zero(out);
            if (!(set >> b & 1) && gain[a][b] > bg) { bg = gain[a][b]; ba = a; bb = b; }
        }
        if (ba < 0) break;
        set = (set & ~(Mask(1) << ba)) | (Mask(1) << bb);
    }
    return set;
}
inline Mask top_set(const std::array<double, POOL>& q, int k) {  // the k largest, ties to the lower number
    std::array<int, POOL> o; std::iota(o.begin(), o.end(), 0);
    std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return q[a] > q[b]; });
    Mask m = 0; for (int i = 0; i < k; i++) m |= Mask(1) << o[i];
    return m;
}

// ---------------------------------------------------------------- exact chance rates and tests
inline double fair_p4() { double p = 0; for (int k = 4; k <= PICK; k++) p += fair_k(SET, k); return p; }
inline double fair_p_line3() {  // a 3-if-4 wheel holds a 3+ line: every subset of its 15 points, weighted by the ways to fill the rest
    uint32_t lines[15]; for (int l = 0; l < 15; l++) { lines[l] = 0; for (int i : W34[l]) lines[l] |= 1u << i; }
    double hit = 0;
    for (uint32_t s = 0; s < (1u << SET); s++) {
        int k = std::popcount(s); if (k > PICK) continue;
        bool any = false; for (uint32_t l : lines) any |= std::popcount(l & s) >= 3;
        if (any) hit += choose(POOL - SET, PICK - k);
    }
    return hit / choose(POOL, PICK);
}
inline double binom_sf(int n, int h, double p) {  // P(X >= h), X ~ Binomial(n, p)
    double s = 0;
    for (int x = h; x <= n; x++) s += std::exp(std::lgamma(n + 1.0) - std::lgamma(x + 1.0) - std::lgamma(n - x + 1.0) + x * std::log(p) + (n - x) * std::log1p(-p));
    return std::min(1.0, s);
}
inline double sign_flip_p(const std::vector<double>& d, int B, uint64_t seed) {  // one-sided: P(sum with random signs >= the real sum)
    double obs = 0; for (double x : d) obs += x;
    Rng r{seed}; int ge = 0;
    for (int b = 0; b < B; b++) { double s = 0; for (double x : d) s += (r.next() >> 63) ? x : -x; ge += s >= obs - 1e-12; }
    return (ge + 1.0) / (B + 1.0);
}
