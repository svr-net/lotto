// The neural network overlaid on the particle filter, in plain C++: one small MLP shared by all 59 numbers
// (F features -> H -> H -> 1, tanh), whose output nudges each number's log-weight away from the filter's forecast.
// It is trained online, walk-forward, by Adam on the exact log-probability of each real draw's order, with a
// penalty on the size of its nudges. Its last layer starts at zero, so until the real draws teach it something it
// changes nothing. The learning rate and the penalty are not chosen by hand: the driver runs a family of networks
// across both and weights them by how well each predicted the real draws so far.
#pragma once
#include <cmath>
#include <vector>

#include "assimilate.hpp"

struct Mlp {
    static constexpr int F = 7, H = 16;
    // layout: W1[H][F] b1[H] W2[H][H] b2[H] W3[H] b3
    static constexpr int N1 = H * F, N2 = N1 + H, N3 = N2 + H * H, N4 = N3 + H, N5 = N4 + H, NP = N5 + 1;
    std::vector<double> p, m, v; long t = 0;
    double lr = 1e-3, decay = 0, penalty = 1.0;  // penalty: on the mean squared nudge
    explicit Mlp(uint64_t seed) : p(NP, 0.0), m(NP, 0.0), v(NP, 0.0) {
        Rng r{seed};
        for (int i = 0; i < N1; i++) p[i] = gauss(r) / std::sqrt(double(F));
        for (int i = N2; i < N3; i++) p[i] = gauss(r) / std::sqrt(double(H));
    }
    struct Cache { double x[F], h1[H], h2[H]; };
    double forward(const double* x, Cache& c) const {
        for (int i = 0; i < F; i++) c.x[i] = x[i];
        for (int j = 0; j < H; j++) { double s = p[N1 + j]; for (int i = 0; i < F; i++) s += p[j * F + i] * x[i]; c.h1[j] = std::tanh(s); }
        for (int j = 0; j < H; j++) { double s = p[N3 + j]; for (int i = 0; i < H; i++) s += p[N2 + j * H + i] * c.h1[i]; c.h2[j] = std::tanh(s); }
        double o = p[N5]; for (int j = 0; j < H; j++) o += p[N4 + j] * c.h2[j];
        return o;
    }
    void backward(const Cache& c, double dout, std::vector<double>& g) const {
        double d2[H], d1[H];
        g[N5] += dout;
        for (int j = 0; j < H; j++) { g[N4 + j] += dout * c.h2[j]; d2[j] = dout * p[N4 + j] * (1 - c.h2[j] * c.h2[j]); }
        for (int j = 0; j < H; j++) { g[N3 + j] += d2[j]; for (int i = 0; i < H; i++) g[N2 + j * H + i] += d2[j] * c.h1[i]; }
        for (int i = 0; i < H; i++) { double s = 0; for (int j = 0; j < H; j++) s += d2[j] * p[N2 + j * H + i]; d1[i] = s * (1 - c.h1[i] * c.h1[i]); }
        for (int j = 0; j < H; j++) { g[N1 + j] += d1[j]; for (int i = 0; i < F; i++) g[j * F + i] += d1[j] * c.x[i]; }
    }
    // the nudges for one night: 59 outputs, centred (only differences between numbers matter)
    void nudges(const double* X, double* s, Cache* cache = nullptr) const {
        Cache tmp; double mean = 0;
        for (int i = 0; i < POOL; i++) mean += s[i] = forward(X + i * F, cache ? cache[i] : tmp);
        mean /= POOL;
        for (int i = 0; i < POOL; i++) s[i] -= mean;
    }
    // adds the gradient of  -log P(order | base + nudges) + penalty * mean(nudge^2)  for one round; returns the log P
    double accumulate(const double* X, const double* base, const Order& o, std::vector<double>& g) const {
        Cache c[POOL]; double s[POOL], z[POOL], gz[POOL], ds[POOL], mean = 0;
        nudges(X, s, c);
        for (int i = 0; i < POOL; i++) z[i] = base[i] + s[i];
        double lp = pl_logprob(z, o, gz);
        for (int i = 0; i < POOL; i++) mean += ds[i] = -gz[i] + 2 * penalty * s[i] / POOL;
        mean /= POOL;
        for (int i = 0; i < POOL; i++) backward(c[i], ds[i] - mean, g);
        return lp;
    }
    void step(std::vector<double>& g, int batch) {  // Adam with decoupled weight decay
        t++;
        const double b1 = 0.9, b2 = 0.999, c1 = 1 - std::pow(b1, double(t)), c2 = 1 - std::pow(b2, double(t));
        for (int i = 0; i < NP; i++) {
            double gi = g[i] / batch;
            m[i] = b1 * m[i] + (1 - b1) * gi; v[i] = b2 * v[i] + (1 - b2) * gi * gi;
            p[i] -= lr * (m[i] / c1 / (std::sqrt(v[i] / c2) + 1e-8) + decay * p[i]);
            g[i] = 0;
        }
    }
};
