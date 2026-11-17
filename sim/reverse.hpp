// Reverse-engineering the draws: physical hypotheses about how the machines produce them, each a Bayesian model
// learned walk-forward from the exact probability of every draw's order, and a forecast of which machine and ball
// set will run next.
//
// Hypotheses (each with its own scale tau of how far from fair it lets the numbers stray, tau over a grid weighted
// by the evidence, equal prior):
//   machine            each machine has its own fixed bias over the 59 numbers
//   ball set           each ball set has its own fixed bias (worn or heavier balls)
//   machine x set      each pairing has its own bias
//   ball position      numbers have different chances of coming out 1st, 2nd, ... 6th (a weight effect)
//   repeats            numbers drawn in the last round are more (or less) likely next
//   neighbours         numbers next to the last round's numbers are more (or less) likely
//   final digit        numbers sharing a final digit with the last round's numbers are more (or less) likely
// Each is tracked as a Gaussian over its parameters, updated by a Laplace step on each real draw (the same update as
// the particle filter's, without drift), which also gives that draw's marginal likelihood: the sum over rounds is the
// hypothesis's evidence against fair. The evidence of a hypothesis that is not true is below zero on average, but its
// spread depends on the model, so it is calibrated by running the same hypothesis on fair simulated histories with the
// real machines and ball sets (reverse_test), which gives each a p-value.
#pragma once
#include <cmath>
#include <map>

#include "assimilate.hpp"

constexpr double RTAUS[] = {0.01, 0.02, 0.05, 0.1, 0.2};
constexpr int NT = int(std::size(RTAUS));
inline double tau_logev(const double* lev) { double v[NT]; for (int t = 0; t < NT; t++) v[t] = lev[t] - std::log(double(NT)); double mx = *std::max_element(v, v + NT), s = 0; for (double x : v) s += std::exp(x - mx); return mx + std::log(s); }
inline void tau_post(const double* lev, double* w) { double mx = *std::max_element(lev, lev + NT), s = 0; for (int t = 0; t < NT; t++) s += w[t] = std::exp(lev[t] - mx); for (int t = 0; t < NT; t++) w[t] /= s; }

// One Laplace step for a static Gaussian (m, v) over the 59 log-weights on one ordered draw; returns its log marginal
// likelihood (likelihood at the updated mean, the prior's pull and the change of volume).
inline double laplace_numbers(double* m, double* v, const Order& o) {
    double g[POOL], h[POOL] = {}, e[POOL], total = 0, mx = m[0];
    for (int i = 1; i < POOL; i++) mx = std::max(mx, m[i]);
    for (int i = 0; i < POOL; i++) total += e[i] = std::exp(m[i] - mx);
    Mask used = 0;
    for (int k = 0; k < PICK; k++) { for (int j = 0; j < POOL; j++) if (!(used >> j & 1)) { double pj = e[j] / total; h[j] += pj * (1 - pj); } used |= Mask(1) << o[k]; total -= e[o[k]]; }
    pl_logprob(m, o, g);
    double z[POOL], lap = 0;
    for (int i = 0; i < POOL; i++) {
        double vn = 1 / (1 / v[i] + h[i]), step = vn * g[i];
        z[i] = m[i] + step; lap += -0.5 * step * step / v[i] + 0.5 * std::log(vn / v[i]); v[i] = vn;
    }
    double lp = pl_logprob(z, o) + lap;
    std::copy(z, z + POOL, m);
    return lp;
}
// The same for one position k of the order only: logits m over the numbers still in the machine
inline double laplace_position(double* m, double* v, const Order& o, int k) {
    Mask used = 0; for (int j = 0; j < k; j++) used |= Mask(1) << o[j];
    auto lse = [&](const double* z) {
        double mx = -1e300, s = 0;
        for (int j = 0; j < POOL; j++) if (!(used >> j & 1)) mx = std::max(mx, z[j]);
        for (int j = 0; j < POOL; j++) if (!(used >> j & 1)) s += std::exp(z[j] - mx);
        return mx + std::log(s);
    };
    double l = lse(m), z[POOL], lap = 0;
    for (int j = 0; j < POOL; j++) {
        if (used >> j & 1) { z[j] = m[j]; continue; }
        double p = std::exp(m[j] - l), g = (j == o[k]) - p, vn = 1 / (1 / v[j] + p * (1 - p)), step = vn * g;
        z[j] = m[j] + step; lap += -0.5 * step * step / v[j] + 0.5 * std::log(vn / v[j]); v[j] = vn;
    }
    double lp = z[o[k]] - lse(z) + lap;
    std::copy(z, z + POOL, m);
    return lp;
}
// One scalar effect beta on a feature x of every number (logits beta x): Laplace on one ordered draw
inline double laplace_scalar(double& mb, double& vb, const double* x, const Order& o) {
    auto stats = [&](double b, double& g, double& h) {  // log P(order), its gradient and curvature in b
        Mask used = 0; double lp = 0; g = 0; h = 0;
        for (int k = 0; k < PICK; k++) {
            double mx = -1e300; for (int j = 0; j < POOL; j++) if (!(used >> j & 1)) mx = std::max(mx, b * x[j]);
            double s = 0, sx = 0, sxx = 0;
            for (int j = 0; j < POOL; j++) if (!(used >> j & 1)) { double e = std::exp(b * x[j] - mx); s += e; sx += e * x[j]; sxx += e * x[j] * x[j]; }
            double ex = sx / s; lp += b * x[o[k]] - mx - std::log(s); g += x[o[k]] - ex; h += sxx / s - ex * ex;
            used |= Mask(1) << o[k];
        }
        return lp;
    };
    double g, h; stats(mb, g, h);
    double vn = 1 / (1 / vb + h), step = vn * g, z = mb + step, lap = -0.5 * step * step / vb + 0.5 * std::log(vn / vb);
    double g2, h2, lp = stats(z, g2, h2) + lap;
    mb = z; vb = vn;
    return lp;
}

// ---------------------------------------------------------------- the hypotheses
enum Hyp { MACHINE, BALL_SET, PAIR, POSITION, REPEATS, NEIGHBOURS, DIGIT, NHYP };
inline const char* HYP_NAMES[NHYP] = {"machine", "ball set", "machine x set", "ball position", "repeats", "neighbours", "final digit"};
inline bool is_group(int h) { return h <= PAIR; }
inline bool is_scalar(int h) { return h >= REPEATS; }
inline std::string group_key(int h, const Round& r) { return h == MACHINE ? r.machine : h == BALL_SET ? r.ball_set : r.machine + "|" + r.ball_set; }
inline void scalar_feature(int h, Mask last, double* x) {  // the feature of every number, from the last round before the night
    for (int i = 0; i < POOL; i++) {
        if (h == REPEATS) x[i] = last >> i & 1;
        else if (h == NEIGHBOURS) x[i] = !(last >> i & 1) && ((i > 0 && (last >> (i - 1) & 1)) || (i + 1 < POOL && (last >> (i + 1) & 1)));
        else { x[i] = 0; for (int j = 0; j < POOL; j++) if ((last >> j & 1) && j != i && (j + 1) % 10 == (i + 1) % 10) x[i] = 1; }
    }
}

struct Gauss { std::vector<double> m, v; };
struct HypState {  // one hypothesis, all its groups or positions, at every tau
    int h;
    std::map<std::string, std::array<Gauss, NT>> groups;
    std::array<std::array<Gauss, PICK>, NT> pos;   // ball position: per tau, per position
    double mb[NT] = {}, vb[NT] = {};                // scalar effects
    double lev[NT] = {};                            // log evidence per tau
    explicit HypState(int hyp) : h(hyp) {
        for (int t = 0; t < NT; t++) { mb[t] = 0; vb[t] = RTAUS[t] * RTAUS[t]; for (auto& g : pos[t]) g = {std::vector<double>(POOL, 0.0), std::vector<double>(POOL, RTAUS[t] * RTAUS[t])}; }
    }
    std::array<Gauss, NT>& group(const std::string& k) {
        auto it = groups.find(k);
        if (it == groups.end()) { std::array<Gauss, NT> a; for (int t = 0; t < NT; t++) a[t] = {std::vector<double>(POOL, 0.0), std::vector<double>(POOL, RTAUS[t] * RTAUS[t])}; it = groups.emplace(k, a).first; }
        return it->second;
    }
    // learn one real round; `last` is the last round's draw before its night (what a forecast for the night could use)
    void update(const Round& r, Mask last) {
        double x[POOL];
        if (is_scalar(h)) scalar_feature(h, last, x);
        for (int t = 0; t < NT; t++) {
            if (is_group(h)) { Gauss& g = group(group_key(h, r))[t]; lev[t] += laplace_numbers(g.m.data(), g.v.data(), r.order); }
            else if (h == POSITION) for (int k = 0; k < PICK; k++) lev[t] += laplace_position(pos[t][k].m.data(), pos[t][k].v.data(), r.order, k);
            else lev[t] += laplace_scalar(mb[t], vb[t], x, r.order);
        }
    }
    double evidence() const { return tau_logev(lev); }  // log P(every order so far | hypothesis), tau averaged
    // the forecast for a round of the coming night as a mixture of machines, given the chance of each group
    void forecast(const std::map<std::string, double>& context, Mask last, Mixture& out) const {
        double w[NT]; tau_post(lev, w);
        out = {};
        if (is_group(h)) {
            for (auto& [k, pk] : context) {
                auto it = groups.find(k);
                for (int t = 0; t < NT; t++) {
                    if (pk * w[t] < 1e-9) continue;
                    double q[POOL]; if (it == groups.end()) std::fill(q, q + POOL, 1.0 / POOL); else softmax(it->second[t].m.data(), q);
                    out.w.push_back(pk * w[t]); out.q.insert(out.q.end(), q, q + POOL);
                }
            }
        } else if (is_scalar(h)) {
            double x[POOL]; scalar_feature(h, last, x);
            for (int t = 0; t < NT; t++) { double z[POOL], q[POOL]; for (int i = 0; i < POOL; i++) z[i] = mb[t] * x[i]; softmax(z, q); out.w.push_back(w[t]); out.q.insert(out.q.end(), q, q + POOL); }
        }
    }
};

// ---------------------------------------------------------------- which machine and ball set run next
// Learned from the schedule: with chance pi the k-th round of a night repeats the k-th round's group of the last night
// (pi counted from the history, a uniform prior), otherwise the group is drawn by its recent frequency, the window
// (10, 50 or all rounds) weighted by how well each predicted the groups so far.
struct ContextModel {
    int h; double rep = 1, trials = 2, lw[3] = {};
    std::vector<std::string> hist;  // the group of every round so far
    static constexpr int WIN[3] = {10, 50, 1 << 30};
    explicit ContextModel(int hyp) : h(hyp) {}
    std::map<std::string, double> freq(int W) const {
        std::map<std::string, double> f; int n = 0;
        for (int t = int(hist.size()) - 1; t >= 0 && n < W; t--, n++) f[hist[t]] += 1;
        for (auto& k : hist) f.emplace(k, 0.0);  // every group seen so far keeps a little chance
        double tot = 0; for (auto& [k, c] : f) tot += c += 0.5;
        for (auto& [k, c] : f) c /= tot;
        return f;
    }
    // the chance of each group for the round in slot `slot` of the coming night; prev: last night's groups
    std::map<std::string, double> predict(const std::vector<std::string>& prev, int slot) const {
        std::map<std::string, double> p;
        if (hist.empty()) return p;
        double a[3], mx = *std::max_element(lw, lw + 3), s = 0;
        for (int i = 0; i < 3; i++) s += a[i] = std::exp(lw[i] - mx);
        for (int i = 0; i < 3; i++) for (auto& [k, c] : freq(WIN[i])) p[k] += a[i] / s * c;
        if (slot < int(prev.size())) { double pi = rep / trials; for (auto& [k, c] : p) c *= 1 - pi; p[prev[slot]] += pi; }
        return p;
    }
    // learn from a round in slot `slot`: its group and how each window would have predicted it
    void update(const std::string& key, const std::vector<std::string>& prev, int slot) {
        if (!hist.empty()) for (int i = 0; i < 3; i++) { auto f = freq(WIN[i]); lw[i] += std::log(f.count(key) ? f[key] : 1e-6); }
        if (slot < int(prev.size())) { trials += 1; rep += prev[slot] == key; }
        hist.push_back(key);
    }
};

// ---------------------------------------------------------------- the calibrated test
struct ReverseResult { double evidence[NHYP] = {}, null_mean[NHYP] = {}, null_p95[NHYP] = {}, p[NHYP] = {}, tau[NHYP] = {}; int nulls = 0; };
inline void hypothesis_evidence(const std::vector<Round>& R, double* ev, double* tau_best) {
    double fair_order = 0; for (int k = 0; k < PICK; k++) fair_order -= std::log(double(POOL - k));
    std::vector<HypState> hs; for (int h = 0; h < NHYP; h++) hs.emplace_back(h);
    Mask last = 0; std::string night;
    for (size_t t = 0; t < R.size(); t++) {
        if (R[t].date != night) { night = R[t].date; last = t ? R[t - 1].draw : 0; }
        for (auto& s : hs) s.update(R[t], last);
    }
    for (int h = 0; h < NHYP; h++) {
        ev[h] = hs[h].evidence() - fair_order * double(R.size());
        if (tau_best) { double w[NT]; tau_post(hs[h].lev, w); tau_best[h] = 0; for (int t = 0; t < NT; t++) tau_best[h] += w[t] * RTAUS[t]; }
    }
}
// the real evidence of every hypothesis, ranked within `nulls` fair histories with the real machines and ball sets
inline ReverseResult reverse_test(const std::vector<Round>& R, int nulls, uint64_t seed, Backend& be) {
    ReverseResult out; out.nulls = nulls;
    hypothesis_evidence(R, out.evidence, out.tau);
    std::vector<std::array<double, NHYP>> null(nulls);
    be.parallel(nulls, [&](int a, int b) {
        for (int k = a; k < b; k++) {
            std::vector<Round> S = R; Rng r{stream(seed, uint64_t(k), 9)};
            for (auto& x : S) {  // a fair machine: the six in a uniformly random order
                int balls[POOL]; for (int i = 0; i < POOL; i++) balls[i] = i;
                x.draw = 0;
                for (int j = 0; j < PICK; j++) { int s = j + int(r.below(uint32_t(POOL - j))); std::swap(balls[j], balls[s]); x.order[j] = uint8_t(balls[j]); x.draw |= Mask(1) << balls[j]; }
            }
            hypothesis_evidence(S, null[k].data(), nullptr);
        }
    });
    for (int h = 0; h < NHYP; h++) {
        std::vector<double> v; int ge = 0;
        for (auto& n : null) { v.push_back(n[h]); ge += n[h] >= out.evidence[h]; }
        std::sort(v.begin(), v.end());
        double s = 0; for (double x : v) s += x;
        out.null_mean[h] = s / nulls; out.null_p95[h] = v[size_t(0.95 * (nulls - 1))]; out.p[h] = (ge + 1.0) / (nulls + 1.0);
    }
    return out;
}
