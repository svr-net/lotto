// The one 15-number wheel: radix digits and a neural network over a particle filter, reset to every real draw.
//
// Every real draw night of the 59-ball era, forecasts are made from the real draws before it only; after it, every
// round that night resets them. Nothing is set by hand to lean the result: each part is learned walk-forward.
//   particle filter   the machine as a hidden state (assimilate.hpp), Rao-Blackwellised: particles over the bias
//                     scale and drift memory (wide log-uniform priors), each tracking the 59 log-weights as a
//                     Gaussian updated on every real draw; resampling moves set by the effective sample size;
//   neural network    the filter's forecast nudged per number by a small MLP (nn.hpp) on the filter's estimate and
//                     uncertainty, gaps, recent and first-ball counts and the radix digit lifts in every prime base,
//                     inputs standardised by running statistics; a family over learning rate and penalty, weighted by
//                     how well each predicted the real draws;
//   radix digits      80 digit strategies in the first ten prime bases (radix.hpp), weighted the same way;
//   mixture           fair, filter, network and radix at equal prior weight, then weighted by their real predictions.
// Inner loop: each forecast is simulated (thousands of draws of six) to choose its 15 numbers, those most often
// holding 4+ of the six (hill-climbing by swaps), and, on fresh draws, to state its own chance of 4+. The mixture's
// 15 are the wheel: 15 lines of the "3 if 4" design, at least one line matching 3 whenever 4 of the six are among them.
// Scored from --start: the exact log-probability each forecast gave the real set against fair's (sign-flip test),
// the real rounds with 4+ among its 15 and with a 3+ line, against exact chance.
// Validation: the same pipeline on simulated histories from a machine with six numbers made likelier by each
// --validate delta (0 = fair), to show what it finds when there is something to find.
//
//   assimilate [--csv data/uk_lotto_full.csv] [--out results/assimilation.json] [--exports results/exports]
//              [--start 2017-01-01] [--particles 2048] [--samples 4096] [--validate 0,0.1,0.3,0.6,1] [--backend threads|cpu]
#include <cstdio>
#include <memory>

#include "nn.hpp"
#include "radix.hpp"
#include "threads.hpp"

struct Config {  // compute budget only: none of these leans the forecasts
    std::string start = "2017-01-01";
    int particles = 2048, samples = 4096, batches = 4, batch = 8;
    uint64_t seed = 2026;
};
constexpr double NN_LR[] = {1e-4, 3e-4, 1e-3}, NN_PENALTY[] = {1, 10, 100};  // the family of networks, weighted by the draws
constexpr int NETS = int(std::size(NN_LR) * std::size(NN_PENALTY));
constexpr int PARTS = 4, MODELS = PARTS + 1;  // fair, filter, network, radix; then the mixture of them
static const char* MODEL_NAMES[MODELS] = {"fair", "particle filter", "filter + neural net", "radix digits", "mixture"};
constexpr int BASE_F = 7;

struct Tally { std::vector<double> diff; int rounds = 0, hits4 = 0, line3 = 0, nights = 0; double forecast4 = 0; };
struct Outcome {
    Tally t[MODELS];
    double tau_q[3] = {}, memory_q[3] = {}, log_ev_vs_fair = 0; int resamples = 0;
    double alpha[PARTS] = {}, nets[NETS] = {}, radix[EXPERTS] = {}, nudge_rms = 0;
    int rounds = 0, nights = 0; std::string first_night, last_night;
    Mask next15 = 0; double next_forecast4 = 0; std::string next_night; std::array<double, POOL> next_q{};
};

static std::string next_draw_date(const std::string& d) {  // the next Wednesday or Saturday
    int y = std::stoi(d.substr(0, 4)), m = std::stoi(d.substr(5, 2)), day = std::stoi(d.substr(8, 2));
    auto days = [](int y, int m, int d) { y -= m <= 2; int era = y / 400, yoe = y - era * 400, doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
                                          return era * 146097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719468; };
    long z = days(y, m, day);
    do z++; while (((z + 4) % 7 + 7) % 7 != 3 && ((z + 4) % 7 + 7) % 7 != 6);  // 0 = Sunday
    z += 719468; long era = z / 146097, doe = z - era * 146097, yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    long doy = doe - (365 * yoe + yoe / 4 - yoe / 100), mp = (5 * doy + 2) / 153, dd = doy - (153 * mp + 2) / 5 + 1, mm = mp < 10 ? mp + 3 : mp - 9, yy = yoe + era * 400 + (mm <= 2);
    char buf[64]; std::snprintf(buf, sizeof buf, "%04ld-%02ld-%02ld", yy, mm, dd);
    return buf;
}

// raw features of every number from the draws before round t0 (Mlp::F per number); scaled later by running statistics
static void features(const std::vector<Round>& R, int t0, const std::array<double, POOL>& marg, const double* sd, const double* radix_feat, double* X) {
    auto counts = [&](int W, bool first_ball) {
        std::array<double, POOL> c{};
        for (int t = std::max(0, t0 - W); t < t0; t++) { if (first_ball) c[R[t].order[0]] += 1; else for (int i = 0; i < POOL; i++) c[i] += R[t].draw >> i & 1; }
        return c;
    };
    auto c20 = counts(20, false), c100 = counts(100, false), f200 = counts(200, true);
    for (int i = 0; i < POOL; i++) {
        int gap = t0 + 1; for (int t = t0 - 1; t >= 0; t--) if (R[t].draw >> i & 1) { gap = t0 - t; break; }
        double* x = X + i * Mlp::F;
        x[0] = std::log(POOL * marg[i]); x[1] = sd[i]; x[2] = std::log1p(double(gap));
        x[3] = c20[i]; x[4] = c100[i]; x[5] = t0 > 0 && (R[t0 - 1].draw >> i & 1); x[6] = f200[i];
        for (int b = 0; b < NBASES; b++) x[BASE_F + b] = radix_feat[i * NBASES + b];
    }
}
struct Standardiser {  // running mean and variance of each feature over every number of the earlier nights (Welford)
    double n = 0, mean[Mlp::F] = {}, m2[Mlp::F] = {};
    void apply(double* X) const {
        for (int i = 0; i < POOL; i++) for (int k = 0; k < Mlp::F; k++) {
            double sd = n > 1 ? std::sqrt(m2[k] / (n - 1)) : 0;
            X[i * Mlp::F + k] = sd > 1e-12 ? (X[i * Mlp::F + k] - mean[k]) / sd : 0;
        }
    }
    void update(const double* X) {
        for (int i = 0; i < POOL; i++) { n++; for (int k = 0; k < Mlp::F; k++) { double x = X[i * Mlp::F + k], d = x - mean[k]; mean[k] += d / n; m2[k] += d * (x - mean[k]); } }
    }
};
static void normalise(const double* lw, double* w, int n) {
    double mx = *std::max_element(lw, lw + n), t = 0;
    for (int i = 0; i < n; i++) t += w[i] = std::exp(lw[i] - mx);
    for (int i = 0; i < n; i++) w[i] /= t;
}

static Outcome run(const std::vector<Round>& R, const Config& C, Backend& be, bool with_next) {
    std::vector<std::pair<int, int>> nights;
    for (int t = 0; t < int(R.size()); t++) if (t == 0 || R[t].date != R[t - 1].date) nights.push_back({t, t + 1}); else nights.back().second = t + 1;
    if (with_next) nights.push_back({int(R.size()), int(R.size())});
    Filter filter(C.particles, stream(C.seed, 100), be);
    RadixHistory H(R);
    std::vector<Mlp> nets;
    for (int k = 0; k < NETS; k++) { nets.emplace_back(stream(C.seed, 77, k)); nets[k].lr = NN_LR[k / std::size(NN_PENALTY)]; nets[k].penalty = NN_PENALTY[k % std::size(NN_PENALTY)]; }
    double lw_net[NETS] = {}, lw_rad[EXPERTS] = {}, lw[PARTS] = {};  // equal priors throughout
    Standardiser stdz;
    std::vector<double> RX, RB; std::vector<Order> RO;  // replay: standardised features, filter log-forecast and order of every round
    const double fair_lp = -std::log(choose(POOL, PICK));
    double fair_order = 0; for (int k = 0; k < PICK; k++) fair_order -= std::log(double(POOL - k));
    Mixture fair = Mixture::fair();
    Outcome out; double rms_sum = 0; int rms_n = 0;

    for (size_t n = 0; n < nights.size(); n++) {
        const auto [t0, t1] = nights[n];
        const bool next = t0 == int(R.size()), scored = !next && R[t0].date >= C.start;
        // ---- forecasts from the real draws before this night
        Mixture filt = filter_mixture(filter, be);
        auto marg = filt.marginal();
        double mean[POOL] = {}, sd[POOL] = {};  // the filter's spread over each log-weight: across particles and within each
        for (int p = 0; p < filter.P; p++) { double w = filter.weight(p); const double* th = filter.theta(p); const double* vr = filter.var(p); for (int i = 0; i < POOL; i++) { mean[i] += w * th[i]; sd[i] += w * (th[i] * th[i] + vr[i]); } }
        for (int i = 0; i < POOL; i++) sd[i] = std::sqrt(std::max(0.0, sd[i] - mean[i] * mean[i]));
        std::vector<double> rq; double rfeat[POOL * NBASES];
        radix_experts(H, R, t0, rq, rfeat);
        Mixture rad; double wrad[EXPERTS]; normalise(lw_rad, wrad, EXPERTS);
        rad.w.assign(wrad, wrad + EXPERTS); rad.q = rq;
        std::vector<double> raw(POOL * Mlp::F), X; double base[POOL];
        features(R, t0, marg, sd, rfeat, raw.data());
        X = raw; stdz.apply(X.data()); stdz.update(raw.data());
        for (int i = 0; i < POOL; i++) base[i] = std::log(marg[i]);
        double wnet[NETS]; normalise(lw_net, wnet, NETS);
        Mixture nnm; std::vector<std::array<double, POOL>> qnet(NETS); double rms = 0;
        for (int k = 0; k < NETS; k++) {
            double s[POOL], zl[POOL]; nets[k].nudges(X.data(), s);
            for (int i = 0; i < POOL; i++) { zl[i] = base[i] + s[i]; rms += wnet[k] * s[i] * s[i] / POOL; }
            softmax(zl, qnet[k].data());
            nnm.w.push_back(wnet[k]); nnm.q.insert(nnm.q.end(), qnet[k].begin(), qnet[k].end());
        }
        double alpha[PARTS]; normalise(lw, alpha, PARTS);
        Mixture mix; mix.add(fair, alpha[0]); mix.add(filt, alpha[1]); mix.add(nnm, alpha[2]); mix.add(rad, alpha[3]);
        const Mixture* fc[MODELS] = {&fair, &filt, &nnm, &rad, &mix};
        const uint64_t ns = stream(C.seed, 1000 + n);

        if (next) {  // the next draw: the mixture's 15 are the wheel
            auto draws = sample_sets(mix, C.samples * 2, stream(ns, 1), be);
            out.next15 = best_fifteen(draws, top_set(mix.marginal(), SET));
            out.next_forecast4 = double(count4(sample_sets(mix, C.samples * 2, stream(ns, 2), be), out.next15)) / (C.samples * 2);
            out.next_night = next_draw_date(R.back().date);
            out.next_q = mix.marginal();
            break;
        }
        // ---- score the night's real rounds
        std::vector<std::array<double, PARTS>> lps; std::vector<std::array<double, NETS>> lpn; std::vector<std::array<double, EXPERTS>> lpr;
        for (int t = t0; t < t1; t++) {
            std::array<double, NETS> ln; double pn = 0;
            for (int k = 0; k < NETS; k++) { Probs q; std::copy(qnet[k].begin(), qnet[k].end(), q.begin()); ln[k] = set_logprob(q, R[t].draw); pn += wnet[k] * std::exp(ln[k] - fair_lp); }
            std::array<double, EXPERTS> lr; double pr = 0;
            be.parallel(EXPERTS, [&](int a, int b) { for (int e = a; e < b; e++) { Probs q; std::copy(&rq[size_t(e) * POOL], &rq[size_t(e) * POOL] + POOL, q.begin()); lr[e] = set_logprob(q, R[t].draw); } });
            for (int e = 0; e < EXPERTS; e++) pr += wrad[e] * std::exp(lr[e] - fair_lp);
            std::array<double, PARTS> lp = {fair_lp, mixture_set_logprob(filt, R[t].draw, be), fair_lp + std::log(pn), fair_lp + std::log(pr)};
            double mixp = 0; for (int m = 0; m < PARTS; m++) mixp += alpha[m] * std::exp(lp[m] - fair_lp);
            if (scored) for (int m = 0; m < MODELS; m++) out.t[m].diff.push_back((m < PARTS ? lp[m] : fair_lp + std::log(mixp)) - fair_lp);
            lps.push_back(lp); lpn.push_back(ln); lpr.push_back(lr);
        }
        if (scored) {
            if (out.first_night.empty()) out.first_night = R[t0].date;
            out.last_night = R[t0].date; out.nights++; out.rounds += t1 - t0;
            for (int m = 0; m < MODELS; m++) {
                Mask set;
                if (m == 0) { Rng r{stream(ns, 9)}; set = 0; while (std::popcount(set) < SET) set |= Mask(1) << r.below(POOL); }  // chance: any 15
                else set = best_fifteen(sample_sets(*fc[m], C.samples, stream(ns, 10 + m), be), top_set(fc[m]->marginal(), SET));
                auto fresh = sample_sets(*fc[m], C.samples, stream(ns, 20 + m), be);  // the model's own chance of 4+, on draws the choice never saw
                Tally& T = out.t[m]; T.nights++; T.forecast4 += double(count4(fresh, set)) / C.samples;
                Wheel w = wheel_of(set);
                for (int t = t0; t < t1; t++) {
                    T.rounds++; T.hits4 += std::popcount(R[t].draw & set) >= 4;
                    int best = 0; for (Mask l : w.lines) best = std::max(best, std::popcount(R[t].draw & l));
                    T.line3 += best >= 3;
                }
            }
            rms_sum += rms; rms_n++;
        }
        // ---- learn from the night's real rounds: weights, networks, filter
        for (auto& lp : lps) for (int m = 0; m < PARTS; m++) lw[m] += lp[m];
        for (auto& ln : lpn) for (int k = 0; k < NETS; k++) lw_net[k] += ln[k];
        for (auto& lr : lpr) for (int e = 0; e < EXPERTS; e++) lw_rad[e] += lr[e];
        for (int t = t0; t < t1; t++) { RX.insert(RX.end(), X.begin(), X.end()); RB.insert(RB.end(), base, base + POOL); RO.push_back(R[t].order); }
        const int stored = int(RO.size()), fresh = t1 - t0;
        be.parallel(NETS, [&](int a, int b) {  // each network trains on the night's rounds and a replay of all earlier ones
            std::vector<double> g(Mlp::NP, 0.0);
            for (int k = a; k < b; k++) {
                Rng rr{stream(ns, 30, k)};
                for (int bt = 0; bt < C.batches; bt++) {
                    for (int j = 0; j < C.batch; j++) {
                        int idx = (bt == 0 && j < fresh) ? stored - fresh + j : int(rr.below(uint32_t(stored)));
                        nets[k].accumulate(&RX[size_t(idx) * POOL * Mlp::F], &RB[size_t(idx) * POOL], RO[idx], g);
                    }
                    nets[k].step(g, C.batch);
                }
            }
        });
        for (int t = t0; t < t1; t++) assimilate(filter, R[t].order, stream(C.seed, uint64_t(t), 200), be);
        out.log_ev_vs_fair = filter.log_evidence - fair_order * t1;
    }
    std::vector<std::pair<double, double>> tw, mw;
    for (int p = 0; p < filter.P; p++) { tw.push_back({filter.tau(p), filter.weight(p)}); mw.push_back({filter.memory(p), filter.weight(p)}); }
    const double qs[3] = {0.05, 0.5, 0.95};
    for (int i = 0; i < 3; i++) { out.tau_q[i] = weighted_quantile(tw, qs[i]); out.memory_q[i] = weighted_quantile(mw, qs[i]); }
    out.resamples = filter.resamples;
    normalise(lw, out.alpha, PARTS); normalise(lw_net, out.nets, NETS); normalise(lw_rad, out.radix, EXPERTS);
    out.nudge_rms = rms_n ? std::sqrt(rms_sum / rms_n) : 0;
    return out;
}

// ---------------------------------------------------------------- reporting
static std::string mask_json(Mask m) { std::string s = "["; for (int i = 0; i < POOL; i++) if (m >> i & 1) s += (s.size() > 1 ? ", " : "") + std::to_string(i + 1); return s + "]"; }
static void write_outcome(FILE* f, const Outcome& o, double p4, double p3, const char* ind) {
    std::fprintf(f, "%s\"nights\": %d, \"rounds\": %d, \"first_night\": \"%s\", \"last_night\": \"%s\",\n", ind, o.nights, o.rounds, o.first_night.c_str(), o.last_night.c_str());
    std::fprintf(f, "%s\"filter\": {\"tau\": [%.5f, %.5f, %.5f], \"memory_rounds\": [%.0f, %.0f, %.0f], \"log_evidence_vs_fair\": %.4f, \"resamples\": %d},\n",
                 ind, o.tau_q[0], o.tau_q[1], o.tau_q[2], o.memory_q[0], o.memory_q[1], o.memory_q[2], o.log_ev_vs_fair, o.resamples);
    std::fprintf(f, "%s\"networks\": [", ind);
    for (int k = 0; k < NETS; k++) std::fprintf(f, "%s{\"lr\": %g, \"penalty\": %g, \"weight\": %.6f}", k ? ", " : "", NN_LR[k / std::size(NN_PENALTY)], NN_PENALTY[k % std::size(NN_PENALTY)], o.nets[k]);
    std::fprintf(f, "], \"nudge_rms\": %.5f,\n%s\"radix\": {\"bases\": {", o.nudge_rms, ind);
    for (int b = 0; b < NBASES; b++) { double s = 0; for (int r = 0; r < RULES; r++) s += o.radix[b * RULES + r]; std::fprintf(f, "%s\"%d\": %.6f", b ? ", " : "", PRIMES[b], s); }
    std::fprintf(f, "}, \"rules\": {");
    for (int r = 0; r < RULES; r++) { double s = 0; for (int b = 0; b < NBASES; b++) s += o.radix[b * RULES + r]; std::fprintf(f, "%s\"%s\": %.6f", r ? ", " : "", RULE_NAMES[r], s); }
    std::fprintf(f, "}},\n%s\"mixture_weights\": {", ind);
    for (int m = 0; m < PARTS; m++) std::fprintf(f, "%s\"%s\": %.6f", m ? ", " : "", MODEL_NAMES[m], o.alpha[m]);
    std::fprintf(f, "},\n%s\"models\": {\n", ind);
    for (int m = 0; m < MODELS; m++) {
        const Tally& T = o.t[m]; double sum = 0; for (double d : T.diff) sum += d;
        std::fprintf(f, "%s \"%s\": {\"log_lik_ratio\": %.4f, \"per_round\": %.6f, \"p_better\": %.4f, \"rounds\": %d, \"hits4\": %d, \"rate4\": %.5f, \"p4\": %.4f, \"forecast4\": %.5f,"
                        " \"line3\": %d, \"rate3\": %.5f, \"p3\": %.4f}%s\n",
                     ind, MODEL_NAMES[m], sum, T.diff.empty() ? 0 : sum / T.diff.size(), m ? sign_flip_p(T.diff, 20000, 4242 + m) : 1.0, T.rounds, T.hits4, double(T.hits4) / T.rounds,
                     binom_sf(T.rounds, T.hits4, p4), T.forecast4 / std::max(1, T.nights), T.line3, double(T.line3) / T.rounds, binom_sf(T.rounds, T.line3, p3), m + 1 < MODELS ? "," : "");
    }
    std::fprintf(f, "%s}", ind);
}
static void print_outcome(const char* label, const Outcome& o, double p4, double p3) {
    std::printf("%s: %d rounds from %s; filter tau %.4f (%.4f-%.4f), memory %.0f rounds, evidence vs fair %+.2f; weights fair %.3f filter %.3f net %.3f radix %.3f\n", label, o.rounds,
                o.first_night.c_str(), o.tau_q[1], o.tau_q[0], o.tau_q[2], o.memory_q[1], o.log_ev_vs_fair, o.alpha[0], o.alpha[1], o.alpha[2], o.alpha[3]);
    std::printf("  %-20s %10s %8s %8s %9s %8s %9s %8s\n", "model", "logLR", "p", "4+ rate", "(chance)", "p", "3+ line", "p");
    for (int m = 0; m < MODELS; m++) {
        const Tally& T = o.t[m]; double sum = 0; for (double d : T.diff) sum += d;
        std::printf("  %-20s %+10.3f %8.4f %7.2f%% %8.2f%% %8.4f %8.2f%% %8.4f\n", MODEL_NAMES[m], sum, m ? sign_flip_p(T.diff, 20000, 4242 + m) : 1.0,
                    100.0 * T.hits4 / T.rounds, 100 * p4, binom_sf(T.rounds, T.hits4, p4), 100.0 * T.line3 / T.rounds, binom_sf(T.rounds, T.line3, p3));
    }
}
static void write_exports(const std::string& dir, const Outcome& o) {  // the wheel's lines for entering in the app, and as CSV
    Wheel w = wheel_of(o.next15);
    std::string nums; for (int i = 0; i < POOL; i++) if (o.next15 >> i & 1) nums += (nums.empty() ? "" : " ") + std::to_string(i + 1);
    FILE* a = std::fopen((dir + "/wheel_app.txt").c_str(), "w"); FILE* c = std::fopen((dir + "/wheel.csv").c_str(), "w");
    if (!a || !c) { std::fprintf(stderr, "cannot write the exports in %s\n", dir.c_str()); std::exit(2); }
    std::fprintf(a, "UK Lotto: \"3 if 4\" wheel for %s on %s\n%zu lines, £%zu a draw (£2 a line).\nGuarantee: whenever 4 of the six drawn numbers are among these 15, at least one line matches 3.\n\n",
                 o.next_night.c_str(), nums.c_str(), w.lines.size(), w.lines.size() * 2);
    std::fprintf(c, "line,n1,n2,n3,n4,n5,n6\n");
    for (size_t l = 0; l < w.lines.size(); l++) {
        std::fprintf(a, "Line %2zu:", l + 1); std::fprintf(c, "%zu", l + 1);
        for (int i = 0; i < POOL; i++) if (w.lines[l] >> i & 1) { std::fprintf(a, " %3d", i + 1); std::fprintf(c, ",%d", i + 1); }
        std::fprintf(a, "\n"); std::fprintf(c, "\n");
    }
    std::fclose(a); std::fclose(c);
}

int main(int argc, char** argv) {
    std::string csv = "data/uk_lotto_full.csv", out = "results/assimilation.json", exports = "results/exports", backend_name = "threads", validate = "0,0.1,0.3,0.6,1";
    Config C; bool real_on = true;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--csv") csv = v; else if (a == "--out") out = v; else if (a == "--exports") exports = v; else if (a == "--start") C.start = v; else if (a == "--backend") backend_name = v;
        else if (a == "--particles") C.particles = std::stoi(v); else if (a == "--samples") C.samples = std::stoi(v);
        else if (a == "--validate") validate = v; else if (a == "--seed") C.seed = std::stoull(v); else if (a == "--real") real_on = v != "0";
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    std::unique_ptr<Backend> be;
    if (backend_name == "cpu") be = std::make_unique<CpuBackend>(); else if (backend_name == "threads") be = std::make_unique<ThreadsBackend>();
    else { std::fprintf(stderr, "unknown backend %s\n", backend_name.c_str()); return 2; }
    const std::vector<Round> R = load_rounds(csv);
    const double p4 = fair_p4(), p3 = fair_p_line3();
    std::printf("backend %s; %zu rounds of the 59-ball era; %d particles; %d networks; %d radix experts; %d simulated draws per choice; chance: 4+ of 15 %.4f, 3+ line %.4f\n",
                be->name(), R.size(), C.particles, NETS, EXPERTS, C.samples, p4, p3);
    std::fflush(stdout);

    Outcome real;
    if (real_on) {
        real = run(R, C, *be, true);
        print_outcome("real draws", real, p4, p3);
        std::printf("  wheel for %s: %s (the mixture's own chance of 4+: %.2f%%)\n", real.next_night.c_str(), mask_json(real.next15).c_str(), 100 * real.next_forecast4);
        std::fflush(stdout);
        if (exports != "none") write_exports(exports, real);
    }
    std::vector<double> deltas;
    for (size_t a = 0; a < validate.size();) { size_t b = validate.find(',', a); if (b == std::string::npos) b = validate.size(); if (b > a) deltas.push_back(std::stod(validate.substr(a, b - a))); a = b + 1; }
    struct Val { double delta; Mask planted; double oracle4; Outcome o; };
    std::vector<Val> vals;
    for (size_t v = 0; v < deltas.size(); v++) {
        const double d = deltas[v];
        Rng pr{stream(C.seed, 5000 + v)}; Mask planted = 0;
        while (std::popcount(planted) < PICK) planted |= Mask(1) << pr.below(POOL);
        double q[POOL]; for (int i = 0; i < POOL; i++) q[i] = (planted >> i & 1) ? 1 + d : 1.0;
        std::vector<Round> S = R;
        for (size_t t = 0; t < S.size(); t++) {  // each round drawn from the planted machine, ball by ball
            Rng r{stream(C.seed, 7000 + v, t)}; Mask got = 0; double tot = 0; for (double x : q) tot += x;
            for (int k = 0; k < PICK; k++) {
                double u = u53(r) * tot; int i = 0;
                for (; i < POOL - 1; i++) { if (got >> i & 1) continue; u -= q[i]; if (u < 0) break; }
                while (got >> i & 1) i--;
                S[t].order[k] = uint8_t(i); got |= Mask(1) << i; tot -= q[i];
            }
            S[t].draw = got;
        }
        Mask best = planted; for (int i = 0; std::popcount(best) < SET; i++) best |= Mask(1) << i;
        double oracle = double(count4(sample_sets(Mixture::single(q), 400000, stream(C.seed, 8000 + v), *be), best)) / 400000;
        Outcome o = run(S, C, *be, false);
        char label[64]; std::snprintf(label, sizeof label, "simulated machine, six numbers +%.0f%%", 100 * d);
        print_outcome(label, o, p4, p3);
        std::printf("  planted %s; knowing them, the best 15 hold 4+ in %.2f%% of rounds\n", mask_json(planted).c_str(), 100 * oracle);
        std::fflush(stdout);
        vals.push_back({d, planted, oracle, std::move(o)});
    }

    FILE* f = std::fopen(out.c_str(), "w");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", out.c_str()); return 2; }
    std::fprintf(f, "{\n \"backend\": \"%s\", \"particles\": %d, \"networks\": %d, \"radix_experts\": %d, \"samples\": %d, \"start\": \"%s\", \"rounds_total\": %zu, \"last_draw\": \"%s\",\n"
                    " \"chance\": {\"rate4\": %.6f, \"rate3\": %.6f},\n \"real\": {\n",
                 be->name(), C.particles, NETS, EXPERTS, C.samples, C.start.c_str(), R.size(), R.back().date.c_str(), p4, p3);
    if (real_on) {
        write_outcome(f, real, p4, p3, "  ");
        std::fprintf(f, ",\n  \"next\": {\"night\": \"%s\", \"numbers\": %s, \"forecast4\": %.5f, \"lines\": [", real.next_night.c_str(), mask_json(real.next15).c_str(), real.next_forecast4);
        Wheel nw = wheel_of(real.next15);
        for (size_t l = 0; l < nw.lines.size(); l++) std::fprintf(f, "%s%s", l ? ", " : "", mask_json(nw.lines[l]).c_str());
        std::fprintf(f, "], \"weights\": [");
        for (int i = 0; i < POOL; i++) std::fprintf(f, "%s%.7f", i ? ", " : "", real.next_q[i]);
        std::fprintf(f, "]}\n");
    }
    std::fprintf(f, " },\n \"validation\": [\n");
    for (size_t v = 0; v < vals.size(); v++) {
        std::fprintf(f, "  {\"delta\": %g, \"planted\": %s, \"oracle_rate4\": %.5f,\n", vals[v].delta, mask_json(vals[v].planted).c_str(), vals[v].oracle4);
        write_outcome(f, vals[v].o, p4, p3, "   ");
        std::fprintf(f, "}%s\n", v + 1 < vals.size() ? "," : "");
    }
    std::fprintf(f, " ]\n}\n");
    std::fclose(f);
    std::printf("wrote %s\n", out.c_str());
}
