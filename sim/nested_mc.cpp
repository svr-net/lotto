// Nested Monte Carlo for the lotto wheels, reset to the real draws at every step.
//
// Outer loop: every real draw night from --start. At each one the simulation is reset to reality: the state
// is the real history up to that night, nothing simulated carries over.
// Inner loop: --paths simulated futures of --horizon rounds from that state, under each model:
//   fair       every number equally likely;
//   hindsight  balls drawn one at a time in proportion to each number's count so far (hindsight as
//              reality), each simulated path adding its own draws to its counts.
// Overlay: the real rounds after each reset are scored against the inner forecasts, for a 15-line "3 if 4"
// wheel on the 15 most-drawn numbers at the reset (chosen from earlier draws only): the log score of the
// real number of drawn balls among the 15 in the next round, and, over windows that don't overlap, the
// rounds with a 3+ line (forecast mean, 90% range, calibration) and the chance of a round with 4+ of 15.
// Forward: the frozen wheel simulated 104 rounds on from the latest real draw, with its real record since
// it was frozen ranked within forecasts reset to that day.
//
// This file is the driver: it reads the data, walks the resets and writes the JSON. Every calculation runs in
// the shared kernels: the fused simulation and its histograms (kernel.hpp, shared with the browser build), the
// model scores (models.hpp), on an injected backend (--backend cpu | threads; threads.hpp is a lock-free pool).
//
//   nested_mc --first-draw 2026-11-07 [--csv data/uk_lotto_full.csv] [--wheel results/exports/frozen_wheel.csv]
//             [--out results/nested_mc.json]
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>

#include "models.hpp"
#include "threads.hpp"

static std::unique_ptr<Backend> make_backend(const std::string& name) {
    if (name == "cpu") return std::make_unique<CpuBackend>();
    if (name == "threads") return std::make_unique<ThreadsBackend>();
    std::fprintf(stderr, "unknown backend %s\n", name.c_str()); std::exit(2);
}

// ---------------------------------------------------------------- data
struct Round { std::string date; Mask draw; };
static std::vector<std::vector<std::string>> read_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
    std::vector<std::vector<std::string>> rows; std::string line, cell;
    while (std::getline(in, line)) {
        std::stringstream ss(line); rows.emplace_back();
        while (std::getline(ss, cell, ',')) rows.back().push_back(cell);
    }
    return rows;
}
static int column(const std::vector<std::string>& head, const std::string& name) {
    auto it = std::find(head.begin(), head.end(), name);
    if (it == head.end()) { std::fprintf(stderr, "no column %s\n", name.c_str()); std::exit(2); }
    return int(it - head.begin());
}
static Mask numbers_mask(const std::vector<std::string>& row, int first) {
    Mask m = 0;
    for (int i = 0; i < PICK; i++) m |= Mask(1) << (std::stoi(row[first + i]) - 1);
    return m;
}
struct ModelStats { std::vector<double> logk, pred, pit, p4, pprize; int inside = 0; };

// ---------------------------------------------------------------- every strategy, reset to the real draws
// Each strategy's number weights at each reset night (scripts/strategy_weights.py, fitted on the real draws
// before that night only) become a model of the machine: q = (1 - λ) v + λ / 59, balls drawn one at a time
// in proportion to q (λ = 1 is the fair model). λ is itself fitted walk-forward, the one with the best log
// score on the earlier nights, so a strategy that carries no information drifts to λ = 1. Two strategies
// choose among the others, again from earlier nights only: "best so far" (the strategy, or fair, with the
// best log score until then) and "Bayesian mixture" (every strategy and λ, and fair, weighted by how well
// each predicted the real draws so far, half the prior on fair). The mixture is the optimum forecaster in the
// log-score sense: overall it can trail the best of its models in hindsight by at most the log of that
// model's prior weight. At every night the real draw is scored exactly (the log of the
// probability the model gave that set of six) and, through the inner simulations, on the model's 15 numbers
// in the 3-if-4 wheel; at nights a horizon apart the real rounds after the reset are laid over the model's
// and the fair model's simulated futures.
static std::string numbers_json(Mask m) { std::string s = "["; for (int i = 0; i < POOL; i++) if (m >> i & 1) s += (s.size() > 1 ? ", " : "") + std::to_string(i + 1); return s + "]"; }
static std::string json_str(const std::string& s) { std::string o = "\""; for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; } return o + "\""; }
struct Model {  // one strategy's forecasts at every scored night
    std::string name; std::vector<Probs> q, rank; std::vector<double> lp, lambda;  // rank: what orders its picks
    Probs next{}, next_rank{}; double next_lambda = 1; std::string note;
};

static void strategy_study(FILE* f, Backend& backend, const std::vector<Round>& rounds, const std::string& path,
                           const std::string& start, int horizon, int paths, const double fair15[]) {
    std::ifstream probe(path);
    if (!probe) { std::printf("no strategy weights at %s (run scripts/strategy_weights.py); strategies skipped\n", path.c_str()); return; }
    auto rows = read_csv(path);
    const int T = int(rounds.size());
    std::map<std::string, int> first_round;
    for (int t = T - 1; t >= 0; t--) first_round[rounds[t].date] = t;
    std::vector<std::string> names, nights; std::map<std::string, int> sidx, nidx;
    std::vector<std::vector<Probs>> v;  // [strategy][night]
    for (size_t i = 1; i < rows.size(); i++) {
        auto& r = rows[i];
        if (r.size() < size_t(2 + POOL) || r[0] < start) continue;
        bool next = r[0] > rounds[T - 1].date;
        if (!next && !first_round.count(r[0])) { std::fprintf(stderr, "no real draw on %s\n", r[0].c_str()); std::exit(2); }
        if (!sidx.count(r[1])) { sidx[r[1]] = int(names.size()); names.push_back(r[1]); v.emplace_back(); }
        if (!nidx.count(r[0])) { nidx[r[0]] = int(nights.size()); nights.push_back(r[0]); }
        auto& w = v[sidx[r[1]]]; w.resize(nights.size());
        Probs p; double s = 0;
        for (int k = 0; k < POOL; k++) s += p[k] = std::max(0.0, std::stod(r[2 + k]));
        for (double& x : p) x = s > 0 ? x / s : 1.0 / POOL;
        w[nidx[r[0]]] = p;
    }
    std::vector<int> night_t;  // the night's first real round, or T for the next draw
    for (auto& d : nights) night_t.push_back(d > rounds[T - 1].date ? T : first_round[d]);
    int N = 0; while (N < int(nights.size()) && night_t[N] < T) N++;  // scored nights; a later one is the next draw
    const bool has_next = N < int(nights.size());
    const int S = int(names.size());
    const double fair_lp = -std::log(choose(POOL, PICK));

    // exact log score of every real draw under every strategy and λ
    std::vector<Mask> real_draws; for (int n = 0; n < N; n++) real_draws.push_back(rounds[night_t[n]].draw);
    const auto lp = log_scores(backend, v, real_draws);  // exact log score of every real draw under every strategy and λ
    // λ fitted walk-forward: the best total on the earlier nights (ties go to the larger λ, nearer fair)
    std::vector<Model> models;
    auto best_g = [&](const std::vector<double>& cum) { int b = G - 1; for (int g = G - 2; g >= 0; g--) if (cum[g] > cum[b]) b = g; return b; };
    for (int s = 0; s < S; s++) {
        Model m; m.name = names[s]; std::vector<double> cum(G, 0.0);
        for (int n = 0; n < N; n++) {
            int b = best_g(cum);
            m.q.push_back(smooth(v[s][n], LAMBDA[b])); m.rank.push_back(v[s][n]); m.lp.push_back(lp[s][b][n]); m.lambda.push_back(LAMBDA[b]);
            for (int g = 0; g < G; g++) cum[g] += lp[s][g][n];
        }
        int b = best_g(cum); m.next_lambda = LAMBDA[b];
        if (has_next) { m.next = smooth(v[s][N], LAMBDA[b]); m.next_rank = v[s][N]; }
        models.push_back(std::move(m));
    }
    Probs uni; uni.fill(1.0 / POOL);
    // best so far: the strategy (or fair) with the best total log score on the earlier nights
    Model ftl; ftl.name = "best so far (walk-forward)"; std::vector<int> chosen(S + 1, 0);
    {
        std::vector<double> cum(S, 0.0);
        auto pick = [&] { int b = -1; double best = 0; for (int s = 0; s < S; s++) if (cum[s] > best + 1e-9) { best = cum[s]; b = s; } return b; };
        for (int n = 0; n < N; n++) {
            int b = pick(); chosen[b + 1]++;
            ftl.q.push_back(b < 0 ? uni : models[b].q[n]); ftl.rank.push_back(b < 0 ? uni : models[b].rank[n]); ftl.lp.push_back(b < 0 ? fair_lp : models[b].lp[n]); ftl.lambda.push_back(b < 0 ? 1 : models[b].lambda[n]);
            for (int s = 0; s < S; s++) cum[s] += models[s].lp[n] - fair_lp;
        }
        int b = pick();
        ftl.next = b < 0 ? uni : models[b].next; ftl.next_rank = b < 0 ? uni : models[b].next_rank; ftl.next_lambda = b < 0 ? 1 : models[b].next_lambda;
        ftl.note = b < 0 ? "fair" : names[b];
    }
    // Bayesian mixture over every strategy at every λ < 1, and fair: half the prior on fair (the hypothesis every
    // test so far supports), the other half shared equally, so the mixture can lose at most log 2 to fair overall
    Model mix; mix.name = "Bayesian mixture (walk-forward)"; double mix_fair_weight = 1; std::vector<double> mix_strategy_weight(S, 0.0);
    {
        const int E = S * (G - 1) + 1;
        std::vector<double> logw(E, std::log(0.5 / (E - 1)));
        logw[E - 1] = std::log(0.5);
        auto expert_lp = [&](int e, int n) { return e == E - 1 ? fair_lp : lp[e / (G - 1)][e % (G - 1)][n]; };
        auto expert_q = [&](int e, int n) { return e == E - 1 ? uni : smooth(v[e / (G - 1)][n], LAMBDA[e % (G - 1)]); };
        auto weights = [&] {
            double mx = *std::max_element(logw.begin(), logw.end()), z = 0; std::vector<double> w(E);
            for (int e = 0; e < E; e++) z += w[e] = std::exp(logw[e] - mx);
            for (double& x : w) x /= z;
            return w;
        };
        for (int n = 0; n <= N && (n < N || has_next); n++) {
            auto w = weights(); Probs q{}; double p = 0;
            for (int e = 0; e < E; e++) { if (w[e] < 1e-15) continue; auto qe = expert_q(e, n); for (int i = 0; i < POOL; i++) q[i] += w[e] * qe[i]; }
            if (n == N) { mix.next = mix.next_rank = q; mix_fair_weight = w[E - 1]; for (int e = 0; e < E - 1; e++) mix_strategy_weight[e / (G - 1)] += w[e]; break; }
            for (int e = 0; e < E; e++) p += w[e] * std::exp(expert_lp(e, n) - fair_lp);
            mix.q.push_back(q); mix.rank.push_back(q); mix.lp.push_back(fair_lp + std::log(p)); mix.lambda.push_back(w[E - 1]);
            for (int e = 0; e < E; e++) logw[e] += expert_lp(e, n);
        }
        if (!has_next) { auto w = weights(); mix_fair_weight = w[E - 1]; for (int e = 0; e < E - 1; e++) mix_strategy_weight[e / (G - 1)] += w[e]; }
        mix.note = "weight on fair " + std::to_string(mix_fair_weight);
    }
    models.push_back(std::move(ftl)); models.push_back(std::move(mix));

    // inner simulations from every reset, real rounds laid over them
    std::vector<int> windows; for (int n = 0, last = -horizon; n < N; n++) if (night_t[n] + horizon <= T && night_t[n] - last >= horizon) { windows.push_back(n); last = night_t[n]; }
    std::vector<char> is_window(N, 0); for (int n : windows) is_window[n] = 1;
    const double m15 = double(PICK) * SET / POOL, v15 = PICK * (double(SET) / POOL) * (1 - double(SET) / POOL) * (POOL - PICK) / (POOL - 1);
    const double m6 = double(PICK) * PICK / POOL, v6 = PICK * (double(PICK) / POOL) * (1 - double(PICK) / POOL) * (POOL - PICK) / (POOL - 1);
    std::printf("strategies: %d models over %d reset nights from %s (%zu windows of %d rounds), %d inner paths per night\n", int(models.size()), N, nights[0].c_str(), windows.size(), horizon, paths);
    std::fprintf(f, " \"strategies\": {\"nights\": %d, \"first_night\": \"%s\", \"last_night\": \"%s\", \"windows\": %zu, \"lambda_grid\": [", N, nights[0].c_str(), nights[N - 1].c_str(), windows.size());
    for (int g = 0; g < G; g++) std::fprintf(f, "%s%g", g ? ", " : "", LAMBDA[g]);
    std::fprintf(f, "], \"fair_log_score\": %.5f,\n  \"chance\": {\"matches15\": %.4f, \"matches6\": %.4f},\n  \"models\": {\n", fair_lp, m15, m6);
    struct Line { std::string name; double diff, z, k1diff, z15, z6; };
    std::vector<Line> table;
    for (size_t s = 0; s < models.size(); s++) {
        Model& M = models[s]; Rng rng{0xC0FFEEull + s};
        std::vector<double> diff, k1diff, k15, k6, pred, fpred, pit, real_prize; int inside = 0, finside = 0;
        for (int n = 0; n < N; n++) {
            int t = night_t[n];
            Mask set15 = top_mask(M.rank[n], SET, rng), set6 = top_mask(M.rank[n], PICK, rng);
            Wheel w = wheel_of(set15);
            int k = std::popcount(rounds[t].draw & set15);
            diff.push_back(M.lp[n] - fair_lp); k15.push_back(k); k6.push_back(std::popcount(rounds[t].draw & set6));
            Job j{{}, true, paths, is_window[n] ? horizon : 1, &w, {is_window[n] ? horizon : 1}, 0x5A1E0000ull + s * 100003 + n};
            for (int i = 0; i < POOL; i++) j.weights[i] = float(M.q[n][i]);
            j.learn = false;
            Result r = backend.simulate(j);
            k1diff.push_back(std::log((r.k1().count(k) + 0.5) / (paths + 0.5 * (PICK + 1))) - std::log(fair15[k]));
            if (!is_window[n]) continue;
            int prize = 0;
            for (int h = 0; h < horizon; h++) { int best = 0; for (Mask l : w.lines) best = std::max(best, std::popcount(rounds[t + h].draw & l)); prize += best >= 3; }
            real_prize.push_back(prize);
            Result fr = backend.simulate({{}, false, paths, horizon, &w, {horizon}, 0xFA1F0000ull + s * 100003 + n});
            Forecast m = Forecast::of(r.prize(0), prize, rng.unit()), fm = Forecast::of(fr.prize(0), prize, 0);
            pred.push_back(m.mean); fpred.push_back(fm.mean); pit.push_back(m.pit); inside += m.inside90; finside += fm.inside90;
        }
        double z15 = (mean(k15) - m15) / std::sqrt(v15 / N), z6 = (mean(k6) - m6) / std::sqrt(v6 / N);
        double used = 0; for (double l : M.lambda) used += l < 1;
        Rng nr{0xBEEFull + s};
        Mask next15 = has_next ? top_mask(M.next_rank, SET, nr) : 0, next6 = has_next ? top_mask(M.next_rank, PICK, nr) : 0;
        std::fprintf(f, "   %s: {\"log_score_diff\": %.5f, \"z\": %.3f, \"bits_per_draw\": %.5f, \"k1_log_score_diff\": %.5f, \"k1_z\": %.3f,"
                        " \"matches15\": %.4f, \"z15\": %.3f, \"matches6\": %.4f, \"z6\": %.3f, \"lambda_final\": %g, \"share_used\": %.4f,"
                        " \"real_prize_mean\": %.4f, \"pred_prize_mean\": %.4f, \"fair_pred_prize_mean\": %.4f, \"coverage90\": %.4f, \"fair_coverage90\": %.4f, \"pit_p\": %.4f",
                     json_str(M.name).c_str(), mean(diff), zstat(diff), mean(diff) / std::log(2.0), mean(k1diff), zstat(k1diff), mean(k15), z15, mean(k6), z6,
                     M.next_lambda, used / N, mean(real_prize), mean(pred), mean(fpred), pit.empty() ? 0 : double(inside) / pit.size(),
                     pit.empty() ? 0 : double(finside) / pit.size(), pit_p(pit));
        if (has_next) std::fprintf(f, ", \"next15\": %s, \"next6\": %s", numbers_json(next15).c_str(), numbers_json(next6).c_str());
        if (!M.note.empty()) std::fprintf(f, ", \"note\": %s", json_str(M.note).c_str());
        std::fprintf(f, "}%s\n", s + 1 < models.size() ? "," : "");
        table.push_back({M.name, mean(diff), zstat(diff), mean(k1diff), z15, z6});
    }
    // how high the best of S strategies' z climbs by luck alone
    Rng lr{77}; std::vector<double> mx;
    for (int i = 0; i < 20000; i++) { double b = -1e9; for (int s = 0; s < S; s++) { double u1 = std::max(1e-12, double(lr.unit())), u2 = lr.unit(); b = std::max(b, std::sqrt(-2 * std::log(u1)) * std::cos(6.283185307179586 * u2)); } mx.push_back(b); }
    std::fprintf(f, "  },\n  \"best_of_luck_z\": {\"median\": %.3f, \"p95\": %.3f},\n  \"best_so_far_chosen\": {\"fair\": %d", quantile(mx, 0.5), quantile(mx, 0.95), chosen[0]);
    for (int s = 0; s < S; s++) if (chosen[s + 1]) std::fprintf(f, ", %s: %d", json_str(names[s]).c_str(), chosen[s + 1]);
    std::fprintf(f, "},\n  \"mixture_weights\": {\"fair\": %.6f", mix_fair_weight);
    for (int s = 0; s < S; s++) std::fprintf(f, ", %s: %.6f", json_str(names[s]).c_str(), mix_strategy_weight[s]);
    std::fprintf(f, "}");
    if (has_next) std::fprintf(f, ",\n  \"next_night\": \"%s\"", nights[N].c_str());
    std::fprintf(f, "\n },\n");
    std::sort(table.begin(), table.end(), [](auto& a, auto& b) { return a.diff > b.diff; });
    std::printf("  %-34s %12s %7s %10s %7s %7s\n", "model (ranked by log score)", "vs fair/draw", "z", "k15 vs fair", "z15", "z6");
    for (auto& l : table) std::printf("  %-34s %+12.5f %+7.2f %+10.5f %+7.2f %+7.2f\n", l.name.c_str(), l.diff, l.z, l.k1diff, l.z15, l.z6);
    std::printf("  best of %d strategies by luck alone: z %.2f (95%%: %.2f)\n", S, quantile(mx, 0.5), quantile(mx, 0.95));
}

int main(int argc, char** argv) {
    std::string csv = "data/uk_lotto_full.csv", wheel_csv = "results/exports/frozen_wheel.csv", out = "results/nested_mc.json";
    std::string strategies_csv = "results/cache/strategy_weights.csv";
    std::string start = "2017-01-01", first_draw, backend_name = "threads";
    int horizon = 26, paths = 2000, forward_paths = 20000; double price = 2.0;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--csv") csv = v; else if (a == "--strategies") strategies_csv = v; else if (a == "--wheel") wheel_csv = v; else if (a == "--out") out = v;
        else if (a == "--start") start = v; else if (a == "--first-draw") first_draw = v; else if (a == "--backend") backend_name = v;
        else if (a == "--horizon") horizon = std::stoi(v); else if (a == "--paths") paths = std::stoi(v);
        else if (a == "--forward-paths") forward_paths = std::stoi(v); else if (a == "--price") price = std::stod(v);
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    auto backend = make_backend(backend_name);

    auto rows = read_csv(csv);
    int c_date = column(rows[0], "date"), c_n1 = column(rows[0], "n1"), c_draw = column(rows[0], "draw_no");
    std::vector<Round> rounds;
    for (size_t i = 1; i < rows.size(); i++)  // the 59-ball era starts at draw 2066
        if (rows[i].size() > size_t(c_n1 + 5) && std::stoi(rows[i][c_draw]) >= 2066) rounds.push_back({rows[i][c_date], numbers_mask(rows[i], c_n1)});
    const int T = int(rounds.size());
    std::vector<std::array<float, POOL>> counts(T + 1);  // counts[t]: draws of each number before round t, plus 1
    counts[0].fill(1.0f);
    for (int t = 0; t < T; t++) { counts[t + 1] = counts[t]; for (int i = 0; i < POOL; i++) if (rounds[t].draw >> i & 1) counts[t + 1][i] += 1; }

    auto wrows = read_csv(wheel_csv);
    Wheel frozen; std::vector<int> frozen_numbers;
    for (size_t i = 1; i < wrows.size(); i++) if (wrows[i].size() >= 7) { Mask m = numbers_mask(wrows[i], 1); frozen.lines.push_back(m); frozen.set |= m; }
    for (int i = 0; i < POOL; i++) if (frozen.set >> i & 1) frozen_numbers.push_back(i + 1);

    // ---- outer loop: reset at each real draw night, inner simulations, real rounds laid over them
    std::vector<int> resets;
    for (int t = 0; t + horizon <= T; t++) if (rounds[t].date >= start && (t == 0 || rounds[t - 1].date != rounds[t].date)) resets.push_back(t);
    double fair_k15[PICK + 1];
    for (int k = 0; k <= PICK; k++) fair_k15[k] = fair_k(SET, k);
    const char* models[2] = {"fair", "hindsight as reality"};
    ModelStats st[2]; std::vector<double> diff, real_prize, real_any4;
    Rng tie{12345}; int last_window = -horizon, windows = 0;
    std::printf("backend %s; %zu resets from %s; %d paths x %d rounds per model\n", backend->name(), resets.size(), start.c_str(), paths, horizon);
    for (size_t n = 0; n < resets.size(); n++) {
        int t = resets[n];
        std::array<int, POOL> order; for (int i = 0; i < POOL; i++) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return counts[t][a] > counts[t][b]; });
        Mask pick = 0; for (int i = 0; i < SET; i++) pick |= Mask(1) << order[i];
        Wheel w = wheel_of(pick);
        int k_real = std::popcount(rounds[t].draw & w.set), prize_real = 0, hits4_real = 0;
        for (int h = 0; h < horizon; h++) {
            int best = 0; for (Mask l : w.lines) best = std::max(best, std::popcount(rounds[t + h].draw & l));
            prize_real += best >= 3; hits4_real += std::popcount(rounds[t + h].draw & w.set) >= 4;
        }
        bool window = t - last_window >= horizon;  // window statistics on resets a full horizon apart
        if (window) { last_window = t; windows++; real_prize.push_back(prize_real); real_any4.push_back(hits4_real > 0); }
        double logs[2];
        for (int m = 0; m < 2; m++) {
            Result r = backend->simulate({counts[t], m == 1, paths, horizon, &w, {horizon}, 0x5EED0000ull + n * 2 + m});
            logs[m] = m == 0 ? std::log(fair_k15[k_real]) : std::log((r.k1().count(k_real) + 0.5) / (paths + 0.5 * (PICK + 1)));
            st[m].logk.push_back(logs[m]);
            if (!window) continue;
            Forecast fc = Forecast::of(r.prize(0), prize_real, tie.unit());
            st[m].pred.push_back(fc.mean); st[m].inside += fc.inside90; st[m].pit.push_back(fc.pit);
            st[m].p4.push_back(1 - r.hits4(0).count(0) / paths); st[m].pprize.push_back(1 - r.prize(0).count(0) / paths);
        }
        diff.push_back(logs[1] - logs[0]);
    }
    double dm = mean(diff), dz = zstat(diff);

    FILE* f = std::fopen(out.c_str(), "w");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", out.c_str()); return 2; }
    std::fprintf(f, "{\n \"backend\": \"%s\", \"start\": \"%s\", \"horizon\": %d, \"paths\": %d, \"resets\": %zu, \"windows\": %d,\n", backend->name(), start.c_str(), horizon, paths, resets.size(), windows);
    std::fprintf(f, " \"frozen_numbers\": [");
    for (size_t i = 0; i < frozen_numbers.size(); i++) std::fprintf(f, "%s%d", i ? ", " : "", frozen_numbers[i]);
    std::fprintf(f, "],\n \"actual\": {\"prize_rounds_mean\": %.4f, \"any4_rate\": %.4f},\n \"models\": {\n", mean(real_prize), mean(real_any4));
    for (int m = 0; m < 2; m++) {
        double brier = 0, calib = pit_p(st[m].pit);
        for (size_t i = 0; i < st[m].p4.size(); i++) brier += (st[m].p4[i] - real_any4[i]) * (st[m].p4[i] - real_any4[i]);
        std::fprintf(f, "  \"%s\": {\"log_score_k1\": %.5f, \"pred_prize_mean\": %.4f, \"coverage90\": %.4f, \"pit_p\": %.4f, \"pred_any4\": %.4f, \"brier_any4\": %.4f}%s\n",
                     models[m], mean(st[m].logk), mean(st[m].pred), double(st[m].inside) / windows, calib, mean(st[m].p4), brier / windows, m ? "" : ",");
        std::printf("  %-22s log score %.4f; prize rounds per %d: forecast %.2f vs real %.2f; 90%% range held %.0f%%; calibration p %.2f; any 4+ %.3f vs %.3f\n",
                    models[m], mean(st[m].logk), horizon, mean(st[m].pred), mean(real_prize), 100.0 * st[m].inside / windows, calib, mean(st[m].p4), mean(real_any4));
    }
    std::fprintf(f, " },\n \"log_score_diff\": {\"mean\": %.5f, \"z\": %.3f},\n", dm, dz);
    std::printf("  hindsight minus fair, log score per reset: %+.4f (z %+.2f)\n", dm, dz);
    if (strategies_csv != "none") strategy_study(f, *backend, rounds, strategies_csv, start, horizon, paths, fair_k15);

    // ---- forward: the frozen wheel from the latest real draw, and its real record ranked from its first draw
    const double cost = frozen.lines.size() * price;
    int t0 = T; if (!first_draw.empty()) for (int t = T - 1; t >= 0 && rounds[t].date >= first_draw; t--) t0 = t;
    int real_rounds = T - t0, real_prize_rounds = 0, real_money = 0;
    for (int t = t0; t < T; t++) {
        int best = 0;
        for (Mask l : frozen.lines) { int m = std::popcount(rounds[t].draw & l); real_money += PAY[m]; best = std::max(best, m); }
        real_prize_rounds += best >= 3;
    }
    const double real_net = real_money - cost * real_rounds;
    std::fprintf(f, " \"cost_per_round\": %.2f,\n \"forward\": {\"from\": \"%s\", \"real_since_frozen\": {\"first_draw\": \"%s\", \"rounds\": %d, \"prize_rounds\": %d, \"net\": %.2f},\n  \"models\": {\n",
                 cost, rounds[T - 1].date.c_str(), first_draw.c_str(), real_rounds, real_prize_rounds, real_net);
    const std::vector<int> cks = {26, 52, 104};
    for (int m = 0; m < 2; m++) {
        Result r = backend->simulate({counts[T], m == 1, forward_paths, 104, &frozen, cks, 0xF0F0ull + m});
        std::fprintf(f, "   \"%s\": {\"checkpoints\": {", models[m]);
        float o[MAX_CHECKPOINTS * 7];
        summarise(r, cks, cost, o);  // the same summary the browser build shows
        for (size_t c = 0; c < cks.size(); c++) {
            const float* q = o + c * 7;
            std::fprintf(f, "%s\"%d\": {\"prize_rounds\": [%.0f, %.0f, %.0f], \"net\": [%.0f, %.0f, %.0f], \"p_no_prize\": %.4f}", c ? ", " : "", cks[c],
                         q[0], q[1], q[2], q[3], q[4], q[5], q[6]);
            if (cks[c] == 104) std::printf("  forward, %-22s 1 year: prize rounds %.0f (%.0f to %.0f), net %.0f (%.0f to %.0f)\n", models[m], q[1], q[0], q[2], q[4], q[3], q[5]);
        }
        std::fprintf(f, "}");
        if (real_rounds) {
            Result b = backend->simulate({counts[t0], m == 1, forward_paths, real_rounds, &frozen, {real_rounds}, 0xBA5Eull + m});
            Hist money = b.money(0);  // net ranks as money does: the cost is the same for every path
            std::fprintf(f, ", \"real_rank\": {\"net_pct\": %.1f, \"p_no_prize_so_far\": %.4f}", 100 * (money.below(real_money) + money.count(real_money) / 2) / forward_paths,
                         b.prize(0).count(0) / forward_paths);
        }
        std::fprintf(f, "}%s\n", m ? "" : ",");
    }
    std::fprintf(f, "  }\n }\n}\n");
    std::fclose(f);
    std::printf("wrote %s\n", out.c_str());
}
