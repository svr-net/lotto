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
// The kernel (kernel.hpp) is shared with the browser build; backends are injected (--backend cpu | threads).
//
//   nested_mc --first-draw 2026-11-07 [--csv data/uk_lotto_full.csv] [--wheel results/exports/frozen_wheel.csv]
//             [--out results/nested_mc.json]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include "kernel.hpp"

constexpr int SET = 15;
// 15 lines on 15 points in which any 4 points share 3 with some line ("3 if 4"); as in scripts/consensus_wheels.py
constexpr int W34[15][6] = {{0, 1, 6, 10, 12, 14}, {0, 2, 4, 9, 10, 11}, {3, 8, 9, 10, 13, 14}, {1, 2, 3, 4, 6, 8},
    {2, 5, 8, 11, 12, 14}, {0, 4, 5, 6, 8, 13}, {5, 6, 7, 8, 9, 12}, {1, 2, 5, 7, 10, 13}, {3, 5, 6, 7, 10, 11},
    {3, 4, 7, 10, 12, 14}, {1, 4, 9, 11, 12, 13}, {0, 1, 7, 8, 11, 14}, {0, 1, 3, 5, 9, 14}, {0, 2, 3, 7, 12, 13},
    {2, 6, 9, 11, 13, 14}};

struct ThreadsBackend : Backend {
    unsigned n = std::max(1u, std::thread::hardware_concurrency());
    const char* name() const override { return "threads"; }
    void run(const Job& j, Result& r) override {
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < n; t++)
            pool.emplace_back(run_paths, std::cref(j), std::ref(r), int(uint64_t(j.paths) * t / n), int(uint64_t(j.paths) * (t + 1) / n));
        for (auto& t : pool) t.join();
    }
};
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
static Wheel wheel_on(std::array<int, SET> nums) {  // the 3-if-4 design on these 15 numbers (1-based)
    std::sort(nums.begin(), nums.end());
    Wheel w;
    for (auto& row : W34) { Mask m = 0; for (int i : row) m |= Mask(1) << (nums[i] - 1); w.lines.push_back(m); }
    for (int n : nums) w.set |= Mask(1) << (n - 1);
    return w;
}

// ---------------------------------------------------------------- statistics
static double mean(const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return v.empty() ? 0 : s / v.size(); }
static double choose(int n, int k) { double r = 1; for (int i = 0; i < k; i++) r = r * (n - i) / (i + 1); return r; }

struct ModelStats { std::vector<double> logk, pred, pit, p4, pprize; int inside = 0; };

int main(int argc, char** argv) {
    std::string csv = "data/uk_lotto_full.csv", wheel_csv = "results/exports/frozen_wheel.csv", out = "results/nested_mc.json";
    std::string start = "2017-01-01", first_draw, backend_name = "threads";
    int horizon = 26, paths = 2000, forward_paths = 20000; double price = 2.0;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--csv") csv = v; else if (a == "--wheel") wheel_csv = v; else if (a == "--out") out = v;
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
    double fair_k[PICK + 1];
    for (int k = 0; k <= PICK; k++) fair_k[k] = choose(SET, k) * choose(POOL - SET, PICK - k) / choose(POOL, PICK);
    const char* models[2] = {"fair", "hindsight as reality"};
    ModelStats st[2]; std::vector<double> diff, real_prize, real_any4;
    Rng tie{12345}; int last_window = -horizon, windows = 0;
    std::printf("backend %s; %zu resets from %s; %d paths x %d rounds per model\n", backend->name(), resets.size(), start.c_str(), paths, horizon);
    for (size_t n = 0; n < resets.size(); n++) {
        int t = resets[n];
        std::array<int, POOL> order; for (int i = 0; i < POOL; i++) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return counts[t][a] > counts[t][b]; });
        std::array<int, SET> pick; for (int i = 0; i < SET; i++) pick[i] = order[i] + 1;
        Wheel w = wheel_on(pick);
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
            double pk[PICK + 1] = {};
            for (int k : r.k1) pk[k] += 1;
            logs[m] = m == 0 ? std::log(fair_k[k_real]) : std::log((pk[k_real] + 0.5) / (paths + 0.5 * (PICK + 1)));
            st[m].logk.push_back(logs[m]);
            if (!window) continue;
            std::vector<double> pr(r.prize.begin(), r.prize.end());
            double below = 0, equal = 0, any4 = 0, anyp = 0;
            for (int p = 0; p < paths; p++) { below += pr[p] < prize_real; equal += pr[p] == prize_real; any4 += r.hits4[p] > 0; anyp += pr[p] > 0; }
            st[m].pred.push_back(mean(pr));
            st[m].inside += prize_real >= quantile(pr, 0.05) && prize_real <= quantile(pr, 0.95);
            st[m].pit.push_back((below + tie.unit() * equal) / paths);
            st[m].p4.push_back(any4 / paths); st[m].pprize.push_back(anyp / paths);
        }
        diff.push_back(logs[1] - logs[0]);
    }
    double dm = mean(diff), dv = 0;
    for (double d : diff) dv += (d - dm) * (d - dm);
    double dz = dm / std::sqrt(dv / (diff.size() - 1) / diff.size());

    FILE* f = std::fopen(out.c_str(), "w");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", out.c_str()); return 2; }
    std::fprintf(f, "{\n \"backend\": \"%s\", \"start\": \"%s\", \"horizon\": %d, \"paths\": %d, \"resets\": %zu, \"windows\": %d,\n", backend->name(), start.c_str(), horizon, paths, resets.size(), windows);
    std::fprintf(f, " \"frozen_numbers\": [");
    for (size_t i = 0; i < frozen_numbers.size(); i++) std::fprintf(f, "%s%d", i ? ", " : "", frozen_numbers[i]);
    std::fprintf(f, "],\n \"actual\": {\"prize_rounds_mean\": %.4f, \"any4_rate\": %.4f},\n \"models\": {\n", mean(real_prize), mean(real_any4));
    for (int m = 0; m < 2; m++) {
        int hist[5] = {}; for (double u : st[m].pit) hist[std::min(4, int(u * 5))]++;
        double chi = 0, e = windows / 5.0, brier = 0;
        for (int b : hist) chi += (b - e) * (b - e) / e;
        for (size_t i = 0; i < st[m].p4.size(); i++) brier += (st[m].p4[i] - real_any4[i]) * (st[m].p4[i] - real_any4[i]);
        double pit_p = std::exp(-chi / 2) * (1 + chi / 2);  // chi-square survival, 4 degrees of freedom
        std::fprintf(f, "  \"%s\": {\"log_score_k1\": %.5f, \"pred_prize_mean\": %.4f, \"coverage90\": %.4f, \"pit_p\": %.4f, \"pred_any4\": %.4f, \"brier_any4\": %.4f}%s\n",
                     models[m], mean(st[m].logk), mean(st[m].pred), double(st[m].inside) / windows, pit_p, mean(st[m].p4), brier / windows, m ? "" : ",");
        std::printf("  %-22s log score %.4f; prize rounds per %d: forecast %.2f vs real %.2f; 90%% range held %.0f%%; calibration p %.2f; any 4+ %.3f vs %.3f\n",
                    models[m], mean(st[m].logk), horizon, mean(st[m].pred), mean(real_prize), 100.0 * st[m].inside / windows, pit_p, mean(st[m].p4), mean(real_any4));
    }
    std::fprintf(f, " },\n \"log_score_diff\": {\"mean\": %.5f, \"z\": %.3f},\n", dm, dz);
    std::printf("  hindsight minus fair, log score per reset: %+.4f (z %+.2f)\n", dm, dz);

    // ---- forward: the frozen wheel from the latest real draw, and its real record ranked from its first draw
    const double cost = frozen.lines.size() * price;
    int t0 = T; if (!first_draw.empty()) for (int t = T - 1; t >= 0 && rounds[t].date >= first_draw; t--) t0 = t;
    int real_rounds = T - t0, real_prize_rounds = 0; double real_net = -cost * real_rounds;
    for (int t = t0; t < T; t++) {
        int best = 0;
        for (Mask l : frozen.lines) { int m = std::popcount(rounds[t].draw & l); real_net += PAY[m]; best = std::max(best, m); }
        real_prize_rounds += best >= 3;
    }
    std::fprintf(f, " \"cost_per_round\": %.2f,\n \"forward\": {\"from\": \"%s\", \"real_since_frozen\": {\"first_draw\": \"%s\", \"rounds\": %d, \"prize_rounds\": %d, \"net\": %.2f},\n  \"models\": {\n",
                 cost, rounds[T - 1].date.c_str(), first_draw.c_str(), real_rounds, real_prize_rounds, real_net);
    const std::vector<int> cks = {26, 52, 104};
    for (int m = 0; m < 2; m++) {
        Result r = backend->simulate({counts[T], m == 1, forward_paths, 104, &frozen, cks, 0xF0F0ull + m});
        std::fprintf(f, "   \"%s\": {\"checkpoints\": {", models[m]);
        for (size_t c = 0; c < cks.size(); c++) {
            std::vector<double> pr(forward_paths), net(forward_paths); double none = 0;
            for (int p = 0; p < forward_paths; p++) { pr[p] = r.prize[p * cks.size() + c]; net[p] = r.money[p * cks.size() + c] - cost * cks[c]; none += pr[p] == 0; }
            std::fprintf(f, "%s\"%d\": {\"prize_rounds\": [%.0f, %.0f, %.0f], \"net\": [%.0f, %.0f, %.0f], \"p_no_prize\": %.4f}", c ? ", " : "", cks[c],
                         quantile(pr, 0.05), quantile(pr, 0.5), quantile(pr, 0.95), quantile(net, 0.05), quantile(net, 0.5), quantile(net, 0.95), none / forward_paths);
            if (cks[c] == 104) std::printf("  forward, %-22s 1 year: prize rounds %.0f (%.0f to %.0f), net %.0f (%.0f to %.0f)\n", models[m],
                                           quantile(pr, 0.5), quantile(pr, 0.05), quantile(pr, 0.95), quantile(net, 0.5), quantile(net, 0.05), quantile(net, 0.95));
        }
        std::fprintf(f, "}");
        if (real_rounds) {
            Result b = backend->simulate({counts[t0], m == 1, forward_paths, real_rounds, &frozen, {real_rounds}, 0xBA5Eull + m});
            double below = 0, equal = 0, none = 0;
            for (int p = 0; p < forward_paths; p++) { double net = b.money[p] - cost * real_rounds; below += net < real_net; equal += net == real_net; none += b.prize[p] == 0; }
            std::fprintf(f, ", \"real_rank\": {\"net_pct\": %.1f, \"p_no_prize_so_far\": %.4f}", 100 * (below + equal / 2) / forward_paths, none / forward_paths);
        }
        std::fprintf(f, "}%s\n", m ? "" : ",");
    }
    std::fprintf(f, "  }\n }\n}\n");
    std::fclose(f);
    std::printf("wrote %s\n", out.c_str());
}
