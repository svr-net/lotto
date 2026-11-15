// The radix strategies: every ball written in the first ten prime bases (2, 3, 5, ... 29), and each number scored
// by how its digits, position by position, have been drawn. 23 in base 5 is 043, so it borrows from every number
// whose middle digit is 4. Digit classes are not the same size (in base 7 a leading 1 covers 7-13, a leading 8
// only 56-59), so every class is measured against what a fair draw gives it, and a number's weight is the product
// of its digits' lifts. The rules, in every base:
//   hot      each digit class's rate over the last 20, 100 or all rounds (one fair round of prior), as a lift
//   cold     the inverse of the same lifts
//   overdue  rounds since the class last came out, against its fair waiting time
//   early    the class's rate among the first two balls out over the last 200 rounds
// That is 8 rules x 10 bases = 80 experts. The driver weights them by how well each predicted the real draws
// (Hedge with full information on the exact log score), so no base or rule is picked by hand. The hot lifts over
// 100 rounds, summed over each number's digits, also feed the neural network, one input per base.
#pragma once
#include <cmath>

#include "assimilate.hpp"

constexpr int NBASES = 10, RULES = 8, EXPERTS = NBASES * RULES;
constexpr int PRIMES[NBASES] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
inline const char* RULE_NAMES[RULES] = {"hot, 20 rounds", "hot, 100 rounds", "hot, all rounds", "cold, 20 rounds", "cold, 100 rounds",
                                        "cold, all rounds", "overdue", "early balls, 200 rounds"};

struct Digits {  // one base: the digit of each number at each position, and each class's fair share
    int base = 0, width = 0;
    std::vector<std::array<int, POOL>> digit;    // [position][number]
    std::vector<std::vector<double>> share, p;   // [position][digit]: share of the 59 numbers; chance per round it comes out
    explicit Digits(int b) : base(b) {
        long top = 1; while (top <= POOL) { top *= b; width++; }
        long pw = 1;
        for (int j = 0; j < width; j++, pw *= b) {
            std::array<int, POOL> d; std::vector<double> sz(b, 0.0);
            for (int n = 0; n < POOL; n++) { d[n] = int(((n + 1) / pw) % b); sz[d[n]] += 1; }
            digit.push_back(d);
            std::vector<double> s(b), q(b);
            for (int c = 0; c < b; c++) { s[c] = sz[c] / POOL; q[c] = 1 - choose(POOL - int(sz[c]), PICK) / choose(POOL, PICK); }
            share.push_back(s); p.push_back(q);
        }
    }
};
struct RadixHistory {  // running counts so any window's counts are a difference of prefixes
    std::vector<Digits> bases;
    std::vector<std::array<int, POOL>> cum, early;  // [round][number]: drawn so far; among the first two balls so far
    explicit RadixHistory(const std::vector<Round>& R) {
        for (int b : PRIMES) bases.emplace_back(b);
        cum.assign(R.size() + 1, {}); early.assign(R.size() + 1, {});
        for (size_t t = 0; t < R.size(); t++) {
            cum[t + 1] = cum[t]; early[t + 1] = early[t];
            for (int n = 0; n < POOL; n++) cum[t + 1][n] += R[t].draw >> n & 1;
            early[t + 1][R[t].order[0]]++; early[t + 1][R[t].order[1]]++;
        }
    }
};

// Every expert's ball probabilities (q: EXPERTS x POOL) and the network's radix inputs (feat: POOL x NBASES) from the
// rounds before t0.
inline void radix_experts(const RadixHistory& H, const std::vector<Round>& R, int t0, std::vector<double>& q, double* feat) {
    q.assign(size_t(EXPERTS) * POOL, 0.0);
    int last[POOL]; for (int n = 0; n < POOL; n++) { last[n] = -1; for (int t = t0 - 1; t >= 0; t--) if (R[t].draw >> n & 1) { last[n] = t; break; } }
    for (int bi = 0; bi < NBASES; bi++) {
        const Digits& D = H.bases[bi];
        double logw[RULES][POOL] = {}, hot100[POOL] = {};
        for (int j = 0; j < D.width; j++) {
            const int b = D.base;
            auto lifts = [&](int W, bool first_two) {  // smoothed rate of each class over the window, against fair
                const int n0 = std::max(0, t0 - W), rounds = t0 - n0, per = first_two ? 2 : PICK;
                std::vector<double> cnt(b, 0.0), out(b);
                const auto& hi = first_two ? H.early[t0] : H.cum[t0]; const auto& lo = first_two ? H.early[n0] : H.cum[n0];
                for (int n = 0; n < POOL; n++) cnt[D.digit[j][n]] += hi[n] - lo[n];
                for (int c = 0; c < b; c++) out[c] = D.share[j][c] > 0 ? (cnt[c] + per * D.share[j][c]) / ((double(per) * rounds + per) * D.share[j][c]) : 1;
                return out;
            };
            auto h20 = lifts(20, false), h100 = lifts(100, false), hall = lifts(t0, false), e200 = lifts(200, true);
            std::vector<int> lastc(b, -1);
            for (int n = 0; n < POOL; n++) lastc[D.digit[j][n]] = std::max(lastc[D.digit[j][n]], last[n]);
            for (int n = 0; n < POOL; n++) {
                const int c = D.digit[j][n];
                const double od = D.p[j][c] > 0 ? (t0 - lastc[c]) * D.p[j][c] : 1;  // waiting time against its fair mean
                const double l[RULES] = {h20[c], h100[c], hall[c], 1 / h20[c], 1 / h100[c], 1 / hall[c], od, e200[c]};
                for (int r = 0; r < RULES; r++) logw[r][n] += std::log(std::max(1e-9, l[r]));
                hot100[n] += std::log(h100[c]);
            }
        }
        for (int r = 0; r < RULES; r++) softmax(logw[r], &q[size_t(bi * RULES + r) * POOL]);
        for (int n = 0; n < POOL; n++) feat[n * NBASES + bi] = hot100[n];
    }
}
