// Reading the draws: the CSV helpers, and the 59-ball era's rounds with the order the balls came out, the machine
// and the ball set.
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "kernel.hpp"

using Order = std::array<uint8_t, PICK>;  // the six numbers (0-based) in the order the machine drew them
struct Round { std::string date; Mask draw; Order order{}; std::string machine, ball_set; };

inline std::vector<std::vector<std::string>> read_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
    std::vector<std::vector<std::string>> rows; std::string line, cell;
    while (std::getline(in, line)) {
        std::stringstream ss(line); rows.emplace_back();
        while (std::getline(ss, cell, ',')) rows.back().push_back(cell);
    }
    return rows;
}
inline int column(const std::vector<std::string>& head, const std::string& name) {
    auto it = std::find(head.begin(), head.end(), name);
    if (it == head.end()) { std::fprintf(stderr, "no column %s\n", name.c_str()); std::exit(2); }
    return int(it - head.begin());
}
inline Mask numbers_mask(const std::vector<std::string>& row, int first) {
    Mask m = 0;
    for (int i = 0; i < PICK; i++) m |= Mask(1) << (std::stoi(row[first + i]) - 1);
    return m;
}
inline std::vector<Round> load_rounds(const std::string& csv) {  // the 59-ball era, from draw 2066, in draw order
    auto rows = read_csv(csv);
    const auto& h = rows[0];
    int c_date = column(h, "date"), c_n1 = column(h, "n1"), c_draw = column(h, "draw_no"), c_d1 = column(h, "d1");
    int c_mach = column(h, "machine"), c_set = column(h, "ball_set");
    auto trim = [](std::string x) { while (!x.empty() && (x.back() == ' ' || x.back() == '\r')) x.pop_back(); while (!x.empty() && x[0] == ' ') x.erase(0, 1); return x; };
    std::vector<Round> rounds;
    for (size_t i = 1; i < rows.size(); i++) {
        auto& r = rows[i];
        if (r.size() <= size_t(std::max({c_n1 + 5, c_d1 + 5, c_mach, c_set})) || std::stoi(r[c_draw]) < 2066) continue;
        Round x{r[c_date], numbers_mask(r, c_n1), {}, trim(r[c_mach]), trim(r[c_set])};
        for (int k = 0; k < PICK; k++) x.order[k] = uint8_t(std::stoi(r[c_d1 + k]) - 1);
        rounds.push_back(x);
    }
    return rounds;
}
