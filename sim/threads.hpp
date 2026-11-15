// The native multi-core backend: a persistent pool that never locks.
//
// No mutex and no condition variable. The caller publishes a task by bumping an epoch; idle workers spin-wait on
// it (pause, then yield, then short sleeps if nothing comes for a while). Work is handed out in chunks: each worker
// claims the next chunk by compare-exchange on a shared cursor and counts itself done when none are left; the
// caller works as worker 0 and spin-waits until all have counted in. A simulation runs the fused kernel into each
// worker's own histogram, which the worker then adds into the result bin by bin with compare-exchange. Every path
// is seeded by its index, so the result does not depend on which worker ran which chunk: it is identical to
// CpuBackend's.
#pragma once
#include <atomic>
#include <chrono>
#include <thread>

#include "kernel.hpp"

inline void spin_wait(unsigned& spins) {
    if (++spins < 256) {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        asm volatile("yield");
#endif
    } else if (spins < 16384) {
        std::this_thread::yield();
    } else {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}
inline void cas_add(uint32_t& bin, uint32_t v) {
    std::atomic_ref<uint32_t> a(bin);
    uint32_t old = a.load(std::memory_order_relaxed);
    while (!a.compare_exchange_weak(old, old + v, std::memory_order_relaxed, std::memory_order_relaxed)) {}
}

class ThreadsBackend final : public Backend {
public:
    explicit ThreadsBackend(unsigned n = std::max(1u, std::thread::hardware_concurrency())) : n_(n), slots_(n) {
        for (unsigned id = 1; id < n_; id++) pool_.emplace_back([this, id] { loop(id); });
    }
    ~ThreadsBackend() override {
        stop_.store(true, std::memory_order_relaxed);
        epoch_.fetch_add(1, std::memory_order_release);
        for (auto& t : pool_) t.join();
    }
    const char* name() const override { return "threads"; }
    unsigned workers() const { return n_; }

    void run(const Job& j, Result& r) override {
        dispatch(j.paths, std::max(1, j.paths / int(n_ * 8)), [&](unsigned id, int from, int to, bool last) {
            const Layout& L = r.L;
            auto& s = slots_[id];
            if (s.h.size() < L.size()) s.h.resize(L.size(), 0);
            if (from < to) { run_paths(j, L, from, to, s.h.data(), s.top); s.used = true; }
            if (!last || !s.used) return;
            // add the bins this worker touched into the result, leaving its own histogram zero again
            auto merge = [&](uint32_t a, uint32_t b) { for (uint32_t i = a; i < b; i++) if (s.h[i]) { cas_add(r.h[i], s.h[i]); s.h[i] = 0; } };
            merge(L.k1(), L.k1() + PICK + 1);
            for (uint32_t c = 0; c < L.nck; c++) { merge(L.prize(c), L.money(c)); merge(L.money(c), L.money(c) + s.top[c]); s.top[c] = 0; }
            s.used = false;
        });
    }
    void parallel(int n, const std::function<void(int, int)>& body) override {
        dispatch(n, 1, [&](unsigned, int from, int to, bool) { if (from < to) body(from, to); });
    }

private:
    // step(worker, from, to, last): called for each claimed chunk, then once with last = true and an empty range
    using Step = std::function<void(unsigned, int, int, bool)>;
    struct alignas(64) Slot { std::vector<uint32_t> h; uint32_t top[MAX_CHECKPOINTS] = {}; bool used = false; };

    void dispatch(int n, int chunk, const Step& step) {
        step_ = &step; n_items_ = n; chunk_ = chunk;
        cursor_.store(0, std::memory_order_relaxed);
        done_.store(0, std::memory_order_relaxed);
        epoch_.fetch_add(1, std::memory_order_release);  // publishes the task to the spinning workers
        work(0);
        for (unsigned spins = 0; done_.load(std::memory_order_acquire) != n_;) spin_wait(spins);
    }
    void loop(unsigned id) {
        for (uint64_t seen = 0;;) {
            uint64_t e;
            for (unsigned spins = 0; (e = epoch_.load(std::memory_order_acquire)) == seen;) spin_wait(spins);
            seen = e;
            if (stop_.load(std::memory_order_relaxed)) return;
            work(id);
        }
    }
    bool claim(int& from, int& to) {  // the next chunk, by compare-exchange on the shared cursor
        int cur = cursor_.load(std::memory_order_relaxed);
        do {
            if (cur >= n_items_) return false;
            to = std::min(n_items_, cur + chunk_);
        } while (!cursor_.compare_exchange_weak(cur, to, std::memory_order_relaxed, std::memory_order_relaxed));
        from = cur;
        return true;
    }
    void work(unsigned id) {
        for (int from, to; claim(from, to);) (*step_)(id, from, to, false);
        (*step_)(id, 0, 0, true);
        done_.fetch_add(1, std::memory_order_acq_rel);
    }

    const unsigned n_;
    std::vector<Slot> slots_;
    std::vector<std::thread> pool_;
    const Step* step_ = nullptr; int n_items_ = 0, chunk_ = 1;
    alignas(64) std::atomic<uint64_t> epoch_{0};
    alignas(64) std::atomic<int> cursor_{0};
    alignas(64) std::atomic<unsigned> done_{0};
    alignas(64) std::atomic<bool> stop_{false};
};
