// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Michael King — part of ltkit (https://github.com/BTFProdigy/ltkit)
// Licensed under the Apache License, Version 2.0. See the LICENSE and NOTICE
// files in the project root. Attribution must be retained in derivative works.
// libtorch backend end-to-end test: train an MLP, run IMP, check the engine
// invariants over a real tensor backend.
#include "ltkit/ltkit.hpp"
#include "ltkit/torch_backend.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

using ltkit::Criterion;
using ltkit::ImpConfig;
using ltkit::RewindPolicy;
using ltkit::Scope;

static float lcg(uint64_t& s) {
    s = s * 6364136223846793005ULL + 1ULL;
    return static_cast<float>(s >> 33) / static_cast<float>(1ULL << 31) * 2.0f - 1.0f; // ~[-1,1]
}

// y = argmax(x . W), shared W across splits.
static void make_split(const std::vector<float>& w, int d, int k, int n, uint64_t seed,
                       std::vector<float>& x, std::vector<int64_t>& y) {
    uint64_t s = seed;
    x.clear();
    y.clear();
    for (int r = 0; r < n; ++r) {
        std::vector<float> row(d);
        for (int i = 0; i < d; ++i) row[i] = lcg(s);
        int best_j = 0;
        float best = -1e30f;
        for (int j = 0; j < k; ++j) {
            float logit = 0;
            for (int i = 0; i < d; ++i) logit += row[i] * w[i * k + j];
            if (logit > best) { best = logit; best_j = j; }
        }
        for (int i = 0; i < d; ++i) x.push_back(row[i]);
        y.push_back(best_j);
    }
}

int main() {
    torch::manual_seed(0);
    const int d = 16, k = 3, n = 256;
    uint64_t ws = 42;
    std::vector<float> w(d * k);
    for (auto& v : w) v = lcg(ws);

    std::vector<float> xt, xv;
    std::vector<int64_t> yt, yv;
    make_split(w, d, k, n, 1, xt, yt);
    make_split(w, d, k, n, 7, xv, yv);

    auto dev = torch::kCPU;
    auto Xt = torch::from_blob(xt.data(), {n, d}, torch::kFloat32).clone();
    auto Yt = torch::from_blob(yt.data(), {n}, torch::kLong).clone();
    auto Xv = torch::from_blob(xv.data(), {n, d}, torch::kFloat32).clone();
    auto Yv = torch::from_blob(yv.data(), {n}, torch::kLong).clone();

    auto model = ltkit::TorchMlp::make({d, 32, k}, Xt, Yt, Xv, Yv, 0.05, dev);

    ImpConfig cfg;
    cfg.rounds = 4;
    cfg.prune_rate = 0.3f;
    cfg.epochs_per = 40;
    cfg.scope = Scope::Global;
    cfg.rewind = RewindPolicy::Init;

    auto res = ltkit::run_imp(model, cfg);

    assert(res.history.front().sparsity == 0.0f);
    for (std::size_t i = 1; i < res.history.size(); ++i)
        assert(res.history[i].sparsity >= res.history[i - 1].sparsity);
    assert(res.history.back().sparsity > 0.5f);

    // mask-persistence: pruned weights read back as exactly zero
    for (auto& kv : res.masks) {
        auto scores = model.scores(kv.first, Criterion::Magnitude);
        for (std::size_t i = 0; i < kv.second.size(); ++i)
            if (!kv.second[i]) assert(scores[i] == 0.0f);
    }

    float acc = res.history.back().metric;
    std::printf("torch_smoke OK  final_sparsity=%.3f  ticket_acc=%.3f\n",
                res.history.back().sparsity, acc);
    assert(acc > 0.45f); // above chance (0.33 for k=3)
    return 0;
}