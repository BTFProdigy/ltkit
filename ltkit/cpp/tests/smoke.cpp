// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Michael King — part of ltkit (https://github.com/BTFProdigy/ltkit)
// Licensed under the Apache License, Version 2.0. See the LICENSE and NOTICE
// files in the project root. Attribution must be retained in derivative works.
// Engine smoke test over a pure in-memory mock model — no tensor backend.
// Mirrors the Rust/Python invariant: sparsity rises monotonically and the
// final masks agree with the model's zeroed weights (mask-persistence).
#include "ltkit/ltkit.hpp"

#include <cassert>
#include <cstdio>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

using ltkit::Criterion;
using ltkit::ImpConfig;
using ltkit::RewindPolicy;
using ltkit::Scope;

struct MockModel {
    using State = std::unordered_map<std::string, std::vector<float>>;

    std::unordered_map<std::string, std::vector<float>> weights;
    std::unordered_map<std::string, std::vector<bool>> masks;

    MockModel() {
        std::vector<float> a, b;
        for (int i = 1; i <= 20; ++i) a.push_back(static_cast<float>(i));
        for (int i = 1; i <= 12; ++i) b.push_back(static_cast<float>(i) * 0.5f);
        weights["a.weight"] = a;
        weights["b.weight"] = b;
        for (auto& kv : weights) masks[kv.first] = std::vector<bool>(kv.second.size(), true);
    }

    void enforce() {
        for (auto& kv : weights) {
            auto& m = masks[kv.first];
            for (std::size_t i = 0; i < kv.second.size(); ++i)
                if (!m[i]) kv.second[i] = 0.0f;
        }
    }

    std::vector<std::string> parameter_groups() const {
        std::vector<std::string> g;
        for (auto& kv : weights) g.push_back(kv.first);
        std::sort(g.begin(), g.end());
        return g;
    }
    std::vector<float> scores(const std::string& name, Criterion) const {
        std::vector<float> s;
        for (float w : weights.at(name)) s.push_back(w < 0 ? -w : w);
        return s;
    }
    void apply_mask(const std::string& name, const std::vector<bool>& mask) {
        masks[name] = mask;
        enforce();
    }
    State snapshot() const { return weights; }
    void restore(const State& s) {
        weights = s;
        enforce();  // masks survive restore
    }
    void fit(std::size_t) { enforce(); }  // training must not revive pruned weights
    float evaluate() const {
        float total = 0;
        for (auto& kv : weights)
            for (float w : kv.second) total += (w < 0 ? -w : w);
        return total / 100.0f;
    }
};

int main() {
    MockModel model;
    ImpConfig cfg;
    cfg.rounds = 4;
    cfg.prune_rate = 0.3f;
    cfg.epochs_per = 1;
    cfg.scope = Scope::Global;
    cfg.rewind = RewindPolicy::Init;

    auto res = ltkit::run_imp(model, cfg);

    assert(res.history.front().sparsity == 0.0f);
    for (std::size_t i = 1; i < res.history.size(); ++i)
        assert(res.history[i].sparsity >= res.history[i - 1].sparsity);
    assert(res.history.back().sparsity > 0.5f);

    for (auto& kv : res.masks) {
        const auto& w = model.weights[kv.first];
        for (std::size_t i = 0; i < kv.second.size(); ++i)
            if (!kv.second[i]) assert(w[i] == 0.0f);
    }
    std::printf("cpp smoke OK  final_sparsity=%.3f\n", res.history.back().sparsity);
    return 0;
}