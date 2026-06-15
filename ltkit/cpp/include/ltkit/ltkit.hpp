// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Michael King — part of ltkit (https://github.com/BTFProdigy/ltkit)
// Licensed under the Apache License, Version 2.0. See the LICENSE and NOTICE
// files in the project root. Attribution must be retained in derivative works.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace ltkit {

/// LTKit IMP engine.
///
/// The backend contract uses the six verbs:
/// `parameter_groups`, `scores`, `apply_mask`, `snapshot`, `restore`, and `fit/evaluate`.
/// It preserves the mask-persistence invariant: once an element is pruned, it stays masked
/// through `fit` and across `restore`.
enum class Criterion {
    Magnitude,
    Gate,
    Random,
    Snip,
};

enum class RewindPolicy {
    Init,
    EarlyK,
    None,
};

enum class Scope {
    Global,
    PerGroup,
};

struct ImpConfig {
    std::size_t rounds = 4;
    float prune_rate = 0.20f;
    std::size_t epochs_per = 100;
    Criterion criterion = Criterion::Magnitude;
    RewindPolicy rewind = RewindPolicy::Init;
    Scope scope = Scope::Global;
    std::size_t early_k_epochs = 0;
    bool verbose = false;
};

struct RoundRecord {
    std::size_t round;
    float keep;
    float metric;
    float sparsity;
};

template<class M>
struct ImpResult {
    std::unordered_map<std::string, std::vector<bool>> masks;
    std::vector<RoundRecord> history;
};

namespace detail {

constexpr std::uint64_t kSeed = 0x4d595df4d0f33173ULL;

struct Candidate {
    std::uint32_t key;
    std::size_t group_pos;
    std::size_t index;
};

struct GroupView {
    const std::string* name;
    const std::vector<bool>* mask;
};

inline std::uint32_t total_order_key(float value) noexcept {
    union {
        float f;
        std::uint32_t u;
    } bits{value};

    return (bits.u & 0x80000000u) ? ~bits.u : (bits.u ^ 0x80000000u);
}

inline float random_score(std::mt19937_64& rng) noexcept {
    constexpr double inv_2pow53 = 1.0 / 9007199254740992.0;
    return static_cast<float>(static_cast<double>(rng() >> 11) * inv_2pow53);
}

inline long long target_kept(float keep, std::size_t count) noexcept {
    const double desired = static_cast<double>(keep) * static_cast<double>(count);
    long long target = static_cast<long long>(desired);
    if (static_cast<double>(target) > desired) {
        --target;
    }
    if (keep > 0.0f && target == 0 && count > 0) {
        target = 1;
    }
    return target;
}

template<class M>
inline std::vector<float> group_scores(const M& model,
                                       const std::string& name,
                                       Criterion criterion,
                                       std::size_t len,
                                       std::mt19937_64& rng) {
    if (criterion == Criterion::Random) {
        std::vector<float> scores;
        scores.reserve(len);
        for (std::size_t i = 0; i < len; ++i) {
            scores.push_back(random_score(rng));
        }
        return scores;
    }
    return model.scores(name, criterion);
}

template<class M>
inline std::unordered_map<std::string, std::vector<bool>> initial_masks(
    const M& model,
    const std::vector<std::string>& group_names,
    Criterion criterion) {
    std::unordered_map<std::string, std::vector<bool>> masks;
    masks.reserve(group_names.size());

    const Criterion score_criterion =
        (criterion == Criterion::Random) ? Criterion::Magnitude : criterion;

    for (const auto& name : group_names) {
        const auto scores = model.scores(name, score_criterion);
        masks.emplace(name, std::vector<bool>(scores.size(), true));
    }

    return masks;
}

inline float sparsity(const std::unordered_map<std::string, std::vector<bool>>& masks) noexcept {
    std::size_t total = 0;
    std::size_t kept = 0;
    for (const auto& entry : masks) {
        total += entry.second.size();
        kept += static_cast<std::size_t>(std::count(entry.second.begin(), entry.second.end(), true));
    }
    return total == 0 ? 0.0f : static_cast<float>(total - kept) / static_cast<float>(total);
}

template<class M>
inline std::unordered_map<std::string, std::vector<bool>> prune_per_group(
    const M& model,
    const std::vector<std::string>& group_names,
    const std::unordered_map<std::string, std::vector<bool>>& masks,
    Criterion criterion,
    float keep,
    std::mt19937_64& rng) {
    std::unordered_map<std::string, std::vector<bool>> next;
    next.reserve(group_names.size());

    for (const auto& name : group_names) {
        const auto mask_it = masks.find(name);
        const std::vector<bool>& current = mask_it->second;

        const auto scores = group_scores(model, name, criterion, current.size(), rng);

        std::vector<std::size_t> kept_idx;
        kept_idx.reserve(current.size());
        for (std::size_t i = 0; i < current.size(); ++i) {
            if (current[i]) {
                kept_idx.push_back(i);
            }
        }

        const long long target = target_kept(keep, kept_idx.size());
        if (target <= 0) {
            next.emplace(name, std::vector<bool>(current.size(), false));
            continue;
        }

        const std::size_t target_count = static_cast<std::size_t>(target);
        if (target_count >= kept_idx.size()) {
            next.emplace(name, current);
            continue;
        }

        std::vector<Candidate> candidates;
        candidates.reserve(kept_idx.size());
        for (std::size_t idx : kept_idx) {
            candidates.push_back(Candidate{total_order_key(scores[idx]), 0, idx});
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) {
                      if (a.key != b.key) {
                          return a.key > b.key;
                      }
                      return a.index < b.index;
                  });

        std::vector<bool> new_mask(current.size(), false);
        for (std::size_t i = 0; i < target_count; ++i) {
            new_mask[candidates[i].index] = true;
        }
        next.emplace(name, new_mask);
    }

    return next;
}

template<class M>
inline std::unordered_map<std::string, std::vector<bool>> prune_global(
    const M& model,
    const std::vector<std::string>& group_names,
    const std::unordered_map<std::string, std::vector<bool>>& masks,
    Criterion criterion,
    float keep,
    std::mt19937_64& rng) {
    std::vector<GroupView> groups;
    groups.reserve(group_names.size());

    std::vector<Candidate> candidates;
    std::size_t total_kept = 0;

    for (std::size_t group_pos = 0; group_pos < group_names.size(); ++group_pos) {
        const auto& name = group_names[group_pos];
        const auto mask_it = masks.find(name);
        const std::vector<bool>& current = mask_it->second;

        const auto scores = group_scores(model, name, criterion, current.size(), rng);

        std::size_t kept_count = 0;
        for (std::size_t idx = 0; idx < current.size(); ++idx) {
            if (current[idx]) {
                ++kept_count;
                candidates.push_back(
                    Candidate{total_order_key(scores[idx]), group_pos, idx});
            }
        }

        total_kept += kept_count;
        groups.push_back(GroupView{&name, &current});
    }

    const long long target = target_kept(keep, total_kept);
    if (target <= 0) {
        std::unordered_map<std::string, std::vector<bool>> next;
        next.reserve(group_names.size());
        for (const auto& group : groups) {
            next.emplace(*group.name, std::vector<bool>(group.mask->size(), false));
        }
        return next;
    }

    const std::size_t target_count = static_cast<std::size_t>(target);
    if (target_count >= total_kept) {
        std::unordered_map<std::string, std::vector<bool>> next;
        next.reserve(group_names.size());
        for (const auto& group : groups) {
            next.emplace(*group.name, *group.mask);
        }
        return next;
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  if (a.key != b.key) {
                      return a.key > b.key;
                  }
                  if (a.group_pos != b.group_pos) {
                      return a.group_pos < b.group_pos;
                  }
                  return a.index < b.index;
              });

    std::vector<std::vector<bool>> next_masks;
    next_masks.reserve(groups.size());
    for (const auto& group : groups) {
        next_masks.emplace_back(group.mask->size(), false);
    }

    for (std::size_t i = 0; i < target_count; ++i) {
        next_masks[candidates[i].group_pos][candidates[i].index] = true;
    }

    std::unordered_map<std::string, std::vector<bool>> next;
    next.reserve(group_names.size());
    for (std::size_t i = 0; i < groups.size(); ++i) {
        next.emplace(*groups[i].name, next_masks[i]);
    }
    return next;
}

template<class M>
inline void apply_masks(M& model,
                        const std::vector<std::string>& group_names,
                        const std::unordered_map<std::string, std::vector<bool>>& masks) {
    for (const auto& name : group_names) {
        const auto it = masks.find(name);
        if (it != masks.end()) {
            model.apply_mask(name, it->second);
        }
    }
}

} // namespace detail

/// Run IMP with the six backend verbs and the mask-persistence invariant.
template<class M>
ImpResult<M> run_imp(M& model, const ImpConfig& config) {
    const std::vector<std::string> group_names = model.parameter_groups();
    std::unordered_map<std::string, std::vector<bool>> masks =
        detail::initial_masks(model, group_names, config.criterion);

    const auto init_state = model.snapshot();
    const auto rewind_state = [&]() {
        if (config.rewind == RewindPolicy::EarlyK && config.early_k_epochs > 0 && config.rounds > 0) {
            model.restore(init_state);
            model.fit(config.early_k_epochs);
            return model.snapshot();
        }
        return init_state;
    }();

    std::vector<RoundRecord> history;
    history.reserve(config.rounds);

    float keep = 1.0f;
    std::mt19937_64 rng(detail::kSeed);

    for (std::size_t round = 0; round < config.rounds; ++round) {
        if (config.rewind != RewindPolicy::None) {
            model.restore(rewind_state);
        }

        detail::apply_masks(model, group_names, masks);

        model.fit(config.epochs_per);
        const float metric = model.evaluate();

        history.push_back(RoundRecord{round, keep, metric, detail::sparsity(masks)});

        keep *= (1.0f - config.prune_rate);

        if (config.scope == Scope::Global) {
            masks = detail::prune_global(model, group_names, masks, config.criterion, keep, rng);
        } else {
            masks = detail::prune_per_group(model, group_names, masks, config.criterion, keep, rng);
        }
    }

    detail::apply_masks(model, group_names, masks);

    return ImpResult<M>{masks, history};
}

} // namespace ltkit
