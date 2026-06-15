// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Michael King — part of ltkit (https://github.com/BTFProdigy/ltkit)
// Licensed under the Apache License, Version 2.0. See the LICENSE and NOTICE
// files in the project root. Attribution must be retained in derivative works.
#pragma once

#include <torch/torch.h>

#include "ltkit/ltkit.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ltkit {

struct TorchModel {
    using State = std::unordered_map<std::string, torch::Tensor>;

    std::shared_ptr<torch::nn::Module> module;
    std::function<torch::Tensor(torch::Tensor)> forward_fn;
    std::function<float(const torch::Tensor&, const torch::Tensor&)> eval_fn;
    std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)> loss_fn;
    std::unordered_map<std::string, torch::Tensor> masks;
    torch::Tensor x, y, xv, yv;
    double lr;
    torch::Device device;

    TorchModel(std::shared_ptr<torch::nn::Module> mod,
               std::function<torch::Tensor(torch::Tensor)> fwd,
               torch::Tensor x_, torch::Tensor y_, torch::Tensor xv_, torch::Tensor yv_,
               double lr_, torch::Device dev,
               std::function<float(const torch::Tensor&, const torch::Tensor&)> ev = {},
               std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)> lf = {})
        : module(mod),
          forward_fn(fwd),
          eval_fn(ev),
          loss_fn(lf),
          x(x_.to(dev).to(torch::kFloat32)),
          y(y_.to(dev).to(torch::kLong).flatten()),
          xv(xv_.to(dev).to(torch::kFloat32)),
          yv(yv_.to(dev).to(torch::kLong).flatten()),
          lr(lr_),
          device(dev) {
        TORCH_CHECK(module, "TorchModel requires a module");
        TORCH_CHECK(forward_fn, "TorchModel requires a forward function");
    }

    torch::Tensor param(const std::string& name) const {
        auto params = module->named_parameters();
        for (const auto& item : params) {
            if (item.key() == name) {
                return item.value();
            }
        }
        TORCH_CHECK(false, "missing parameter ", name);
    }

    std::vector<std::string> parameter_groups() const {
        std::vector<std::string> groups;
        auto params = module->named_parameters();
        for (const auto& item : params) {
            const auto& name = item.key();
            if (name.size() >= 7 && name.compare(name.size() - 7, 7, ".weight") == 0 &&
                item.value().dim() >= 2) {
                groups.push_back(name);
            }
        }
        std::sort(groups.begin(), groups.end());
        return groups;
    }

    std::vector<float> scores(const std::string& name, Criterion) const {
        auto flat = param(name).detach().abs().reshape({-1}).to(torch::kCPU).contiguous();
        const auto* ptr = flat.data_ptr<float>();
        return std::vector<float>(ptr, ptr + flat.numel());
    }

    void apply_mask(const std::string& name, const std::vector<bool>& mask) {
        auto w = param(name);
        std::vector<float> mask_data;
        mask_data.reserve(mask.size());
        for (bool kept : mask) {
            mask_data.push_back(kept ? 1.0f : 0.0f);
        }
        auto mask_tensor = torch::tensor(mask_data, torch::TensorOptions().dtype(torch::kFloat32).device(device))
                               .reshape(w.sizes());
        masks[name] = mask_tensor;
        torch::NoGradGuard g;
        w.mul_(mask_tensor);
    }

    State snapshot() const {
        State s;
        auto params = module->named_parameters();
        for (const auto& item : params) {
            s.emplace(item.key(), item.value().detach().clone());
        }
        return s;
    }

    void restore(const State& s) {
        torch::NoGradGuard g;
        auto params = module->named_parameters();
        for (const auto& item : params) {
            const auto it = s.find(item.key());
            if (it != s.end()) {
                item.value().copy_(it->second);
            }
        }
        reapply_masks();
    }

    void fit(std::size_t epochs) {
        module->train();
        auto params = module->parameters();
        torch::optim::Adam opt(params, torch::optim::AdamOptions(lr));
        for (std::size_t e = 0; e < epochs; ++e) {
            opt.zero_grad();
            auto out = forward_fn(x);
            auto loss = loss_fn ? loss_fn(out, y) : torch::nn::functional::cross_entropy(out, y);
            loss.backward();
            opt.step();
            reapply_masks();
        }
    }

    float evaluate() const {
        module->eval();
        torch::NoGradGuard g;
        auto out = forward_fn(xv);
        if (eval_fn) {
            return eval_fn(out, yv);
        }
        auto pred = out.argmax(1);
        auto correct = pred.eq(yv).sum().item<int64_t>();
        return yv.size(0) == 0 ? 0.0f : float(correct) / float(yv.size(0));
    }

private:
    void reapply_masks() {
        torch::NoGradGuard g;
        for (const auto& kv : masks) {
            param(kv.first).mul_(kv.second);
        }
    }
};

struct TorchMlp {
    static TorchModel make(const std::vector<int64_t>& dims,
                           torch::Tensor x, torch::Tensor y, torch::Tensor xv, torch::Tensor yv,
                           double lr, torch::Device dev) {
        TORCH_CHECK(dims.size() >= 2, "dims must have at least two entries");
        auto seq = torch::nn::Sequential();
        for (std::size_t i = 0; i + 1 < dims.size(); ++i) {
            seq->push_back(torch::nn::Linear(torch::nn::LinearOptions(dims[i], dims[i + 1]).bias(true)));
            if (i + 2 < dims.size()) {
                seq->push_back(torch::nn::Functional([](torch::Tensor t) { return torch::relu(t); }));
            }
        }
        seq->to(dev);
        std::shared_ptr<torch::nn::Module> mod = std::static_pointer_cast<torch::nn::Module>(seq.ptr());
        auto fwd = [seq](torch::Tensor in) mutable { return seq->forward(in); };
        return TorchModel(mod, fwd, x, y, xv, yv, lr, dev);
    }
};

} // namespace ltkit