#pragma once

#include <torch/torch.h>
#include "ltkit/ltkit.hpp"

namespace ltkit {

struct TorchMlp {
    using State = std::unordered_map<std::string, torch::Tensor>;

    std::vector<torch::nn::Linear> layers;
    std::unordered_map<std::string, torch::Tensor> masks;
    torch::Tensor x, y, xv, yv;
    double lr;
    torch::Device device;

    TorchMlp(const std::vector<int64_t>& dims,
             torch::Tensor x_,
             torch::Tensor y_,
             torch::Tensor xv_,
             torch::Tensor yv_,
             double lr_,
             torch::Device dev)
        : x(x_.to(dev).to(torch::kFloat32)),
          y(y_.to(dev).to(torch::kLong).flatten()),
          xv(xv_.to(dev).to(torch::kFloat32)),
          yv(yv_.to(dev).to(torch::kLong).flatten()),
          lr(lr_),
          device(dev) {
        TORCH_CHECK(dims.size() >= 2, "dims must have at least two entries");
        layers.reserve(dims.size() - 1);
        for (std::size_t i = 0; i + 1 < dims.size(); ++i) {
            layers.emplace_back(torch::nn::LinearOptions(dims[i], dims[i + 1]).bias(true));
            layers.back()->to(device);
        }
    }

    std::vector<std::string> parameter_groups() const {
        std::vector<std::string> groups;
        groups.reserve(layers.size());
        for (std::size_t i = 0; i < layers.size(); ++i) {
            groups.push_back(wname(i));
        }
        return groups;
    }

    std::vector<float> scores(const std::string& name, Criterion) const {
        const std::size_t i = layer_index(name);
        auto flat = layers[i]->weight.detach().abs().reshape({-1}).to(torch::kCPU).contiguous();
        const auto* ptr = flat.data_ptr<float>();
        return std::vector<float>(ptr, ptr + flat.numel());
    }

    void apply_mask(const std::string& name, const std::vector<bool>& mask) {
        const std::size_t i = layer_index(name);
        std::vector<float> mask_data;
        mask_data.reserve(mask.size());
        for (bool kept : mask) {
            mask_data.push_back(kept ? 1.0f : 0.0f);
        }
        auto mask_tensor = torch::tensor(mask_data, torch::TensorOptions().dtype(torch::kFloat32).device(device))
                               .reshape(layers[i]->weight.sizes());
        masks[name] = mask_tensor;
        {
            torch::NoGradGuard g;
            layers[i]->weight.mul_(mask_tensor);
        }
    }

    State snapshot() const {
        State s;
        s.reserve(layers.size() * 2);
        for (std::size_t i = 0; i < layers.size(); ++i) {
            s.emplace(wname(i), layers[i]->weight.detach().clone());
            s.emplace("l" + std::to_string(i) + ".bias", layers[i]->bias.detach().clone());
        }
        return s;
    }

    void restore(const State& s) {
        {
            torch::NoGradGuard g;
            for (std::size_t i = 0; i < layers.size(); ++i) {
                layers[i]->weight.copy_(s.at(wname(i)));
                layers[i]->bias.copy_(s.at("l" + std::to_string(i) + ".bias"));
            }
            reapply_masks();
        }
    }

    void fit(std::size_t epochs) {
        std::vector<torch::Tensor> params;
        params.reserve(layers.size() * 2);
        for (auto& layer : layers) {
            auto ps = layer->parameters();
            params.insert(params.end(), ps.begin(), ps.end());
        }
        torch::optim::Adam opt(params, torch::optim::AdamOptions(lr));
        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            opt.zero_grad();
            auto logits = forward(x);
            auto loss = torch::nn::functional::cross_entropy(logits, y);
            loss.backward();
            opt.step();
            {
                torch::NoGradGuard g;
                reapply_masks();
            }
        }
    }

    float evaluate() const {
        torch::NoGradGuard g;
        auto logits = forward(xv);
        auto pred = logits.argmax(1);
        auto correct = pred.eq(yv).sum().item<int64_t>();
        return float(correct) / float(yv.size(0));
    }

private:
    torch::Tensor forward(const torch::Tensor& in) const {
        auto h = in;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            auto& layer = const_cast<torch::nn::Linear&>(layers[i]);
            h = layer->forward(h);
            if (i + 1 != layers.size()) {
                h = torch::relu(h);
            }
        }
        return h;
    }

    std::string wname(std::size_t i) const {
        return "l" + std::to_string(i) + ".weight";
    }

    std::size_t layer_index(const std::string& name) const {
        const auto dot = name.find('.');
        TORCH_CHECK(dot != std::string::npos, "invalid parameter name: ", name);
        return static_cast<std::size_t>(std::stoull(name.substr(1, dot - 1)));
    }

    void reapply_masks() {
        for (const auto& entry : masks) {
            const std::size_t i = layer_index(entry.first);
            layers[i]->weight.mul_(entry.second);
        }
    }
};

} // namespace ltkit
