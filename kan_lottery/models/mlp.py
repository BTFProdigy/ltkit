"""
Ternary MLP with per-edge sigmoid gates and tanh-based quantization.
"""
import torch
import torch.nn as nn
from .gates import HardConcreteGate, ternarize


class TernaryMLPLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int, activation: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias   = nn.Parameter(torch.zeros(out_f))
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        # Scale init so tanh(5*w) ≈ ±1 for ~half of weights
        with torch.no_grad():
            self.weight.data *= 0.3
        self.gate = HardConcreteGate((out_f, in_f))
        self.act  = nn.ReLU() if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = ternarize(self.weight)          # tanh-based soft ternary
        g   = self.gate()                     # (out_f, in_f)
        out = x @ (w_q * g).T + self.bias
        return self.act(out)

    def l0_penalty(self):
        return self.gate.l0_penalty()

    def sparsity(self) -> float:
        return 1.0 - self.gate.hard_mask().mean().item()

    def hard_mask(self):
        return self.gate.hard_mask()


class TernaryMLP(nn.Module):
    def __init__(self, dims: list):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            last = (i == len(dims) - 2)
            layers.append(TernaryMLPLayer(dims[i], dims[i + 1], activation=not last))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def l0_penalty(self) -> torch.Tensor:
        return sum(l.l0_penalty() for l in self.layers)

    def complexity_penalty(self) -> torch.Tensor:
        return torch.tensor(0.0)   # MLP has no basis complexity

    def total_gates(self) -> int:
        return sum(l.gate.log_alpha.numel() for l in self.layers)

    def active_gates(self) -> int:
        return int(sum(l.gate.hard_mask().sum().item() for l in self.layers))

    def sparsity(self) -> float:
        return 1.0 - self.active_gates() / max(self.total_gates(), 1)

    def get_masks(self) -> list:
        return [l.hard_mask() for l in self.layers]

    def apply_masks(self, masks: list):
        for layer, mask in zip(self.layers, masks):
            with torch.no_grad():
                layer.gate.log_alpha.data = torch.where(
                    mask.bool(),
                    torch.full_like(layer.gate.log_alpha, 6.0),
                    torch.full_like(layer.gate.log_alpha, -6.0),
                )
