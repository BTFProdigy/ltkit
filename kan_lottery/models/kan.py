"""
Ternary KAN with per-edge Chebyshev basis expansion and sigmoid gates.
φ_ij(x_j) = Σ_k  tanh(5*a_ijk) * T_k(tanh(x_j))
gate g_ij multiplies the entire edge output.
"""
import torch
import torch.nn as nn
from .gates import HardConcreteGate, ternarize, quintary_ste, bitnet_ternary, bitnet_ternary_row

_QUANTIZERS = {"soft": ternarize, "q5": quintary_ste, "bitnet": bitnet_ternary, "bitnet_row": bitnet_ternary_row}


def _chebyshev_basis(x: torch.Tensor, order: int) -> torch.Tensor:
    """Chebyshev T_0..T_{order-1}. x assumed in [-1,1]. Returns (..., order)."""
    x = x.clamp(-1.0, 1.0)
    polys = [torch.ones_like(x), x]
    for k in range(2, order):
        polys.append(2 * x * polys[-1] - polys[-2])
    return torch.stack(polys[:order], dim=-1)


class TernaryKANLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int, basis_order: int = 6,
                 activation: bool = True, quant: str = "soft"):
        super().__init__()
        self.in_f        = in_f
        self.out_f       = out_f
        self.basis_order = basis_order
        self._quantize   = _QUANTIZERS[quant]

        # Coefficients (out_f, in_f, basis_order) — small init so tanh stays in linear regime
        self.coeffs = nn.Parameter(0.1 * torch.randn(out_f, in_f, basis_order))
        self.register_buffer('coeff_mask', torch.ones_like(self.coeffs))
        self.bias   = nn.Parameter(torch.zeros(out_f))
        self.gate   = HardConcreteGate((out_f, in_f))
        self.scale  = nn.Parameter(torch.ones(out_f))
        self.act    = nn.ReLU() if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise inputs → [-1,1]
        x_norm = torch.tanh(x)                          # (batch, in_f)
        # Basis: (batch, in_f, order)
        B = _chebyshev_basis(x_norm, self.basis_order)
        # Soft-ternary coefficients
        c_q = self._quantize(self.coeffs) * self.coeff_mask                # (out_f, in_f, order)
        # φ_ij(x_j) = Σ_k c_ijk * B_k(x_j)  → (batch, out_f, in_f)
        phi = torch.einsum("bik,oik->boi", B, c_q)
        # Gates
        g   = self.gate()                                # (out_f, in_f)
        phi = phi * g.unsqueeze(0)
        out = phi.sum(dim=-1) * self.scale + self.bias
        return self.act(out)

    def l0_penalty(self):      return self.gate.l0_penalty()
    def complexity_penalty(self): return self.coeffs.abs().sum()
    def coeff_magnitude(self): return self.coeffs.detach().abs()
    def apply_coeff_mask(self, mask):
        with torch.no_grad():
            self.coeff_mask.copy_(mask)
    def sparsity(self):         return 1.0 - self.gate.hard_mask().mean().item()
    def hard_mask(self):        return self.gate.hard_mask()

    def active_basis_per_edge(self) -> float:
        mask = self.gate.hard_mask().bool()
        nnz  = (self.coeffs.abs() >= 0.3).float().sum(-1)
        return nnz[mask].mean().item() if mask.any() else 0.0

    def coefficient_entropy(self) -> float:
        vals = self.coeffs.abs().detach().flatten() + 1e-8
        vals = vals / vals.sum()
        return -(vals * vals.log()).sum().item()


class TernaryKAN(nn.Module):
    def __init__(self, dims: list, basis_order: int = 6, quant: str = "soft"):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            last = (i == len(dims) - 2)
            layers.append(TernaryKANLayer(dims[i], dims[i+1], basis_order,
                                          activation=not last, quant=quant))
        self.layers      = nn.ModuleList(layers)
        self.basis_order = basis_order

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def l0_penalty(self):
        return sum(l.l0_penalty() for l in self.layers)

    def complexity_penalty(self):
        return sum(l.complexity_penalty() for l in self.layers)

    def total_gates(self):
        return sum(l.gate.log_alpha.numel() for l in self.layers)

    def active_gates(self):
        return int(sum(l.gate.hard_mask().sum().item() for l in self.layers))

    def sparsity(self):
        return 1.0 - self.active_gates() / max(self.total_gates(), 1)

    def coeff_total(self):
        return sum(l.coeff_mask.numel() for l in self.layers)

    def coeff_active(self):
        return int(sum(l.coeff_mask.sum().item() for l in self.layers))

    def coeff_sparsity(self):
        return 1.0 - self.coeff_active() / max(self.coeff_total(), 1)

    def get_masks(self):
        return [l.hard_mask() for l in self.layers]

    def apply_masks(self, masks):
        for layer, mask in zip(self.layers, masks):
            with torch.no_grad():
                layer.gate.log_alpha.data = torch.where(
                    mask.bool(),
                    torch.full_like(layer.gate.log_alpha, 6.0),
                    torch.full_like(layer.gate.log_alpha, -6.0),
                )

    def functional_complexity(self):
        ab  = [l.active_basis_per_edge() for l in self.layers]
        ent = [l.coefficient_entropy()   for l in self.layers]
        return {
            "mean_active_basis":   sum(ab)  / len(ab),
            "mean_coeff_entropy":  sum(ent) / len(ent),
        }
