"""
Quantization and gate utilities.

Ternary quantization via tanh-based relaxation (forward ≈ ternary, grad flows cleanly).
Gate: sigmoid(log_alpha); L1 sparsity penalty.
"""
import torch
import torch.nn as nn


# ── Ternary Quantization (tanh-based) ────────────────────────────────────────

def ternarize(w: torch.Tensor, temperature: float = 5.0) -> torch.Tensor:
    """
    Soft-ternary: tanh(temperature * w).
    - Saturates near ±1 for |w| > 0.5; near 0 for |w| small
    - Gradient flows everywhere (no dead zones)
    """
    return torch.tanh(temperature * w)


def hard_ternarize(w: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """Hard ternary for final evaluation."""
    return w.sign() * (w.abs() >= threshold).float()


_Q5_LEVELS = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])


def quintary_ste(w: torch.Tensor) -> torch.Tensor:
    """5-level STE: forward snaps to nearest of {-1,-0.5,0,+0.5,+1} on tanh(5w),
    backward is identity through the smooth tanh."""
    s = torch.tanh(5.0 * w)
    levels = _Q5_LEVELS.to(s.device)
    idx = (s.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    s_q = levels[idx]
    return s + (s_q - s).detach()


def bitnet_ternary(w: torch.Tensor) -> torch.Tensor:
    scale = w.abs().mean() + 1e-5
    w_s = w / scale
    w_q = torch.round(w_s).clamp(-1, 1)
    return w + (w_q * scale - w).detach()


def bitnet_ternary_row(w: torch.Tensor) -> torch.Tensor:
    scale = w.abs().mean(dim=(1, 2), keepdim=True) + 1e-5
    w_s = w / scale
    w_q = torch.round(w_s).clamp(-1, 1)
    return w + (w_q * scale - w).detach()


# ── Sigmoid Gate ──────────────────────────────────────────────────────────────

class HardConcreteGate(nn.Module):
    """
    Per-element gate g = sigmoid(log_alpha) in (0, 1).
    Sparsity via L1 penalty on g.  Hard mask: log_alpha < 0 -> pruned.
    """

    def __init__(self, shape, init_mean: float = 2.0,
                 temperature: float = 1.0,
                 beta: float = 0.66, zeta: float = 1.1, gamma: float = -0.1):
        super().__init__()
        self.temperature = temperature
        self.log_alpha = nn.Parameter(
            torch.full(shape, float(init_mean)) + 0.01 * torch.randn(shape)
        )

    def l0_penalty(self) -> torch.Tensor:
        return torch.sigmoid(self.log_alpha).sum()

    def forward(self, deterministic: bool = False) -> torch.Tensor:
        if deterministic or not self.training:
            return torch.sigmoid(self.log_alpha)
        noise = 0.05 * torch.randn_like(self.log_alpha)
        return torch.sigmoid((self.log_alpha + noise) / self.temperature)

    def hard_mask(self, threshold: float = 0.5) -> torch.Tensor:
        return (self.log_alpha >= 0).float()
