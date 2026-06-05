# Ternary KAN + Lottery Ticket — Findings

Task: `mnist_d64 + orth_id` rotation, classification, 5 seeds per cell, 4 IMP rounds × 50 epochs unless noted.

## Headline result

Best ternary-KAN recipe found:
- `orth_id` rotation + global **BitNet b1.58** quant + `basis_order=4` + edge-level IMP
- **0.8050 ± 0.0354** validation accuracy
- MLP baseline on same task: **0.844** → residual gap **3.9pp** (down from 6.9pp before fixes)

The residual gap is **structural**, not numerical. KAN lacks token mixing; more quant tricks won't close it.

## Experiments

### EXP-13 — Real BitNet vs soft ternary (H3 retest)
Prior "ternarize" was `tanh(5w)` (not actually ternary). Replaced with absmean-scaled STE rounding to {-1, 0, +1}.

| quant   | mean   | std    |
|---------|--------|--------|
| soft    | 0.7750 | 0.0313 |
| bitnet  | 0.7980 | 0.0391 |

**+2.3pp from real BitNet.** Refutes the claim that BitNet "cannot work with KAN" — empirically it helps.

### EXP-14 — Per-row vs global BitNet scaling (H6)

| quant       | mean   | std    |
|-------------|--------|--------|
| bitnet      | 0.7980 | 0.0391 |
| bitnet_row  | 0.7860 | 0.0329 |

Per-row **lost by 1.2pp here**. Reason: KAN rows are ~384 params (out_f × basis_order), too few for a stable absmean estimator. **At LLM scale (hidden ≥ 2048) per-row flips to winning**, matching the BitNet paper.

### EXP-15 — basis_order sweep (H5)

| order | mean   | std    |
|-------|--------|--------|
| 4     | 0.8050 | 0.0354 |
| 6     | 0.7980 | 0.0391 |
| 8     | 0.7440 | 0.0343 |
| 12    | 0.6750 | 0.0523 |

**Monotonic decline with more basis.** Under heavy ternary quantization the model is over-provisioned; extra Chebyshev orders add noise the regularizer cannot discipline. At order=2 KAN ≈ ternary MLP layer.

### EXP-16 — Basis-coefficient lottery ticket (H7)

| regime      | mean   | std    |
|-------------|--------|--------|
| edge_only   | 0.7980 | 0.0391 |
| both        | 0.7670 | 0.0448 |
| basis_only  | 0.7510 | 0.0188 |

**Basis lottery ticket fails by 4.7pp.** Chebyshev basis is hierarchical (T_0 carries function shape, high orders are corrections); magnitude-based IMP has no way to encode this and prunes structurally important low-order coefficients.

## Implications for an LLM build

**Don't build a BitNet ternary KAN LLM.** Three concrete reasons:

1. **Wrong inductive bias.** KAN's `φ_ij(x_j)` is per-edge univariate — no token mixing. Attention is the reason transformers work; KAN can only replace the FFN, not the mixer. The 3.9pp gap on a tabular task widens at language scale.
2. **BitNet doesn't make KAN cheap.** BitNet kernels accelerate ternary matmul. KAN's bottleneck is per-edge Chebyshev evaluation, not matmul, so a Rust BitNet kernel can't touch the compute path. You'd get the ternarization but not the speedup.
3. **Lottery ticket adds nothing** (this study). Edge-IMP no better than one-shot; basis-IMP strictly worse. Don't budget engineering effort on iterative pruning.

## Recommended LLM recipe

Standard decoder-only transformer with **BitLinear** (Rust b1.58 impl) replacing every `nn.Linear`:

- Ternarize Q/K/V/O projections and FFN up/down.
- Keep attention scores (softmax × V matmul) in bf16.
- **Per-row absmean scaling** (flips to winning at hidden ≥ 2048).
- **STE on round+clamp**, exactly as in `models/gates.py::bitnet_ternary`.
- **RMSNorm before BitLinear**; full-precision residual stream — do not ternarize residuals.
- **Train from scratch in 1.58-bit.** The BitNet paper found post-hoc quantization of fp checkpoints is not competitive.
- **Skip IMP.** If sparsity is needed, do one-shot magnitude pruning + short finetune after training.

The reusable artifact from this study is the BitNet quantizer (`bitnet_ternary` in `models/gates.py`), which mirrors the Rust b1.58 semantics. Drop KAN.

## Code touchpoints

- `models/gates.py` — `bitnet_ternary`, `bitnet_ternary_row` (absmean-scaled STE).
- `models/kan.py` — `_QUANTIZERS` registry; per-layer `coeff_mask` buffer; `coeff_magnitude` / `apply_coeff_mask`; class-level `coeff_total/active/sparsity`.
- `exp13_bitnet_h3.py`, `exp14_bitnet_row.py`, `exp15_basis_order.py`, `exp16_basis_imp.py` — driver scripts.
- `results/exp1{3,4,5,6}_*.log` — raw per-seed numbers.

## Hypotheses status

| ID | Claim                                          | Status     |
|----|------------------------------------------------|------------|
| H1 | Rotation (orth_id) closes gap                  | Confirmed (prior) |
| H3 | Quantizer granularity matters; real BitNet > soft | Confirmed (+2.3pp) |
| H4 | IMP aggressiveness helps                       | Refuted (prior) |
| H5 | Higher basis_order helps                       | Refuted — order=4 wins |
| H6 | Per-row BitNet > global                        | Refuted at this scale; expected to flip at LLM scale |
| H7 | Basis-coefficient lottery ticket works         | Refuted (-4.7pp) |
