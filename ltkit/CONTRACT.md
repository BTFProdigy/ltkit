# LTKit — the backend contract

LTKit finds lottery tickets (Frankle & Carbin) in *any* trainable model by
driving it through **six verbs**. The IMP engine never touches a tensor
directly; it only calls these six. Every backend — Python, Rust, C++ — implements
the same six. That is the entire portability story.

## The six verbs

| Verb | Signature (conceptual) | Meaning |
|------|------------------------|---------|
| `parameter_groups` | `() -> {name: Group}` | Enumerate the prunable parameter tensors. A *group* is one logical tensor (e.g. a Linear weight) with a stable string name. |
| `scores` | `(name, criterion) -> f32[]` | Per-element importance for one group, flattened in a fixed canonical order. Higher = more important = keep. |
| `apply_mask` | `(name, mask: bool[]) -> ()` | Zero out elements where `mask` is false, and **keep them zero** for the rest of this round (frozen — no gradient revives them). Same canonical order as `scores`. |
| `snapshot` | `() -> State` | Capture init weights θ₀ (and RNG, optimizer if rewind needs it). Opaque handle. |
| `restore` | `(State) -> ()` | Reset trainable weights to a prior snapshot. Masks already applied stay applied. |
| `fit` | `(data, epochs) -> ()` | Train the (masked) model in place for `epochs`. |
| `evaluate` | `(data) -> f32` | Validation metric (higher = better; engine is metric-agnostic). |

> Seven rows, six "verbs": `evaluate` is bundled with `fit` as the eval half of
> the train/eval pair. Counted as the sixth capability.

### Canonical order invariant
For a given group, `scores`, `apply_mask`, and the model's internal layout MUST
agree element-for-element. Index *i* of the score vector, index *i* of the mask,
and the *i*-th weight are the same parameter. Backends pick the order (row-major
is the obvious choice); they must be consistent across all three verbs.

### Mask persistence invariant
Once `apply_mask` zeros an element, it stays zero through `fit` and survives
`restore`. The standard trick: keep a boolean mask buffer per group and
re-multiply after every optimizer step (or register a gradient hook). A backend
that lets a pruned weight drift off zero is **non-conformant** and breaks the
lottery-ticket guarantee.

## The engine (language-agnostic pseudocode)

This is ~40 lines and identical in every language. It is the *only* thing that
needs porting; everything heavy lives behind the six verbs.

```
fn imp(model, data, rounds, prune_rate, criterion, rewind):
    init = model.snapshot()                 # θ₀
    masks = { g: all_true for g in model.parameter_groups() }
    keep = 1.0
    history = []

    for r in 0..rounds:
        model.restore(init)                 # rewind to θ₀ (or early-k snapshot)
        for g, m in masks: model.apply_mask(g, m)

        model.fit(data, epochs)
        history.push({ round: r, keep: keep, metric: model.evaluate(data) })

        keep *= (1.0 - prune_rate)          # e.g. 0.8 → prune 20% of survivors
        for g in model.parameter_groups():
            s = model.scores(g, criterion)
            thresh = kth_smallest(s[where masks[g]], keep)   # global or per-group
            masks[g] = masks[g] AND (s >= thresh)

    return { masks, history }
```

`rewind` policy selects what `restore` targets:
- `init` — classic Frankle & Carbin (reset to θ₀).
- `early_k` — reset to a snapshot taken k steps in (Frankle et al. "rewinding").
- `none` — one-shot pruning (no restore; the only mode an inference-only runtime could support, but those are out of scope here).

## Criteria
`criterion` is a pure function `(group_state) -> f32[]`:
- `magnitude` — `|weight|`. The default; what classic IMP uses.
- `gate` — a learned per-element gate value (HardConcrete etc.). This is the hook
  the deferred KAN backend will use; its gates ARE its scores.
- `random` — control baseline.
- `snip` — `|weight * grad|` from one batch (single-shot, pre-training).

## Scope of backends (decided)

| Language | Backend | Mechanism |
|----------|---------|-----------|
| Python | torch | native; scores = `|weight|` over `nn.Linear/Conv` weights |
| Python | keras/tf | native; `model.trainable_weights` |
| Python | jax/flax | native; pytree of params |
| Rust | tch-rs | libtorch bindings — same tensor core as Python torch |
| Rust | candle | native `VarMap`; target for ternary/BitNet LLMs |
| C++ | libtorch | the C++ torch API; same core again |

`tch-rs` (Rust) and libtorch (C++) and PyTorch (Python) are the same libtorch
engine behind three language skins — one conceptual adapter, three bindings.
Candle is the only fully-independent backend.

KAN (the gate-based reference model in `../kan_lottery/`) is **deferred**; it
plugs in later as a torch backend whose `criterion` is `gate` instead of
`magnitude`. Nothing in the engine changes to accommodate it — proof the
contract is right.
