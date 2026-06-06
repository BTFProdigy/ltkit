# LTKit

A backend-agnostic framework for finding **lottery tickets** (Frankle & Carbin
iterative magnitude pruning with rewind) in trainable models — across PyTorch,
TensorFlow, JAX, Rust (candle + tch-rs), and C++ (libtorch).

The whole design rests on one idea: the IMP algorithm never touches a tensor.
It drives any model through **six verbs**. Every backend in every language
implements those six. See [`CONTRACT.md`](CONTRACT.md) for the authoritative spec.

```
parameter_groups → scores → apply_mask → snapshot → restore → fit/evaluate
```

The engine itself is ~40 lines of pure orchestration, ported verbatim to each
language. All heavy tensor math lives behind the contract.

## Status

| Language | Core engine | Backend | Test |
|----------|:----------:|---------|:----:|
| Python | ✅ | torch | ✅ `tests/test_smoke.py` |
| Python | ✅ | keras/tf | ✅ `tests/test_keras_smoke.py` |
| Python | ✅ | jax (+optax) | ✅ `tests/test_jax_smoke.py` |
| Rust | ✅ | candle (BitNet target) | ✅ `cargo test --features candle` |
| Rust | ✅ | tch-rs (libtorch) | ✅ `cargo test --features tch` |
| C++ | ✅ | libtorch | ✅ `cmake -DTORCH_DIR=… && torch_smoke` |

Every implemented core is validated against the **same invariants**: sparsity
rises monotonically, the returned ticket mask agrees with the model's zeroed
weights, and no pruned weight is revived by training or rewind.

## Layout

```
ltkit/
  CONTRACT.md            canonical six-verb spec (language-agnostic)
  python/
    ltkit/core/          protocol.py, imp.py (engine)
    ltkit/backends/      torch_backend.py, keras_backend.py, jax_backend.py
    tests/test_smoke.py
    pyproject.toml
  rust/
    src/contract.rs      trait PrunableModel + enums
    src/imp.rs           engine (generic over T: PrunableModel)
    src/backends/candle.rs, tch_backend.rs
    tests/{smoke,candle_smoke,tch_smoke}.rs
    Cargo.toml           candle / tch behind optional features
  cpp/
    include/ltkit/ltkit.hpp          header-only engine (template-duck-typed model)
    include/ltkit/torch_backend.hpp  generic libtorch backend (TorchModel + TorchMlp)
    tests/{smoke,torch_smoke}.cpp
    CMakeLists.txt
```

## Running

```bash
# Python (torch + keras/tf)
cd python && PYTHONPATH=$PWD python3 tests/test_smoke.py        # -> smoke OK
KERAS_BACKEND=tensorflow PYTHONPATH=$PWD python3 tests/test_keras_smoke.py  # -> keras smoke OK
JAX_PLATFORMS=cpu PYTHONPATH=$PWD python3 tests/test_jax_smoke.py           # -> jax smoke OK

# Rust core + candle backend
cd rust && cargo test                                          # pure core
cargo test --features candle                                   # candle backend

# Rust tch-rs (libtorch) backend — download-libtorch fetches a CPU libtorch.
# On this box the NVIDIA HPC SDK headers shadow GCC's intrinsics, so force g++
# and drop the NVHPC include path:
env -u CPLUS_INCLUDE_PATH CXX=g++ CC=gcc cargo test --features tch  # -> tch_imp_smoke ok

# C++ header-only core
cd cpp && g++ -std=c++17 -I include tests/smoke.cpp -o smoke && ./smoke

# C++ libtorch backend (use g++, not clang; pip wheel is pre-CXX11 ABI)
cd cpp && cmake -S . -B build -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DTORCH_DIR="$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))')"
cmake --build build -j && ./build/torch_smoke
```

## Usage sketch (Python / torch)

```python
import torch.nn as nn
from ltkit import IMPConfig, Criterion, RewindPolicy, run_imp
from ltkit.backends import TorchBackend

model   = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 3))
backend = TorchBackend(model, task="classification")
cfg     = IMPConfig(rounds=4, prune_rate=0.3, criterion=Criterion.MAGNITUDE,
                    rewind=RewindPolicy.INIT, scope="global")
result  = run_imp(backend, (X, y), (Xv, yv), cfg)
# result.masks  -> the winning ticket;  result.history -> per-round metric/sparsity
```

## Using it on your own model (any architecture)

Every backend is a **generic wrapper**, not an MLP — the MLP in the tests is just
a fixture. Hand it any model you build. The two Python backends auto-discover
prunable layers; the others take your model plus a forward function.

```python
# Vision / generative model, custom prunable layers + a generative metric.
import torch, torch.nn as nn
from ltkit import IMPConfig, Criterion, RewindPolicy, run_imp
from ltkit.backends import TorchBackend

model = MyUNet(...)                      # any nn.Module (CNN, ViT, diffusion, …)

def fid_metric(model, data):             # higher = better, so return -FID / -FAD
    return -compute_fid(model, data)

def diff_loss(model, x, y):              # custom training objective (no labels needed)
    return model.diffusion_loss(x)

backend = TorchBackend(
    model,
    prunable_types=(nn.Linear, nn.Conv2d, nn.Conv1d, nn.ConvTranspose2d),
    loss_fn=diff_loss,                   # optional: overrides CE/MSE
    eval_fn=fid_metric,                  # optional: overrides accuracy
)
cfg = IMPConfig(rounds=8, prune_rate=0.2, criterion=Criterion.MAGNITUDE,
                rewind=RewindPolicy.NONE,  # NONE = one-shot, cheap for big gen models
                scope="global")
result = run_imp(backend, (X, y), (Xv, yv), cfg)
```

- **`prunable_types`** (torch) selects which layer classes to prune; Conv/ConvTranspose
  are included by default. Keras auto-prunes any layer with a `.kernel` (Dense + all Conv).
- **`eval_fn(model, data) -> float`** supplies a domain metric (FID/FAD/perceptual);
  the engine only needs "higher is better". **`loss_fn(model, x, y)`** supplies a custom
  training objective. Both default to classification/regression when omitted.
- **`RewindPolicy.NONE`** is one-shot pruning (prune once, no retrain-from-rewind) —
  the practical mode for large diffusion/audio models where full IMP is too expensive.

The non-Python backends are generic too — pass your model + a forward fn:

| Backend | Entry point | You provide |
|---|---|---|
| Python torch / keras | `TorchBackend(model, …)` / `KerasBackend(model, …)` | the model (layers auto-discovered) |
| Python jax | `JaxBackend(params, apply_fn, …)` | params pytree + `apply_fn(params, X)` (prunable = `ndim>=2`) |
| Rust candle | `CandleModel::new(varmap, forward_fn, …)` | a `VarMap` + `Fn(&Tensor)->Result<Tensor>` |
| Rust tch | `TchModel::new(vs, forward_fn, …)` | a `VarStore` + `Fn(&Tensor)->Tensor` |
| C++ libtorch | `TorchModel(module, forward_fn, …)` | `shared_ptr<nn::Module>` + `std::function` forward |

Each also keeps an MLP convenience builder (`JaxBackend.mlp`, `CandleMlp::new`,
`TchMlp::new`, `TorchMlp::make`) used by the smoke tests. Prunable params are the
weight tensors of rank ≥ 2 (kernels), skipping 1-D biases/norms.

## Notes

- **Rust/candle** is the intended path for ternary / BitNet b1.58 LLMs: the
  contract's `Criterion::Gate` and per-row quantizer hooks plug in here.
- **KAN** (the gate-pruned reference model in `../kan_lottery/`) is a deferred
  backend whose `criterion` is `gate` rather than `magnitude`; the engine needs
  no change to accept it — the proof the contract is right.
- One-shot pruning (no rewind) is `RewindPolicy::None`; it's the only mode an
  inference-only runtime (llama.cpp/GGUF/ONNX) could support, and is out of scope.
