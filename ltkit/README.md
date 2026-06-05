# LTKit

A backend-agnostic framework for finding **lottery tickets** (Frankle & Carbin
iterative magnitude pruning with rewind) in trainable models — across PyTorch,
TensorFlow, JAX, Rust (candle), and C++ (libtorch).

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
| Python | — | keras/tf, jax | planned (code-only; not installed here) |
| Rust | ✅ | candle (BitNet target) | ✅ `cargo test --features candle` |
| Rust | — | tch-rs | planned (shares libtorch) |
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
    ltkit/backends/      torch_backend.py (+ keras/jax planned)
    tests/test_smoke.py
    pyproject.toml
  rust/
    src/contract.rs      trait PrunableModel + enums
    src/imp.rs           engine (generic over T: PrunableModel)
    src/backends/candle.rs
    tests/{smoke,candle_smoke}.rs
    Cargo.toml           candle behind an optional feature
  cpp/
    include/ltkit/ltkit.hpp          header-only engine (template-duck-typed model)
    include/ltkit/torch_backend.hpp  libtorch MLP backend
    tests/{smoke,torch_smoke}.cpp
    CMakeLists.txt
```

## Running

```bash
# Python
cd python && PYTHONPATH=$PWD python3 tests/test_smoke.py        # -> smoke OK

# Rust core + candle backend
cd rust && cargo test                                          # pure core
cargo test --features candle                                   # candle backend

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

## Notes

- **Rust/candle** is the intended path for ternary / BitNet b1.58 LLMs: the
  contract's `Criterion::Gate` and per-row quantizer hooks plug in here.
- **KAN** (the gate-pruned reference model in `../kan_lottery/`) is a deferred
  backend whose `criterion` is `gate` rather than `magnitude`; the engine needs
  no change to accept it — the proof the contract is right.
- One-shot pruning (no rewind) is `RewindPolicy::None`; it's the only mode an
  inference-only runtime (llama.cpp/GGUF/ONNX) could support, and is out of scope.
