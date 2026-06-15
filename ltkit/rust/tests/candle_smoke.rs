// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Michael King — part of ltkit (https://github.com/BTFProdigy/ltkit)
// Licensed under the Apache License, Version 2.0. See the LICENSE and NOTICE
// files in the project root. Attribution must be retained in derivative works.
//! Candle backend end-to-end test: train an MLP, run IMP, check the engine
//! invariants hold over a real tensor backend. Compiled only with `--features candle`.
#![cfg(feature = "candle")]

use candle_core::{Device, Tensor};

use ltkit::backends::candle::CandleMlp;
use ltkit::contract::{Criterion, PrunableModel, RewindPolicy};
use ltkit::imp::{run_imp, ImpConfig, Scope};

fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f32) / ((1u64 << 31) as f32) * 2.0 - 1.0 // ~[-1,1]
}

/// Linearly-separable classification: y = argmax(x · W), shared W across splits.
fn make_split(w: &[f32], d: usize, k: usize, n: usize, seed: u64) -> (Vec<f32>, Vec<u32>) {
    let mut s = seed;
    let mut x = Vec::with_capacity(n * d);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let row: Vec<f32> = (0..d).map(|_| lcg(&mut s)).collect();
        let mut best = (0u32, f32::NEG_INFINITY);
        for j in 0..k {
            let logit: f32 = (0..d).map(|i| row[i] * w[i * k + j]).sum();
            if logit > best.1 {
                best = (j as u32, logit);
            }
        }
        x.extend_from_slice(&row);
        y.push(best.0);
    }
    (x, y)
}

#[test]
fn candle_imp_smoke() {
    let dev = Device::Cpu;
    let (d, k, n) = (16usize, 3usize, 256usize);
    let mut ws = 42u64;
    let w: Vec<f32> = (0..d * k).map(|_| lcg(&mut ws)).collect();

    let (xt, yt) = make_split(&w, d, k, n, 1);
    let (xv, yv) = make_split(&w, d, k, n, 7);
    let to = |v: &[f32], rows: usize| Tensor::from_vec(v.to_vec(), (rows, d), &dev).unwrap();
    let toy = |v: &[u32]| Tensor::from_vec(v.to_vec(), (v.len(),), &dev).unwrap();

    let mut model = CandleMlp::new(
        &[d, 32, k],
        (to(&xt, n), toy(&yt)),
        (to(&xv, n), toy(&yv)),
        0.05,
        dev.clone(),
    )
    .unwrap();

    let cfg = ImpConfig {
        rounds: 4,
        prune_rate: 0.3,
        epochs_per: 40,
        criterion: Criterion::Magnitude,
        rewind: RewindPolicy::Init,
        scope: Scope::Global,
        early_k_epochs: 0,
        verbose: false,
    };
    let res = run_imp(&mut model, &cfg);

    let sp: Vec<f32> = res.history.iter().map(|r| r.sparsity).collect();
    assert_eq!(sp[0], 0.0, "{sp:?}");
    for win in sp.windows(2) {
        assert!(win[1] >= win[0], "non-monotonic sparsity {sp:?}");
    }
    assert!(*sp.last().unwrap() > 0.5, "final sparsity too low {sp:?}");

    // mask-persistence: pruned weights read back as exactly zero
    for (name, mask) in res.masks.iter() {
        let scores = model.scores(name, Criterion::Magnitude);
        for (i, &keep) in mask.iter().enumerate() {
            if !keep {
                assert_eq!(scores[i], 0.0, "pruned {name}[{i}] not zero");
            }
        }
    }

    // winning ticket still beats chance (0.33 for k=3)
    let acc = res.history.last().unwrap().metric;
    assert!(acc > 0.45, "ticket accuracy {acc} not above chance; {sp:?}");
}