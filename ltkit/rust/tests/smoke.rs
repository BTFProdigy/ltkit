// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Michael King — part of ltkit (https://github.com/BTFProdigy/ltkit)
// Licensed under the Apache License, Version 2.0. See the LICENSE and NOTICE
// files in the project root. Attribution must be retained in derivative works.
//! Engine smoke test over a pure in-memory mock model — no tensor backend.
//!
//! Mirrors the Python `test_smoke.py` invariant: sparsity rises monotonically,
//! the final returned masks agree with the model's zeroed weights, and a pruned
//! element stays zero across `fit`/`restore` (mask-persistence invariant).

use std::collections::HashMap;

use ltkit::contract::{Criterion, PrunableModel, RewindPolicy};
use ltkit::imp::{run_imp, ImpConfig, Scope};

struct MockModel {
    weights: HashMap<String, Vec<f32>>,
    masks: HashMap<String, Vec<bool>>,
}

impl MockModel {
    fn new() -> Self {
        let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
        // deterministic, distinct magnitudes so pruning order is well-defined
        weights.insert("a.weight".into(), (1..=20).map(|i| i as f32).collect());
        weights.insert("b.weight".into(), (1..=12).map(|i| (i as f32) * 0.5).collect());
        let masks = weights
            .iter()
            .map(|(k, v)| (k.clone(), vec![true; v.len()]))
            .collect();
        MockModel { weights, masks }
    }

    fn enforce(&mut self) {
        for (name, w) in self.weights.iter_mut() {
            if let Some(m) = self.masks.get(name) {
                for (wi, &keep) in w.iter_mut().zip(m.iter()) {
                    if !keep {
                        *wi = 0.0;
                    }
                }
            }
        }
    }
}

impl PrunableModel for MockModel {
    type State = HashMap<String, Vec<f32>>;

    fn parameter_groups(&self) -> Vec<String> {
        let mut g: Vec<String> = self.weights.keys().cloned().collect();
        g.sort();
        g
    }

    fn scores(&self, name: &str, _c: Criterion) -> Vec<f32> {
        self.weights[name].iter().map(|w| w.abs()).collect()
    }

    fn apply_mask(&mut self, name: &str, mask: &[bool]) {
        self.masks.insert(name.to_string(), mask.to_vec());
        self.enforce();
    }

    fn snapshot(&self) -> Self::State {
        self.weights.clone()
    }

    fn restore(&mut self, state: &Self::State) {
        self.weights = state.clone();
        self.enforce(); // masks survive restore
    }

    fn fit(&mut self, _epochs: usize) {
        self.enforce(); // training must not revive pruned weights
    }

    fn evaluate(&self) -> f32 {
        // fraction of surviving weight mass — purely to have a metric
        let total: f32 = self.weights.values().flatten().map(|w| w.abs()).sum();
        total / 100.0
    }
}

#[test]
fn imp_engine_invariants() {
    let mut model = MockModel::new();
    let cfg = ImpConfig {
        rounds: 4,
        prune_rate: 0.3,
        epochs_per: 1,
        criterion: Criterion::Magnitude,
        rewind: RewindPolicy::Init,
        scope: Scope::Global,
        early_k_epochs: 0,
        verbose: false,
    };
    let res = run_imp(&mut model, &cfg);

    // sparsity rises monotonically, starts at 0
    let sp: Vec<f32> = res.history.iter().map(|r| r.sparsity).collect();
    assert_eq!(sp[0], 0.0, "{sp:?}");
    for w in sp.windows(2) {
        assert!(w[1] >= w[0], "non-monotonic sparsity {sp:?}");
    }
    assert!(*sp.last().unwrap() > 0.5, "final sparsity too low {sp:?}");

    // final masks agree with zeroed weights (persistence across whole run)
    for (name, mask) in res.masks.iter() {
        let w = &model.weights[name];
        for (i, &keep) in mask.iter().enumerate() {
            if !keep {
                assert_eq!(w[i], 0.0, "pruned weight {name}[{i}] not zero");
            }
        }
    }
}