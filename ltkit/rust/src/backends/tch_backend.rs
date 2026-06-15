// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Michael King — part of ltkit (https://github.com/BTFProdigy/ltkit)
// Licensed under the Apache License, Version 2.0. See the LICENSE and NOTICE
// files in the project root. Attribution must be retained in derivative works.
#![allow(dead_code)]

use std::collections::HashMap;
use std::convert::TryFrom;

use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::contract::{Criterion, PrunableModel};

type ForwardFn = Box<dyn Fn(&Tensor) -> Tensor>;
type EvalFn = Box<dyn Fn(&Tensor, &Tensor) -> f32>;

pub struct TchModel {
    vs: nn::VarStore,
    forward_fn: ForwardFn,
    masks: HashMap<String, Tensor>,
    device: Device,
    x: Tensor,
    y: Tensor,
    xv: Tensor,
    yv: Tensor,
    lr: f64,
    eval_fn: Option<EvalFn>,
}

impl TchModel {
    pub fn new(
        vs: nn::VarStore,
        forward_fn: ForwardFn,
        train: (Tensor, Tensor),
        val: (Tensor, Tensor),
        lr: f64,
        device: Device,
        eval_fn: Option<EvalFn>,
    ) -> Self {
        Self {
            vs,
            forward_fn,
            masks: HashMap::new(),
            device,
            x: train.0.to_device(device).to_kind(Kind::Float),
            y: train.1.to_device(device).to_kind(Kind::Int64).flatten(0, -1),
            xv: val.0.to_device(device).to_kind(Kind::Float),
            yv: val.1.to_device(device).to_kind(Kind::Int64).flatten(0, -1),
            lr,
            eval_fn,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        (self.forward_fn)(x)
    }

    fn parameter_groups_result(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .vs
            .variables()
            .iter()
            .filter(|(n, w)| n.ends_with(".weight") && w.size().len() >= 2)
            .map(|(n, _)| n.clone())
            .collect();
        names.sort();
        names
    }

    fn scores_result(&self, name: &str, _criterion: Criterion) -> Vec<f32> {
        let vars = self.vs.variables();
        let w = vars.get(name).expect("missing param group");
        let flat = w.abs().flatten(0, -1).to_device(Device::Cpu);
        Vec::<f32>::try_from(flat).expect("to vec")
    }

    fn apply_mask_result(&mut self, name: &str, mask: &[bool]) {
        let vars = self.vs.variables();
        let w = vars.get(name).expect("missing").shallow_clone();
        assert_eq!(mask.len(), w.numel() as usize);
        let shape = w.size();
        let mask_data: Vec<f32> = mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let mask_t = Tensor::from_slice(&mask_data)
            .to_device(self.device)
            .reshape(&shape);
        tch::no_grad(|| {
            let mut w2 = w.shallow_clone();
            let _ = w2.g_mul_(&mask_t);
        });
        self.masks.insert(name.to_string(), mask_t);
    }

    fn reapply_masks(&self) {
        let vars = self.vs.variables();
        for (name, m) in &self.masks {
            if let Some(w) = vars.get(name) {
                tch::no_grad(|| {
                    let mut w2 = w.shallow_clone();
                    let _ = w2.g_mul_(m);
                });
            }
        }
    }

    fn snapshot_result(&self) -> HashMap<String, Tensor> {
        self.vs
            .variables()
            .iter()
            .map(|(k, v)| (k.clone(), v.detach().copy()))
            .collect()
    }

    fn restore_result(&mut self, state: &HashMap<String, Tensor>) {
        tch::no_grad(|| {
            let vars = self.vs.variables();
            for (name, t) in state {
                if let Some(w) = vars.get(name) {
                    let mut w2 = w.shallow_clone();
                    let _ = w2.copy_(t);
                }
            }
        });
        self.reapply_masks();
    }

    fn fit_result(&mut self, epochs: usize) {
        let mut opt = nn::Adam::default().build(&self.vs, self.lr).expect("opt");
        for _ in 0..epochs {
            let logits = self.forward(&self.x);
            let loss = logits.cross_entropy_for_logits(&self.y);
            opt.backward_step(&loss);
            self.reapply_masks();
        }
    }

    fn evaluate_result(&self) -> f32 {
        if self.yv.numel() == 0 {
            return 0.0;
        }
        let out = self.forward(&self.xv);
        if let Some(ef) = &self.eval_fn {
            return ef(&out, &self.yv);
        }
        let preds = out.argmax(-1, false);
        let acc = preds
            .eq_tensor(&self.yv)
            .to_kind(Kind::Float)
            .mean(Kind::Float);
        f32::try_from(acc).expect("acc")
    }
}

impl PrunableModel for TchModel {
    type State = HashMap<String, Tensor>;

    fn parameter_groups(&self) -> Vec<String> {
        self.parameter_groups_result()
    }

    fn scores(&self, name: &str, criterion: Criterion) -> Vec<f32> {
        self.scores_result(name, criterion)
    }

    fn apply_mask(&mut self, name: &str, mask: &[bool]) {
        self.apply_mask_result(name, mask)
    }

    fn snapshot(&self) -> Self::State {
        self.snapshot_result()
    }

    fn restore(&mut self, state: &Self::State) {
        self.restore_result(state)
    }

    fn fit(&mut self, epochs: usize) {
        self.fit_result(epochs)
    }

    fn evaluate(&self) -> f32 {
        self.evaluate_result()
    }
}

pub struct TchMlp;

impl TchMlp {
    pub fn new(
        dims: &[usize],
        train: (Tensor, Tensor),
        val: (Tensor, Tensor),
        lr: f64,
        device: Device,
    ) -> TchModel {
        assert!(dims.len() >= 2, "dims must have at least input and output sizes");
        let vs = nn::VarStore::new(device);
        let root = &vs.root();
        let mut layers = Vec::with_capacity(dims.len() - 1);
        for i in 0..dims.len() - 1 {
            layers.push(nn::linear(
                root / &format!("l{i}"),
                dims[i] as i64,
                dims[i + 1] as i64,
                Default::default(),
            ));
        }
        let n = layers.len();
        let forward_fn: ForwardFn = Box::new(move |x: &Tensor| {
            let mut h = x.shallow_clone();
            for (i, layer) in layers.iter().enumerate() {
                h = layer.forward(&h);
                if i + 1 != n {
                    h = h.relu();
                }
            }
            h
        });
        TchModel::new(vs, forward_fn, train, val, lr, device, None)
    }
}