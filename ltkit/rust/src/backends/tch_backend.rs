#![allow(dead_code)]

use std::collections::HashMap;
use std::convert::TryFrom;

use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::contract::{Criterion, PrunableModel};

pub struct TchMlp {
    vs: nn::VarStore,
    layers: Vec<nn::Linear>,
    masks: HashMap<String, Tensor>,
    device: Device,
    x: Tensor,
    y: Tensor,
    xv: Tensor,
    yv: Tensor,
    lr: f64,
}

impl TchMlp {
    pub fn new(
        dims: &[usize],
        train: (Tensor, Tensor),
        val: (Tensor, Tensor),
        lr: f64,
        device: Device,
    ) -> Self {
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
        Self {
            vs,
            layers,
            masks: HashMap::new(),
            device,
            x: train.0.to_device(device).to_kind(Kind::Float),
            y: train.1.to_device(device).to_kind(Kind::Int64).flatten(0, -1),
            xv: val.0.to_device(device).to_kind(Kind::Float),
            yv: val.1.to_device(device).to_kind(Kind::Int64).flatten(0, -1),
            lr,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = x.shallow_clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h);
            if i + 1 != self.layers.len() {
                h = h.relu();
            }
        }
        h
    }

    fn parameter_groups_result(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .vs
            .variables()
            .keys()
            .filter(|n| n.ends_with(".weight"))
            .cloned()
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
        assert_eq!(mask.len(), w.numel());
        let shape = w.size();
        let mask_t = Tensor::from_slice(
            &mask
                .iter()
                .map(|&b| if b { 1f32 } else { 0f32 })
                .collect::<Vec<f32>>(),
        )
        .to_device(self.device)
        .reshape(&shape);
        tch::no_grad(|| {
            let mut w = w.shallow_clone();
            let _ = w.g_mul_(&mask_t);
        });
        self.masks.insert(name.to_string(), mask_t);
    }

    fn reapply_masks(&self) {
        let vars = self.vs.variables();
        for (name, m) in &self.masks {
            if let Some(w) = vars.get(name) {
                tch::no_grad(|| {
                    let mut w = w.shallow_clone();
                    let _ = w.g_mul_(m);
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
                    let mut w = w.shallow_clone();
                    let _ = w.copy_(t);
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
        let logits = self.forward(&self.xv);
        let preds = logits.argmax(-1, false);
        let acc = preds
            .eq_tensor(&self.yv)
            .to_kind(Kind::Float)
            .mean(Kind::Float);
        f32::try_from(acc).expect("acc")
    }
}

impl PrunableModel for TchMlp {
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
