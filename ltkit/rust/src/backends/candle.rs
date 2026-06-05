#![allow(dead_code)]

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, loss, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use crate::contract::{Criterion, PrunableModel};

// VarMap is not Debug/Clone, so CandleMlp can't derive them.
pub struct CandleMlp {
    varmap: VarMap,
    layers: Vec<Linear>,
    masks: HashMap<String, Tensor>,
    device: Device,
    x: Tensor,
    y: Tensor,
    xv: Tensor,
    yv: Tensor,
    lr: f64,
}

impl CandleMlp {
    pub fn new(
        dims: &[usize],
        train: (Tensor, Tensor),
        val: (Tensor, Tensor),
        lr: f64,
        device: Device,
    ) -> Result<Self> {
        if dims.len() < 2 {
            candle_core::bail!("dims must have at least input and output sizes");
        }
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut layers = Vec::with_capacity(dims.len() - 1);
        for i in 0..dims.len() - 1 {
            layers.push(linear(dims[i], dims[i + 1], vb.pp(format!("l{i}")))?);
        }
        Ok(Self {
            varmap,
            layers,
            masks: HashMap::new(),
            x: train.0.to_device(&device)?.to_dtype(DType::F32)?,
            y: train.1.to_device(&device)?.to_dtype(DType::U32)?.flatten_all()?,
            xv: val.0.to_device(&device)?.to_dtype(DType::F32)?,
            yv: val.1.to_device(&device)?.to_dtype(DType::U32)?.flatten_all()?,
            device,
            lr,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h)?;
            if i + 1 != self.layers.len() {
                h = h.relu()?;
            }
        }
        Ok(h)
    }

    fn parameter_groups_result(&self) -> Result<Vec<String>> {
        let data = self.varmap.data().lock().unwrap();
        let mut names = data
            .keys()
            .filter(|name| name.contains(".weight"))
            .cloned()
            .collect::<Vec<_>>();
        names.sort();
        Ok(names)
    }

    fn scores_result(&self, name: &str, _criterion: Criterion) -> Result<Vec<f32>> {
        let data = self.varmap.data().lock().unwrap();
        let var = data
            .get(name)
            .unwrap_or_else(|| panic!("missing parameter group {name}"));
        Ok(var.as_tensor().abs()?.flatten_all()?.to_vec1::<f32>()?)
    }

    fn apply_mask_result(&mut self, name: &str, mask: &[bool]) -> Result<()> {
        let var = {
            let data = self.varmap.data().lock().unwrap();
            data.get(name)
                .cloned()
                .unwrap_or_else(|| panic!("missing parameter group {name}"))
        };
        let weight = var.as_tensor();
        if mask.len() != weight.elem_count() {
            candle_core::bail!("mask size mismatch for {name}");
        }
        let mask_tensor = Tensor::from_vec(
            mask.iter().map(|&b| if b { 1f32 } else { 0f32 }).collect::<Vec<_>>(),
            weight.shape().clone(),
            &self.device,
        )?;
        var.set(&weight.mul(&mask_tensor)?)?;
        self.masks.insert(name.to_string(), mask_tensor);
        Ok(())
    }

    fn snapshot_result(&self) -> Result<HashMap<String, Tensor>> {
        let data = self.varmap.data().lock().unwrap();
        let mut state = HashMap::with_capacity(data.len());
        for (name, var) in data.iter() {
            state.insert(name.clone(), var.as_tensor().copy()?);
        }
        Ok(state)
    }

    fn reapply_masks(&self) -> Result<()> {
        for (name, mask) in &self.masks {
            let var = {
                let data = self.varmap.data().lock().unwrap();
                data.get(name)
                    .cloned()
                    .unwrap_or_else(|| panic!("missing parameter group {name}"))
            };
            var.set(&var.as_tensor().mul(mask)?)?;
        }
        Ok(())
    }

    fn restore_result(&mut self, state: &HashMap<String, Tensor>) -> Result<()> {
        {
            let data = self.varmap.data().lock().unwrap();
            for (name, tensor) in state {
                if let Some(var) = data.get(name) {
                    var.set(tensor)?;
                }
            }
        }
        self.reapply_masks()?;
        Ok(())
    }

    fn fit_result(&mut self, epochs: usize) -> Result<()> {
        let params = ParamsAdamW {
            lr: self.lr,
            ..Default::default()
        };
        let mut opt = AdamW::new(self.varmap.all_vars(), params)?;
        for _ in 0..epochs {
            let logits = self.forward(&self.x)?;
            let loss = loss::cross_entropy(&logits, &self.y)?;
            opt.backward_step(&loss)?;
            self.reapply_masks()?;
        }
        Ok(())
    }

    fn evaluate_result(&self) -> Result<f32> {
        if self.yv.elem_count() == 0 {
            return Ok(0.0);
        }
        let logits = self.forward(&self.xv)?;
        let preds = logits.argmax(candle_core::D::Minus1)?.flatten_all()?;
        Ok(preds
            .eq(&self.yv)?
            .to_dtype(DType::F32)?
            .mean_all()?
            .to_scalar::<f32>()?)
    }
}

impl PrunableModel for CandleMlp {
    type State = HashMap<String, Tensor>;

    fn parameter_groups(&self) -> Vec<String> {
        self.parameter_groups_result()
            .expect("candle parameter_groups failed")
    }

    fn scores(&self, name: &str, criterion: Criterion) -> Vec<f32> {
        self.scores_result(name, criterion)
            .expect("candle scores failed")
    }

    fn apply_mask(&mut self, name: &str, mask: &[bool]) {
        self.apply_mask_result(name, mask)
            .expect("candle apply_mask failed")
    }

    fn snapshot(&self) -> Self::State {
        self.snapshot_result().expect("candle snapshot failed")
    }

    fn restore(&mut self, state: &Self::State) {
        self.restore_result(state).expect("candle restore failed")
    }

    fn fit(&mut self, epochs: usize) {
        self.fit_result(epochs).expect("candle fit failed")
    }

    fn evaluate(&self) -> f32 {
        self.evaluate_result().expect("candle evaluate failed")
    }
}
