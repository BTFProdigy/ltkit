#![allow(dead_code)]

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{
    linear, loss, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap,
};

use crate::contract::{Criterion, PrunableModel};

pub type ForwardFn = Box<dyn Fn(&Tensor) -> Result<Tensor>>;
pub type EvalFn = Box<dyn Fn(&Tensor, &Tensor) -> Result<f32>>;

pub struct CandleModel {
    varmap: VarMap,
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

impl CandleModel {
    pub fn new(
        varmap: VarMap,
        forward_fn: ForwardFn,
        train: (Tensor, Tensor),
        val: (Tensor, Tensor),
        lr: f64,
        device: Device,
        eval_fn: Option<EvalFn>,
    ) -> Result<Self> {
        Ok(Self {
            varmap,
            forward_fn,
            masks: HashMap::new(),
            x: train.0.to_device(&device)?.to_dtype(DType::F32)?,
            y: train.1.to_device(&device)?.to_dtype(DType::U32)?.flatten_all()?,
            xv: val.0.to_device(&device)?.to_dtype(DType::F32)?,
            yv: val.1.to_device(&device)?.to_dtype(DType::U32)?.flatten_all()?,
            lr,
            eval_fn,
            device,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        (self.forward_fn)(x)
    }

    fn parameter_groups_result(&self) -> Result<Vec<String>> {
        let data = self.varmap.data().lock().unwrap();
        let mut names = data
            .iter()
            .filter(|(name, var)| name.contains(".weight") && var.as_tensor().rank() >= 2)
            .map(|(name, _)| name.clone())
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
        assert_eq!(mask.len(), weight.elem_count());
        let mask_t = Tensor::from_slice(
            &mask
                .iter()
                .map(|&b| if b { 1f32 } else { 0f32 })
                .collect::<Vec<f32>>(),
            weight.shape().clone(),
            &self.device,
        )?;
        let masked = (weight * &mask_t)?;
        var.set(&masked)?;
        self.masks.insert(name.to_string(), mask_t);
        Ok(())
    }

    fn snapshot_result(&self) -> Result<HashMap<String, Tensor>> {
        let data = self.varmap.data().lock().unwrap();
        data.iter()
            .map(|(name, var)| Ok((name.clone(), var.as_tensor().copy()?)))
            .collect()
    }

    fn reapply_masks_result(&self) -> Result<()> {
        let data = self.varmap.data().lock().unwrap();
        for (name, mask) in &self.masks {
            if let Some(var) = data.get(name) {
                let masked = (var.as_tensor() * mask)?;
                var.set(&masked)?;
            }
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
        self.reapply_masks_result()
    }

    fn fit_result(&mut self, epochs: usize) -> Result<()> {
        let mut opt = AdamW::new(
            self.varmap.all_vars(),
            ParamsAdamW {
                lr: self.lr,
                ..Default::default()
            },
        )?;
        for _ in 0..epochs {
            let logits = self.forward(&self.x)?;
            let loss = loss::cross_entropy(&logits, &self.y)?;
            opt.backward_step(&loss)?;
            self.reapply_masks_result()?;
        }
        Ok(())
    }

    fn evaluate_result(&self) -> Result<f32> {
        if self.yv.elem_count() == 0 {
            return Ok(0.0);
        }
        let out = self.forward(&self.xv)?;
        if let Some(ef) = &self.eval_fn {
            return ef(&out, &self.yv);
        }
        let preds = out.argmax(candle_core::D::Minus1)?.flatten_all()?;
        Ok(preds
            .eq(&self.yv)?
            .to_dtype(DType::F32)?
            .mean_all()?
            .to_scalar::<f32>()?)
    }
}

impl PrunableModel for CandleModel {
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
        self.restore_result(state)
            .expect("candle restore failed")
    }

    fn fit(&mut self, epochs: usize) {
        self.fit_result(epochs).expect("candle fit failed")
    }

    fn evaluate(&self) -> f32 {
        self.evaluate_result()
            .expect("candle evaluate failed")
    }
}

pub struct CandleMlp;

impl CandleMlp {
    pub fn new(
        dims: &[usize],
        train: (Tensor, Tensor),
        val: (Tensor, Tensor),
        lr: f64,
        device: Device,
    ) -> Result<CandleModel> {
        if dims.len() < 2 {
            candle_core::bail!("dims must have at least input and output sizes");
        }
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut layers: Vec<Linear> = Vec::with_capacity(dims.len() - 1);
        for i in 0..dims.len() - 1 {
            layers.push(linear(dims[i], dims[i + 1], vb.pp(format!("l{i}")))?);
        }
        let n = layers.len();
        let forward_fn: ForwardFn = Box::new(move |x: &Tensor| -> Result<Tensor> {
            let mut h = x.clone();
            for (i, layer) in layers.iter().enumerate() {
                h = layer.forward(&h)?;
                if i + 1 != n {
                    h = h.relu()?;
                }
            }
            Ok(h)
        });
        CandleModel::new(varmap, forward_fn, train, val, lr, device, None)
    }
}
