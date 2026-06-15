// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Michael King — part of ltkit (https://github.com/BTFProdigy/ltkit)
// Licensed under the Apache License, Version 2.0. See the LICENSE and NOTICE
// files in the project root. Attribution must be retained in derivative works.
use crate::contract::{Criterion, PrunableModel, RewindPolicy};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scope {
    Global,
    PerGroup,
}

#[derive(Clone, Debug)]
pub struct ImpConfig {
    pub rounds: usize,
    pub prune_rate: f32,
    pub epochs_per: usize,
    pub criterion: Criterion,
    pub rewind: RewindPolicy,
    pub scope: Scope,
    pub early_k_epochs: usize,
    pub verbose: bool,
}

impl Default for ImpConfig {
    fn default() -> Self {
        Self {
            rounds: 4,
            prune_rate: 0.20,
            epochs_per: 100,
            criterion: Criterion::Magnitude,
            rewind: RewindPolicy::Init,
            scope: Scope::Global,
            early_k_epochs: 0,
            verbose: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RoundRecord {
    pub round: usize,
    pub keep: f32,
    pub metric: f32,
    pub sparsity: f32,
}

#[derive(Clone, Debug)]
pub struct ImpResult {
    pub masks: HashMap<String, Vec<bool>>,
    pub history: Vec<RoundRecord>,
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let bits = (self.state >> 32) as u32;
        (bits as f32) / ((u32::MAX as f32) + 1.0)
    }
}

fn target_kept(keep: f32, count: usize) -> isize {
    let mut target = (f64::from(keep) * count as f64).floor() as isize;
    if keep > 0.0 && target == 0 && count > 0 {
        target = 1;
    }
    target
}

fn sparsity(masks: &HashMap<String, Vec<bool>>) -> f32 {
    let mut total = 0usize;
    let mut kept = 0usize;
    for mask in masks.values() {
        total += mask.len();
        kept += mask.iter().filter(|&&b| b).count();
    }
    if total == 0 {
        0.0
    } else {
        (total - kept) as f32 / total as f32
    }
}

fn group_scores<T: PrunableModel>(
    model: &T,
    name: &str,
    criterion: Criterion,
    len: usize,
    rng: &mut Lcg,
) -> Vec<f32> {
    match criterion {
        Criterion::Random => (0..len).map(|_| rng.next_f32()).collect(),
        _ => model.scores(name, criterion),
    }
}

fn initial_masks<T: PrunableModel>(
    model: &T,
    group_names: &[String],
    criterion: Criterion,
) -> HashMap<String, Vec<bool>> {
    let mut masks = HashMap::with_capacity(group_names.len());
    let score_criterion = if matches!(criterion, Criterion::Random) {
        Criterion::Magnitude
    } else {
        criterion
    };

    for name in group_names {
        let scores = model.scores(name, score_criterion);
        masks.insert(name.clone(), vec![true; scores.len()]);
    }

    masks
}

fn prune_per_group<T: PrunableModel>(
    model: &T,
    group_names: &[String],
    masks: &HashMap<String, Vec<bool>>,
    criterion: Criterion,
    keep: f32,
    rng: &mut Lcg,
) -> HashMap<String, Vec<bool>> {
    let mut next = HashMap::with_capacity(group_names.len());

    for name in group_names {
        let current = masks
            .get(name)
            .expect("missing mask for parameter group");
        let scores = group_scores(model, name, criterion, current.len(), rng);
        let kept_idx: Vec<usize> = current
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect();
        let kept_count = kept_idx.len();
        let target = target_kept(keep, kept_count);

        if target <= 0 {
            next.insert(name.clone(), vec![false; current.len()]);
            continue;
        }

        let target = target as usize;
        if target >= kept_count {
            next.insert(name.clone(), current.clone());
            continue;
        }

        let mut candidates: Vec<(f32, usize)> = kept_idx.iter().map(|&idx| (scores[idx], idx)).collect();
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

        let mut new_mask = vec![false; current.len()];
        for &(_, idx) in candidates.iter().take(target) {
            new_mask[idx] = true;
        }
        next.insert(name.clone(), new_mask);
    }

    next
}

fn prune_global<T: PrunableModel>(
    model: &T,
    group_names: &[String],
    masks: &HashMap<String, Vec<bool>>,
    criterion: Criterion,
    keep: f32,
    rng: &mut Lcg,
) -> HashMap<String, Vec<bool>> {
    let mut group_data: Vec<(String, Vec<bool>)> = Vec::with_capacity(group_names.len());
    let mut candidates: Vec<(f32, usize, usize)> = Vec::new();
    let mut total_kept = 0usize;

    for (group_pos, name) in group_names.iter().enumerate() {
        let current = masks
            .get(name)
            .expect("missing mask for parameter group")
            .clone();
        let scores = group_scores(model, name, criterion, current.len(), rng);
        let kept_idx: Vec<usize> = current
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect();

        total_kept += kept_idx.len();
        for &idx in &kept_idx {
            candidates.push((scores[idx], group_pos, idx));
        }

        group_data.push((name.clone(), current));
    }

    let target = target_kept(keep, total_kept);
    if target <= 0 {
        let mut next = HashMap::with_capacity(group_names.len());
        for (name, current) in group_data {
            next.insert(name, vec![false; current.len()]);
        }
        return next;
    }

    let target = target as usize;
    if target >= total_kept {
        let mut next = HashMap::with_capacity(group_names.len());
        for (name, current) in group_data {
            next.insert(name, current);
        }
        return next;
    }

    candidates.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });

    let mut next_masks: Vec<Vec<bool>> = group_data
        .iter()
        .map(|(_, current)| vec![false; current.len()])
        .collect();

    for &(_, group_pos, idx) in candidates.iter().take(target) {
        next_masks[group_pos][idx] = true;
    }

    let mut next = HashMap::with_capacity(group_names.len());
    for ((name, _), mask) in group_data.into_iter().zip(next_masks.into_iter()) {
        next.insert(name, mask);
    }
    next
}

pub fn run_imp<T: PrunableModel>(model: &mut T, config: &ImpConfig) -> ImpResult {
    let group_names = model.parameter_groups();
    let mut masks = initial_masks(model, &group_names, config.criterion);
    let init_state = model.snapshot();
    let rewind_state = if config.rewind == RewindPolicy::EarlyK
        && config.early_k_epochs > 0
        && config.rounds > 0
    {
        model.restore(&init_state);
        model.fit(config.early_k_epochs);
        model.snapshot()
    } else {
        init_state
    };

    let mut history = Vec::with_capacity(config.rounds);
    let mut keep = 1.0f32;
    let mut rng = Lcg::new(0x4d595df4d0f33173);

    for round in 0..config.rounds {
        if config.rewind != RewindPolicy::None {
            model.restore(&rewind_state);
        }

        for name in &group_names {
            let mask = masks
                .get(name)
                .expect("missing mask for parameter group");
            model.apply_mask(name, mask);
        }

        model.fit(config.epochs_per);
        let metric = model.evaluate();
        history.push(RoundRecord {
            round,
            keep,
            metric,
            sparsity: sparsity(&masks),
        });

        keep *= 1.0 - config.prune_rate;
        masks = match config.scope {
            Scope::Global => prune_global(
                model,
                &group_names,
                &masks,
                config.criterion,
                keep,
                &mut rng,
            ),
            Scope::PerGroup => prune_per_group(
                model,
                &group_names,
                &masks,
                config.criterion,
                keep,
                &mut rng,
            ),
        };
    }

    for name in &group_names {
        if let Some(mask) = masks.get(name) {
            model.apply_mask(name, mask);
        }
    }

    ImpResult { masks, history }
}
