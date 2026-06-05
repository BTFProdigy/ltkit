from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .protocol import PrunableModel, Criterion, RewindPolicy
@dataclass
class IMPConfig:
    rounds: int = 4
    prune_rate: float = 0.20
    epochs_per: int = 100
    criterion: Criterion = Criterion.MAGNITUDE
    rewind: RewindPolicy = RewindPolicy.INIT
    scope: str = "global"
    early_k_epochs: int = 0
    verbose: bool = False
@dataclass
class IMPResult:
    masks: dict[str, np.ndarray]
    history: list[dict]
def _flat_scores(model: PrunableModel, name: str, criterion: Criterion, mask: np.ndarray) -> np.ndarray:
    if criterion == Criterion.RANDOM:
        return np.random.random(mask.size)
    return np.asarray(model.scores(name, criterion)).reshape(-1)
def _initial_masks(model: PrunableModel, criterion: Criterion) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for name in model.parameter_groups():
        scores = np.asarray(model.scores(name, criterion if criterion != Criterion.RANDOM else Criterion.MAGNITUDE)).reshape(-1)
        masks[name] = np.ones(scores.size, dtype=bool)
    return masks
def _sparsity(masks: dict[str, np.ndarray]) -> float:
    total = 0
    kept = 0
    for mask in masks.values():
        total += mask.size
        kept += int(mask.sum())
    return 0.0 if total == 0 else float((total - kept) / total)
def _prune_per_group(model: PrunableModel, masks: dict[str, np.ndarray], criterion: Criterion, keep: float) -> dict[str, np.ndarray]:
    next_masks: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        scores = _flat_scores(model, name, criterion, mask)
        current = np.asarray(mask, dtype=bool).reshape(-1)
        kept_idx = np.flatnonzero(current)
        kept_count = kept_idx.size
        target = int(np.floor(keep * kept_count)) if kept_count else 0
        if keep > 0.0 and target == 0 and kept_count:
            target = 1
        if target <= 0:
            next_masks[name] = np.zeros_like(current, dtype=bool)
            continue
        if target >= kept_count:
            next_masks[name] = current.copy()
            continue
        kept_scores = scores[kept_idx]
        top_rel = np.argpartition(-kept_scores, target - 1)[:target]
        selected = kept_idx[top_rel]
        new_mask = np.zeros_like(current, dtype=bool)
        new_mask[selected] = True
        next_masks[name] = new_mask
    return next_masks
def _prune_global(model: PrunableModel, masks: dict[str, np.ndarray], criterion: Criterion, keep: float) -> dict[str, np.ndarray]:
    group_data: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    kept_scores: list[np.ndarray] = []
    for name, mask in masks.items():
        scores = _flat_scores(model, name, criterion, mask)
        current = np.asarray(mask, dtype=bool).reshape(-1)
        kept_idx = np.flatnonzero(current)
        group_data.append((name, current, scores, kept_idx))
        if kept_idx.size:
            kept_scores.append(scores[kept_idx])
    total_kept = sum(item[3].size for item in group_data)
    target = int(np.floor(keep * total_kept)) if total_kept else 0
    if keep > 0.0 and target == 0 and total_kept:
        target = 1
    if target <= 0:
        return {name: np.zeros_like(current, dtype=bool) for name, current, _, _ in group_data}
    if target >= total_kept:
        return {name: current.copy() for name, current, _, _ in group_data}
    flat_scores = np.concatenate(kept_scores)
    top_rel = np.argpartition(-flat_scores, target - 1)[:target]
    selected = np.zeros_like(flat_scores, dtype=bool)
    selected[top_rel] = True
    next_masks: dict[str, np.ndarray] = {}
    offset = 0
    for name, current, _, kept_idx in group_data:
        new_mask = np.zeros_like(current, dtype=bool)
        span = kept_idx.size
        if span:
            chosen = selected[offset : offset + span]
            new_mask[kept_idx[chosen]] = True
        next_masks[name] = new_mask
        offset += span
    return next_masks
def run_imp(model: PrunableModel, train_data, val_data, config: IMPConfig) -> IMPResult:
    if config.scope not in {"global", "per_group"}:
        raise ValueError("config.scope must be 'global' or 'per_group'")
    init_state = model.snapshot()
    masks = _initial_masks(model, config.criterion)
    rewind_state = init_state
    if config.rewind == RewindPolicy.EARLY_K and config.early_k_epochs > 0 and config.rounds > 0:
        model.restore(init_state)
        model.fit(train_data, config.early_k_epochs)
        rewind_state = model.snapshot()
    history: list[dict] = []
    keep = 1.0
    for round_idx in range(config.rounds):
        if config.rewind != RewindPolicy.NONE:
            model.restore(rewind_state)
        for name, mask in masks.items():
            model.apply_mask(name, mask)
        model.fit(train_data, config.epochs_per)
        metric = model.evaluate(val_data)
        history.append({"round": round_idx, "keep": float(keep), "metric": float(metric), "sparsity": _sparsity(masks)})
        keep *= 1.0 - config.prune_rate
        if config.scope == "global":
            masks = _prune_global(model, masks, config.criterion, keep)
        else:
            masks = _prune_per_group(model, masks, config.criterion, keep)
    # Apply the final ticket mask so the returned model state agrees with the
    # returned masks (mask-persistence invariant across the whole run).
    for name, mask in masks.items():
        model.apply_mask(name, mask)
    return IMPResult(masks=masks, history=history)
