from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from ..core.protocol import Criterion, PrunableModel
class TorchBackend(PrunableModel):
    def __init__(
        self,
        model: nn.Module,
        task: str = "classification",
        lr: float = 5e-3,
        batch_size: int = 32,
        prunable_types=(nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d),
        device=None,
        loss_fn=None,
        eval_fn=None,
    ):
        """loss_fn(model, x, y) -> scalar tensor overrides the default CE/MSE training loss (for generative/diffusion/autoencoder models); eval_fn(model, data) -> float overrides the default accuracy/MSE metric (e.g. return -FID for images, -FAD for audio; higher is better)."""
        self.model = model
        self.task = task
        self.lr = lr
        self.batch_size = batch_size
        self.prunable_types = prunable_types
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        params = list(model.parameters())
        self.device = torch.device(device if device is not None else (params[0].device if params else "cpu"))
        self.model.to(self.device)
        self.mask_buffers = {}
        self._mask_hooks = {}
        self._last_batch = None
    def _criterion_name(self, criterion):
        return criterion.value if isinstance(criterion, Criterion) else str(criterion).lower()
    def _named_modules(self):
        return dict(self.model.named_modules())
    def _group_parts(self, name):
        module_name, _, param_name = name.rpartition(".")
        if not module_name:
            raise KeyError(name)
        return self._named_modules()[module_name], param_name
    def parameter_groups(self):
        groups = {}
        for module_name, module in self.model.named_modules():
            if module_name and isinstance(module, self.prunable_types) and getattr(module, "weight", None) is not None:
                groups[f"{module_name}.weight"] = module.weight
        return groups
    def _batch_iter(self, data):
        if isinstance(data, (tuple, list)) and len(data) == 2 and torch.is_tensor(data[0]) and torch.is_tensor(data[1]):
            x, y = data
            for i in range(0, x.shape[0], self.batch_size):
                yield x[i : i + self.batch_size], y[i : i + self.batch_size]
            return
        yield from data
    def _move_batch(self, batch):
        x, y = batch[:2] if isinstance(batch, (tuple, list)) else batch
        return x.to(self.device), y.to(self.device)
    def _apply_current_mask(self, name):
        mask = self.mask_buffers.get(name)
        if mask is None:
            return
        module, param_name = self._group_parts(name)
        with torch.no_grad():
            getattr(module, param_name).mul_(mask)
    def _ensure_snip_grads(self):
        if self._last_batch is None:
            raise RuntimeError("SNIP requires cached last-batch data from fit")
        if all(p.grad is not None for p in self.model.parameters() if p.requires_grad):
            return
        was_training = self.model.training
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        x, y = self._last_batch
        out = self.model(x)
        loss = nn.CrossEntropyLoss()(out, y.long().view(-1)) if self.task == "classification" else nn.MSELoss()(out, y.float())
        loss.backward()
        self.model.train(was_training)
    def scores(self, name: str, criterion: Criterion) -> np.ndarray:
        kind = self._criterion_name(criterion)
        param = self.parameter_groups()[name]
        if kind == Criterion.MAGNITUDE.value:
            return param.detach().abs().reshape(-1).cpu().numpy()
        if kind == Criterion.SNIP.value:
            self._ensure_snip_grads()
            grad = param.grad
            if grad is None:
                raise RuntimeError("SNIP gradients unavailable")
            return (param.detach() * grad.detach()).abs().reshape(-1).cpu().numpy()
        if kind == Criterion.GATE.value:
            module, param_name = self._group_parts(name)
            gate = getattr(module, f"{param_name}_gate", None)
            if gate is None:
                gate = self.model._buffers.get(f"{name.replace('.', '_')}_gate")
            if gate is None or not torch.is_tensor(gate):
                raise RuntimeError(f"missing gate buffer for {name}")
            return gate.detach().reshape(-1).cpu().numpy()
        raise ValueError("RANDOM is handled by the engine")
    def apply_mask(self, name: str, mask: np.ndarray) -> None:
        param = self.parameter_groups()[name]
        module, param_name = self._group_parts(name)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=param.device)
        if mask_t.numel() != param.numel():
            raise ValueError(f"mask size mismatch for {name}")
        mask_t = mask_t.reshape(param.shape)
        if name in self.mask_buffers:
            self.mask_buffers[name].copy_(mask_t)
        else:
            self.mask_buffers[name] = mask_t.clone()
            module.register_buffer(f"{param_name}_mask", self.mask_buffers[name], persistent=True)
            self._mask_hooks[name] = param.register_hook(lambda grad, m=self.mask_buffers[name]: grad * m)
        self._apply_current_mask(name)
    def snapshot(self) -> object:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
    def restore(self, state: object) -> None:
        masks = {name: mask.detach().clone() for name, mask in self.mask_buffers.items()}
        self.model.load_state_dict(state, strict=False)
        for name, mask in masks.items():
            self.mask_buffers[name].copy_(mask)
            self._apply_current_mask(name)
    def fit(self, data, epochs: int) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            return
        opt = torch.optim.Adam(params, lr=self.lr)
        loss_fn = nn.CrossEntropyLoss() if self.task == "classification" else nn.MSELoss()
        self.model.train()
        for _ in range(epochs):
            for batch in self._batch_iter(data):
                x, y = self._move_batch(batch)
                self._last_batch = (x.detach(), y.detach())
                opt.zero_grad(set_to_none=True)
                out = self.model(x)
                if self.loss_fn is not None:
                    loss = self.loss_fn(self.model, x, y)
                else:
                    loss = loss_fn(out, y.long().view(-1)) if self.task == "classification" else loss_fn(out, y.float())
                loss.backward()
                opt.step()
                for name in self.mask_buffers:
                    self._apply_current_mask(name)
    def evaluate(self, data) -> float:
        was_training = self.model.training
        self.model.eval()
        if self.eval_fn is not None:
            metric = float(self.eval_fn(self.model, data))
            self.model.train(was_training)
            return metric
        if self.task == "classification":
            correct = total = 0
            with torch.no_grad():
                for batch in self._batch_iter(data):
                    x, y = self._move_batch(batch)
                    pred = self.model(x).argmax(dim=1)
                    target = y.long().view(-1)
                    correct += (pred == target).sum().item()
                    total += target.numel()
            self.model.train(was_training)
            return 0.0 if total == 0 else correct / total
        mse = count = 0.0
        with torch.no_grad():
            for batch in self._batch_iter(data):
                x, y = self._move_batch(batch)
                diff = self.model(x) - y.float()
                mse += diff.pow(2).sum().item()
                count += diff.numel()
        self.model.train(was_training)
        return 0.0 if count == 0 else -(mse / count)
__all__ = ["TorchBackend"]
