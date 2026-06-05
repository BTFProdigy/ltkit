from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..core.protocol import Criterion, PrunableModel


class JaxBackend(PrunableModel):
    def __init__(self, dims, task="classification", lr=5e-3, batch_size=32, seed=0):
        self.task = task
        self.lr = lr
        self.batch_size = batch_size
        self.masks = {}
        self._last_batch = None
        self._nlayers = len(dims) - 1

        params = {}
        key = jax.random.PRNGKey(seed)
        for i in range(self._nlayers):
            key, k1, k2 = jax.random.split(key, 3)
            fan_in, fan_out = dims[i], dims[i + 1]
            limit = jnp.sqrt(6.0 / (fan_in + fan_out))
            params[f"l{i}.kernel"] = jax.random.uniform(
                k1, (fan_in, fan_out), minval=-limit, maxval=limit
            )
            params[f"l{i}.bias"] = jnp.zeros((fan_out,), dtype=jnp.float32)
        self.params = params

    def _forward(self, params, X):
        h = jnp.asarray(X, jnp.float32)
        for i in range(self._nlayers):
            h = h @ params[f"l{i}.kernel"] + params[f"l{i}.bias"]
            if i < self._nlayers - 1:
                h = jax.nn.relu(h)
        return h

    def parameter_groups(self):
        return {f"l{i}.kernel": self.params[f"l{i}.kernel"] for i in range(self._nlayers)}

    def scores(self, name: str, criterion: Criterion) -> np.ndarray:
        if criterion == Criterion.MAGNITUDE:
            return np.abs(np.asarray(self.params[name])).reshape(-1)

        if criterion == Criterion.SNIP:
            if self._last_batch is None:
                raise RuntimeError("SNIP requires cached last-batch data from fit")
            xb, yb = self._last_batch
            grad_fn = jax.jit(jax.grad(self._loss))
            grads = grad_fn(self.params, xb, yb)
            return np.abs(np.asarray(self.params[name] * grads[name])).reshape(-1)

        if criterion == Criterion.GATE:
            raise RuntimeError("no gate buffer")

        if criterion == Criterion.RANDOM:
            raise ValueError("RANDOM is handled by the engine")

        raise ValueError(criterion)

    def apply_mask(self, name: str, mask: np.ndarray) -> None:
        m = jnp.asarray(np.asarray(mask, bool).reshape(self.params[name].shape), jnp.float32)
        self.masks[name] = m
        self.params[name] = self.params[name] * m

    def _reapply_masks(self):
        for name, m in self.masks.items():
            self.params[name] = self.params[name] * m

    def snapshot(self):
        return {k: np.asarray(v).copy() for k, v in self.params.items()}

    def restore(self, state):
        self.params = {k: jnp.asarray(v) for k, v in state.items()}
        self._reapply_masks()

    def _loss(self, params, xb, yb):
        logits = self._forward(params, xb)
        if self.task == "classification":
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, yb))
        return jnp.mean((logits.reshape(-1) - yb.reshape(-1)) ** 2)

    def fit(self, data, epochs: int) -> None:
        X, y = data
        X = jnp.asarray(np.asarray(X), jnp.float32)
        if self.task == "classification":
            y = jnp.asarray(np.asarray(y), jnp.int32)
        else:
            y = jnp.asarray(np.asarray(y), jnp.float32)

        opt = optax.adam(self.lr)
        params = self.params
        opt_state = opt.init(params)
        grad_fn = jax.jit(jax.grad(self._loss))

        n = int(X.shape[0])
        for _ in range(epochs):
            for i in range(0, n, self.batch_size):
                xb = X[i : i + self.batch_size]
                yb = y[i : i + self.batch_size]
                self._last_batch = (xb, yb)
                grads = grad_fn(params, xb, yb)
                updates, opt_state = opt.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                for name, m in self.masks.items():
                    params[name] = params[name] * m

        self.params = params

    def evaluate(self, data) -> float:
        X, y = data
        logits = self._forward(self.params, jnp.asarray(np.asarray(X), jnp.float32))
        if self.task == "classification":
            preds = np.asarray(jnp.argmax(logits, axis=1))
            return float(np.mean(preds == np.asarray(y).reshape(-1)))
        return float(-np.mean((np.asarray(logits).reshape(-1) - np.asarray(y).reshape(-1)) ** 2))


__all__ = ["JaxBackend"]
