from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..core.protocol import Criterion, PrunableModel


class JaxBackend(PrunableModel):
    """Generic JAX wrapper.

    apply_fn(params, X) -> logits.
    prunable(name, leaf) -> bool selects prunable leaves (default ndim >= 2).
    loss_fn(apply_fn, params, xb, yb) -> scalar overrides default CE/MSE (generative).
    eval_fn(apply_fn, params, data) -> float overrides default accuracy/MSE metric
    (higher better; e.g. -FID/-FAD).
    """

    def __init__(
        self,
        params,
        apply_fn,
        task="classification",
        lr=5e-3,
        batch_size=32,
        loss_fn=None,
        eval_fn=None,
        prunable=None,
        seed=0,
    ):
        self.params = params
        self.apply_fn = apply_fn
        self.task = task
        self.lr = lr
        self.batch_size = batch_size
        self.eval_fn = eval_fn
        self.masks = {}
        self._last_batch = None
        self._prunable = prunable if prunable is not None else (
            lambda name, leaf: jnp.ndim(leaf) >= 2
        )
        self._user_loss = loss_fn

    def _map_named(self, params, fn):
        leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(params)
        new_leaves = [fn(jax.tree_util.keystr(kp), leaf) for kp, leaf in leaves_with_path]
        return jax.tree_util.tree_unflatten(treedef, new_leaves)

    def _named_leaves(self, params):
        return [(jax.tree_util.keystr(kp), leaf) for kp, leaf in jax.tree_util.tree_flatten_with_path(params)[0]]

    def _forward(self, params, X):
        return self.apply_fn(params, jnp.asarray(X, jnp.float32))

    def _loss(self, params, xb, yb):
        if self._user_loss is not None:
            return self._user_loss(self.apply_fn, params, xb, yb)
        logits = self._forward(params, xb)
        if self.task == "classification":
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, yb))
        return jnp.mean((logits.reshape(-1) - yb.reshape(-1)) ** 2)

    def parameter_groups(self):
        return {name: leaf for name, leaf in self._named_leaves(self.params) if self._prunable(name, leaf)}

    def scores(self, name: str, criterion: Criterion) -> np.ndarray:
        groups = self.parameter_groups()

        if criterion == Criterion.MAGNITUDE:
            return np.abs(np.asarray(groups[name])).reshape(-1)

        if criterion == Criterion.SNIP:
            if self._last_batch is None:
                raise RuntimeError("SNIP requires cached last-batch data from fit")
            xb, yb = self._last_batch
            grads = jax.jit(jax.grad(self._loss))(self.params, xb, yb)
            grad_leaf = None
            for grad_name, leaf in self._named_leaves(grads):
                if grad_name == name:
                    grad_leaf = leaf
                    break
            if grad_leaf is None:
                raise KeyError(name)
            return np.abs(np.asarray(groups[name] * grad_leaf)).reshape(-1)

        if criterion == Criterion.GATE:
            raise RuntimeError("no gate buffer")

        if criterion == Criterion.RANDOM:
            raise ValueError("RANDOM is handled by the engine")

        raise ValueError(criterion)

    def apply_mask(self, name: str, mask: np.ndarray) -> None:
        leaf = self.parameter_groups()[name]
        m = jnp.asarray(np.asarray(mask, bool).reshape(leaf.shape), jnp.float32)
        self.masks[name] = m
        self.params = self._map_named(
            self.params,
            lambda n, l: l * self.masks[n] if n in self.masks else l,
        )

    def _reapply_masks(self):
        self.params = self._map_named(
            self.params,
            lambda n, l: l * self.masks[n] if n in self.masks else l,
        )

    def snapshot(self):
        return jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), self.params)

    def restore(self, state):
        self.params = jax.tree_util.tree_map(jnp.asarray, state)
        self._reapply_masks()

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
                params = self._map_named(
                    params,
                    lambda nm, l: l * self.masks[nm] if nm in self.masks else l,
                )

        self.params = params

    def evaluate(self, data) -> float:
        if self.eval_fn is not None:
            return float(self.eval_fn(self.apply_fn, self.params, data))

        X, y = data
        logits = self._forward(self.params, X)
        if self.task == "classification":
            preds = np.asarray(jnp.argmax(logits, axis=1))
            return float(np.mean(preds == np.asarray(y).reshape(-1)))
        return float(-np.mean((np.asarray(logits).reshape(-1) - np.asarray(y).reshape(-1)) ** 2))

    @classmethod
    def mlp(cls, dims, task="classification", lr=5e-3, batch_size=32, seed=0, **kw):
        params = {}
        key = jax.random.PRNGKey(seed)
        nlayers = len(dims) - 1
        for i in range(nlayers):
            key, subkey = jax.random.split(key)
            fan_in, fan_out = dims[i], dims[i + 1]
            limit = jnp.sqrt(6.0 / (fan_in + fan_out))
            params[f"l{i}.kernel"] = jax.random.uniform(
                subkey,
                (fan_in, fan_out),
                minval=-limit,
                maxval=limit,
            )
            params[f"l{i}.bias"] = jnp.zeros((fan_out,), dtype=jnp.float32)

        def apply_fn(params, X):
            h = jnp.asarray(X, jnp.float32)
            for i in range(nlayers):
                h = h @ params[f"l{i}.kernel"] + params[f"l{i}.bias"]
                if i < nlayers - 1:
                    h = jax.nn.relu(h)
            return h

        return cls(
            params,
            apply_fn,
            task=task,
            lr=lr,
            batch_size=batch_size,
            seed=seed,
            **kw,
        )


__all__ = ["JaxBackend"]
