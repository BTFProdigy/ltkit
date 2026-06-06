from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf

from ..core.protocol import Criterion, PrunableModel


class KerasBackend(PrunableModel):
    def __init__(self, model, task="classification", lr=5e-3, batch_size=32,
                 loss_fn=None, eval_fn=None):
        # loss_fn(model, x, y) -> scalar tensor overrides the default CE/MSE
        # training loss (generative / diffusion / autoencoder models).
        # eval_fn(model, data) -> float overrides the default accuracy/MSE metric
        # (e.g. return -FID for images, -FAD for audio; higher is better).
        self.model = model
        self.task = task
        self.lr = lr
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        self.masks = {}
        self._last_batch = None

    def _dense_layers(self):
        return [l for l in self.model.layers if hasattr(l, "kernel")]

    def parameter_groups(self):
        return {f"{l.name}.kernel": l.kernel for l in self._dense_layers()}

    def _layer_for(self, name):
        for layer in self._dense_layers():
            if f"{layer.name}.kernel" == name:
                return layer
        raise KeyError(name)

    def _criterion_name(self, criterion):
        return criterion.value if isinstance(criterion, Criterion) else str(criterion).lower()

    def scores(self, name: str, criterion: Criterion) -> np.ndarray:
        kind = self._criterion_name(criterion)
        kernel = self.parameter_groups()[name]

        if kind == Criterion.MAGNITUDE.value:
            return np.abs(kernel.numpy()).reshape(-1)

        if kind == Criterion.SNIP.value:
            if self._last_batch is None:
                raise RuntimeError("SNIP requires cached last-batch data from fit")
            xb, yb = self._last_batch
            loss_fn = (
                keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                if self.task == "classification"
                else keras.losses.MeanSquaredError()
            )
            with tf.GradientTape() as tape:
                logits = self.model(xb, training=True)
                if self.task == "classification":
                    loss = loss_fn(yb, logits)
                else:
                    loss = loss_fn(tf.cast(yb, tf.float32), logits)
            grad = tape.gradient(loss, kernel)
            if grad is None:
                raise RuntimeError("SNIP gradients unavailable")
            return np.abs((kernel * grad).numpy()).reshape(-1)

        if kind == Criterion.GATE.value:
            layer = self._layer_for(name)
            gate = getattr(layer, f"{name}_gate", None)
            if gate is None:
                raise RuntimeError("no gate buffer")
            return np.asarray(gate.numpy() if hasattr(gate, "numpy") else gate).reshape(-1)

        raise ValueError("RANDOM is handled by the engine")

    def apply_mask(self, name: str, mask: np.ndarray) -> None:
        kernel = self.parameter_groups()[name]
        mask_tensor = tf.constant(
            np.asarray(mask, dtype=bool).reshape(tuple(kernel.shape)),
            dtype=tf.float32,
        )
        self.masks[name] = mask_tensor
        kernel.assign(kernel * mask_tensor)

    def _reapply_masks(self):
        for name, mask in self.masks.items():
            kernel = self.parameter_groups()[name]
            kernel.assign(kernel * mask)

    def snapshot(self):
        return [w.copy() for w in self.model.get_weights()]

    def restore(self, state):
        self.model.set_weights([w.copy() for w in state])
        self._reapply_masks()

    def _batches(self, data):
        X, y = data
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.int32)
        n = int(X.shape[0]) if X.shape[0] is not None else int(tf.shape(X)[0].numpy())
        for i in range(0, n, self.batch_size):
            yield X[i : i + self.batch_size], y[i : i + self.batch_size]

    def fit(self, data, epochs: int) -> None:
        optimizer = keras.optimizers.Adam(self.lr)
        loss_fn = (
            keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            if self.task == "classification"
            else keras.losses.MeanSquaredError()
        )
        for _ in range(epochs):
            for xb, yb in self._batches(data):
                self._last_batch = (xb, yb)
                with tf.GradientTape() as tape:
                    if self.loss_fn is not None:
                        loss = self.loss_fn(self.model, xb, yb)
                    else:
                        logits = self.model(xb, training=True)
                        if self.task == "classification":
                            loss = loss_fn(yb, logits)
                        else:
                            loss = loss_fn(tf.cast(yb, tf.float32), logits)
                grads = tape.gradient(loss, self.model.trainable_variables)
                pairs = [(g, v) for g, v in zip(grads, self.model.trainable_variables) if g is not None]
                if pairs:
                    optimizer.apply_gradients(pairs)
                self._reapply_masks()

    def evaluate(self, data) -> float:
        if self.eval_fn is not None:
            return float(self.eval_fn(self.model, data))
        X, y = data
        logits = self.model(tf.convert_to_tensor(X, tf.float32), training=False)
        if self.task == "classification":
            preds = tf.argmax(logits, axis=1)
            return float(np.mean(preds.numpy() == np.asarray(y).reshape(-1)))
        return float(-np.mean((logits.numpy().reshape(-1) - np.asarray(y).reshape(-1)) ** 2))


__all__ = ["KerasBackend"]
