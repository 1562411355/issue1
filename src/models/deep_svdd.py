"""Deep SVDD：MLP 编码到低维空间，到超球中心的距离作异常分数。

Ruff et al., ICML 2018。中心 c 取初始前向输出的均值（固定）；无偏置项，
避免坍缩到中心（原文做法）。numpy 手写 Adam。
"""
from __future__ import annotations

import numpy as np

from . import BaseModel, register


@register
class DeepSVDD(BaseModel):
    name = "deep_svdd"

    def __init__(self, h1=64, z=8):
        self.h1, self.z = h1, z

    def _init(self, d, seed):
        rng = np.random.default_rng(seed)
        s = lambda a, b: rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(np.float32)
        self.W1 = s(d, self.h1)      # 无偏置（DeepSVDD 惯例）
        self.W2 = s(self.h1, self.z)

    def _encode(self, X):
        X2 = X.reshape(len(X), -1)
        a1 = np.tanh(X2 @ self.W1)
        return X2, a1, a1 @ self.W2  # z

    def fit(self, train_w, val_w, patience=10, max_epochs=50, seed=0, lr=1e-3, batch=64):
        super().fit(train_w, val_w, patience, max_epochs, seed)
        d = train_w.shape[1] * train_w.shape[2]
        self._init(d, seed)
        rng = np.random.default_rng(seed)
        # 中心 = 初始前向输出均值，并做最小值裁剪防坍缩
        _, _, z0 = self._encode(train_w)
        self.c = np.clip(z0.mean(axis=0), -0.1 + 1e-6, 0.1 - 1e-6).astype(np.float32)
        P = [self.W1, self.W2]
        m = [np.zeros_like(p) for p in P]
        v = [np.zeros_like(p) for p in P]
        best, wait, best_P = np.inf, 0, [p.copy() for p in P]
        t = 0
        for epoch in range(max_epochs):
            for xb in self._batches(train_w, batch, rng):
                t += 1
                X2, a1, z = self._encode(xb)
                r = z - self.c
                dZ = 2.0 * r / (len(xb) * z.shape[1])
                gW2 = a1.T @ dZ
                dA1 = dZ @ self.W2.T * (1 - a1**2)
                gW1 = X2.T @ dA1
                self._adam([gW1, gW2], P, m, v, t, lr)
            _, _, zv = self._encode(val_w)
            val_loss = float(np.mean((zv - self.c) ** 2))
            if val_loss < best - 1e-6:
                best, wait, best_P = val_loss, 0, [p.copy() for p in P]
            else:
                wait += 1
                if wait >= patience:
                    break
            self._epochs_run = epoch + 1
        for p, bp in zip(P, best_P):
            p[...] = bp
        return self

    def score_windows(self, W):
        _, _, z = self._encode(W)
        return np.mean((z - self.c) ** 2, axis=1)

    def count_params(self):
        return int(self.W1.size + self.W2.size + self.c.size)
