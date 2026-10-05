"""AE-MLP：MLP 自编码器，重构误差作异常分数（numpy 手写反向传播）。"""
from __future__ import annotations

import numpy as np

from . import BaseModel, register


@register
class AEMLP(BaseModel):
    name = "ae_mlp"
    # 输入 w*f -> h1 -> bottleneck -> h1' -> 重构
    def __init__(self, h1=64, z=16):
        self.h1, self.z = h1, z

    def _init(self, d, seed):
        rng = np.random.default_rng(seed)
        s = lambda a, b: rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(np.float32)
        self.W1, self.b1 = s(d, self.h1), np.zeros(self.h1, np.float32)
        self.W2, self.b2 = s(self.h1, self.z), np.zeros(self.z, np.float32)
        self.W3, self.b3 = s(self.z, self.h1), np.zeros(self.h1, np.float32)
        self.W4, self.b4 = s(self.h1, d), np.zeros(d, np.float32)

    def _forward(self, X):
        X2 = X.reshape(len(X), -1)
        a1 = np.tanh(X2 @ self.W1 + self.b1)
        a2 = np.tanh(a1 @ self.W2 + self.b2)
        a3 = np.tanh(a2 @ self.W3 + self.b3)
        out = a3 @ self.W4 + self.b4
        return X2, a1, a2, a3, out

    def fit(self, train_w, val_w, patience=10, max_epochs=50, seed=0, lr=1e-3, batch=64):
        super().fit(train_w, val_w, patience, max_epochs, seed)
        d = train_w.shape[1] * train_w.shape[2]
        self._init(d, seed)
        rng = np.random.default_rng(seed)
        P = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4]
        m = [np.zeros_like(p) for p in P]
        v = [np.zeros_like(p) for p in P]
        best, wait, best_P = np.inf, 0, [p.copy() for p in P]
        t = 0
        for epoch in range(max_epochs):
            for xb in self._batches(train_w, batch, rng):
                t += 1
                X2, a1, a2, a3, out = self._forward(xb)
                dOut = 2.0 * (out - X2) / (len(xb) * d)
                gW4 = a3.T @ dOut; gb4 = dOut.sum(0)
                da3 = dOut @ self.W4.T * (1 - a3**2)
                gW3 = a2.T @ da3; gb3 = da3.sum(0)
                da2 = da3 @ self.W3.T * (1 - a2**2)
                gW2 = a1.T @ da2; gb2 = da2.sum(0)
                da1 = da2 @ self.W2.T * (1 - a1**2)
                gW1 = X2.T @ da1; gb1 = da1.sum(0)
                self._adam([gW1, gb1, gW2, gb2, gW3, gb3, gW4, gb4], P, m, v, t, lr)
            val_loss = float(np.mean((self._forward(val_w)[4] - val_w.reshape(len(val_w), -1)) ** 2))
            wait, best, best_P = (
                (0, val_loss, [p.copy() for p in P]) if val_loss < best - 1e-6
                else (wait + 1, best, best_P)
            )
            self._epochs_run = epoch + 1
            if wait >= patience:
                break
        for p, bp in zip(P, best_P):
            p[...] = bp
        return self

    def score_windows(self, W):
        X2, _, _, _, out = self._forward(W)
        return np.mean((out - X2) ** 2, axis=1)

    def count_params(self):
        return int(sum(p.size for p in [self.W1, self.b1, self.W2, self.b2,
                                        self.W3, self.b3, self.W4, self.b4]))
