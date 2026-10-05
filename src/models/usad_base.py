"""USAD 适配（KDD 2019，manigalati/usad）。

出处：https://github.com/manigalati/usad
（usad.py 的 UsadModel / training / testing；alpha=beta=0.5 官方默认融合分。）

与官方差异（见 submission/deviations.md）：
- 窗口展平维度 = 协议 120×F；官方对窗口展平后编码
- 输入为管线 z-score（官方管道多为 min-max 到 [0,1]）
- 训练轮数 100（常见复现设置），batch=128，z=32
- max_train=15000 与 distill_ae 同口径
"""
from __future__ import annotations

import numpy as np

from . import BaseModel, register

MAX_TRAIN = 15000


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


class _USADCore:
    def __init__(self, w_size: int, z_size: int = 32):
        torch, nn = _torch()
        self._nn = nn
        self._torch = torch

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(w_size, int(w_size / 2))
                self.linear2 = nn.Linear(int(w_size / 2), int(w_size / 4))
                self.linear3 = nn.Linear(int(w_size / 4), z_size)
                self.relu = nn.ReLU(True)

            def forward(self, w):
                z = self.relu(self.linear3(self.relu(self.linear2(self.relu(self.linear1(w))))))
                return z

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(z_size, int(w_size / 4))
                self.linear2 = nn.Linear(int(w_size / 4), int(w_size / 2))
                self.linear3 = nn.Linear(int(w_size / 2), w_size)
                self.relu = nn.ReLU(True)
                self.sigmoid = nn.Sigmoid()

            def forward(self, z):
                return self.sigmoid(self.linear3(self.relu(self.linear2(self.relu(self.linear1(z))))))

        self.encoder = Encoder()
        self.decoder1 = Decoder()
        self.decoder2 = Decoder()

    def parameters(self):
        for m in (self.encoder, self.decoder1, self.decoder2):
            yield from m.parameters()

    def recon(self, batch):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        return w1, w2, w3


@register
class USAD(BaseModel):
    name = "usad"

    def __init__(self, z_size=32, batch=128, lr=1e-3, epochs=100):
        self.z_size = z_size
        self.batch = batch
        self.lr = lr
        self.epochs = epochs
        self.core = None

    def fit(self, train_w, val_w, patience=10, max_epochs=50, seed=0, **kw):
        torch, nn = _torch()
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        np.random.seed(seed)
        super().fit(train_w, val_w, patience, max_epochs, seed)
        if len(train_w) > MAX_TRAIN:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(train_w), size=MAX_TRAIN, replace=False)
            train_w = train_w[idx]
        w_size = int(train_w.shape[1] * train_w.shape[2])
        core = _USADCore(w_size, self.z_size)
        self.core = core
        opt1 = torch.optim.Adam(list(core.encoder.parameters()) + list(core.decoder1.parameters()), lr=self.lr)
        opt2 = torch.optim.Adam(list(core.encoder.parameters()) + list(core.decoder2.parameters()), lr=self.lr)
        X = torch.tensor(train_w.reshape(len(train_w), w_size), dtype=torch.float32)
        n_epochs = self.epochs
        for epoch in range(n_epochs):
            n = epoch + 1
            perm = torch.randperm(len(X))
            for i in range(0, len(X), self.batch):
                batch = X[perm[i:i + self.batch]]
                if len(batch) < 2:
                    continue
                w1, w2, w3 = core.recon(batch)
                loss1 = (1 / n) * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean((batch - w3) ** 2)
                loss2 = (1 / n) * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean((batch - w3) ** 2)
                opt1.zero_grad(); loss1.backward(); opt1.step()
                w1, w2, w3 = core.recon(batch)
                loss1 = (1 / n) * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean((batch - w3) ** 2)
                loss2 = (1 / n) * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean((batch - w3) ** 2)
                opt2.zero_grad(); loss2.backward(); opt2.step()
        self._epochs_run = n_epochs
        return self

    def score_windows(self, W):
        torch, _ = _torch()
        torch.set_num_threads(1)
        w_size = int(W.shape[1] * W.shape[2])
        X = torch.tensor(W.reshape(len(W), w_size), dtype=torch.float32)
        alpha, beta = 0.5, 0.5  # 官方 testing 默认
        scores = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch):
                batch = X[i:i + self.batch]
                z = self.core.encoder(batch)
                w1 = self.core.decoder1(z)
                w2 = self.core.decoder2(self.core.encoder(w1))
                s = alpha * torch.mean((batch - w1) ** 2, dim=1) + beta * torch.mean((batch - w2) ** 2, dim=1)
                scores.append(s.numpy())
        return np.concatenate(scores, axis=0).astype(np.float64)

    def count_params(self):
        if self.core is None:
            return 0
        return int(sum(p.numel() for p in self.core.parameters()))

    def serialize(self):
        import pickle
        return pickle.dumps({"name": self.name})
