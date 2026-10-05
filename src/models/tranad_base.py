"""TranAD 适配（imperial-qore/TranAD，VLDB 2022）。

出处：https://github.com/imperial-qore/TranAD
（src/models.py 的 TranAD_Transformer / 训练-测试路径；本文件为管线适配，
架构按官方实现移植，超参见 protocol.md hyperparams.tranad。）

与官方差异（见 submission/deviations.md）：
- n_window 采用协议窗口 120（官方示例 n_window=10）
- 输入为管线统一 z-score 窗口；官方数据管道为 min-max
- 分数为 focus/adversarial 两相重构 MSE 融合（0.5/0.5），对齐官方训练融合形式
- max_train=15000 子采样（与 distill_ae 同口径，保证可跑）
"""
from __future__ import annotations

import numpy as np

from . import BaseModel, register

MAX_TRAIN = 15000


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


class _TranADCore:
    """官方 TranAD_Transformer 精简移植（保留 forward 语义）。"""

    def __init__(self, feats: int, n_window: int, hidden: int = 8):
        torch, nn = _torch()
        self._torch = torch
        self._nn = nn
        n = 2 * feats * n_window
        self.feats = feats
        self.n_window = n_window
        self.n = n
        self.n_hidden = hidden
        self.batch = 128
        self.transformer_encoder = nn.Sequential(
            nn.Linear(n, hidden), nn.ReLU(True),
            nn.Linear(hidden, n), nn.ReLU(True))
        self.transformer_decoder1 = nn.Sequential(
            nn.Linear(n, hidden), nn.ReLU(True),
            nn.Linear(hidden, 2 * feats), nn.ReLU(True))
        self.transformer_decoder2 = nn.Sequential(
            nn.Linear(n, hidden), nn.ReLU(True),
            nn.Linear(hidden, 2 * feats), nn.ReLU(True))
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def parameters(self):
        for m in (self.transformer_encoder, self.transformer_decoder1,
                  self.transformer_decoder2, self.fcn):
            yield from m.parameters()

    def encode(self, src, c):
        torch = self._torch
        src = torch.cat((src, c), dim=2)
        src = src.permute(1, 0, 2).flatten(start_dim=1)
        return self.transformer_encoder(src)

    def forward(self, src):
        """src: (n_window, batch, feats) -> (x1, x2) 各 (batch, feats) 最后一步重构。"""
        c = self._torch.zeros_like(src)
        x1 = self.transformer_decoder1(self.encode(src, c))
        x1 = x1.reshape(-1, 1, 2 * self.feats).permute(1, 0, 2)
        x1 = self.fcn(x1)  # (1, B, F)
        # 官方：c = (x1 - src) ** 2，x1 广播到整窗
        c = (x1 - src) ** 2
        x2 = self.transformer_decoder2(self.encode(src, c))
        x2 = x2.reshape(-1, 1, 2 * self.feats).permute(1, 0, 2)
        x2 = self.fcn(x2)
        return x1.squeeze(0), x2.squeeze(0)


@register
class TranAD(BaseModel):
    name = "tranad"

    def __init__(self, lr=None, batch=128, hidden=8, n_window=120):
        self.lr = lr
        self.batch = batch
        self.hidden = hidden
        self.n_window = n_window
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
        feats = train_w.shape[2]
        win = train_w.shape[1]
        core = _TranADCore(feats, win, self.hidden)
        self.core = core
        self.batch = getattr(core, "batch", self.batch)
        lr = self.lr if self.lr is not None else 1e-3
        opt = torch.optim.Adam(core.parameters(), lr=lr)
        # 官方 main.py：num_epochs=5，批次内 window 维在前
        n_epochs = 5
        X = torch.tensor(train_w, dtype=torch.float32)
        msel = nn.MSELoss(reduction="none")
        for epoch in range(n_epochs):
            n = epoch + 1
            perm = torch.randperm(len(X))
            for i in range(0, len(X), self.batch):
                batch = X[perm[i:i + self.batch]]  # (B, W, F)
                if len(batch) < 2:
                    continue
                window = batch.permute(1, 0, 2)  # (W, B, F)
                elem = window[-1, :, :]          # (B, F) 最后时间步
                x1, x2 = core.forward(window)
                loss = (1 / n) * msel(x1, elem) + (1 - 1 / n) * msel(x2, elem)
                loss = loss.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._epochs_run = n_epochs
        return self

    def score_windows(self, W):
        torch, nn = _torch()
        torch.set_num_threads(1)
        self.core.transformer_encoder.eval()
        # 简化：无 dropout，推理同前向
        X = torch.tensor(W, dtype=torch.float32)
        scores = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch):
                batch = X[i:i + self.batch]
                window = batch.permute(1, 0, 2)
                elem = window[-1, :, :]
                x1, x2 = self.core.forward(window)
                # focus + adversarial 融合（0.5/0.5）
                s = 0.5 * ((x1 - elem) ** 2).mean(dim=1) + 0.5 * ((x2 - elem) ** 2).mean(dim=1)
                scores.append(s.numpy())
        return np.concatenate(scores, axis=0).astype(np.float64)

    def count_params(self):
        if self.core is None:
            return 0
        return int(sum(p.numel() for p in self.core.parameters()))

    def serialize(self):
        import pickle
        return pickle.dumps({"name": self.name, "state": self.__dict__.get("core") is not None})
