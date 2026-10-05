"""k 近邻距离：窗口展平后到训练集第 k 近邻的距离作异常分数。

大训练集（>5000 窗）时子采样后再建索引，避免 13 万窗 × 4560 维暴力检索。
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from . import BaseModel, register

MAX_TRAIN = 5000


@register
class KNNDist(BaseModel):
    name = "knn_dist"

    def __init__(self, k=5):
        self.k = k
        self.nn = None

    def fit(self, train_w, val_w, patience=10, max_epochs=50, seed=0):
        super().fit(train_w, val_w, patience, max_epochs, seed)
        X = train_w.reshape(len(train_w), -1)
        if len(X) > MAX_TRAIN:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), size=MAX_TRAIN, replace=False)
            X = X[idx]
        self.nn = NearestNeighbors(n_neighbors=self.k, n_jobs=-1).fit(X)
        return self

    def score_windows(self, W):
        dist, _ = self.nn.kneighbors(W.reshape(len(W), -1))
        return dist[:, -1].astype(np.float64)

    def count_params(self):
        # 存储的训练窗口本身
        return int(self.nn._fit_X.size)
