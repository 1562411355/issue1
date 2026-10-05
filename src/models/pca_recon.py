"""PCA 重构误差：窗口展平后 PCA 重构，误差作异常分数。"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from . import BaseModel, register


@register
class PCARecon(BaseModel):
    name = "pca_recon"
    # 协议显著性检验的对照基线

    def __init__(self, n_components=0.95):
        self.n_components = n_components
        self.pca = None

    def fit(self, train_w, val_w, patience=10, max_epochs=50, seed=0):
        super().fit(train_w, val_w, patience, max_epochs, seed)
        X = train_w.reshape(len(train_w), -1)
        self.pca = PCA(n_components=self.n_components, random_state=seed).fit(X)
        return self

    def score_windows(self, W):
        X = W.reshape(len(W), -1)
        Z = self.pca.transform(X)
        R = self.pca.inverse_transform(Z)
        return np.mean((X - R) ** 2, axis=1)

    def count_params(self):
        # components + mean
        return int(self.pca.components_.size + self.pca.mean_.size)
