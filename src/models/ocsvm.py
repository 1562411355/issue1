"""单类 SVM：窗口展平后 OCSVM，-decision_function 作异常分数。

SMD 等大训练集（>5000 窗）时随机子采样再拟合：OCSVM 精确解是 O(n²)～O(n³)，
在 13 万窗上不可行；子采样 5000 窗为可复现随机子集（seed 固定）。
"""
from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM

from . import BaseModel, register

MAX_TRAIN = 5000  # 超过则子采样（可复现，见 Limitations）


@register
class OCSVM(BaseModel):
    name = "ocsvm"

    def __init__(self, nu=0.05, gamma="scale"):
        self.nu = nu
        self.gamma = gamma
        self.clf = None

    def fit(self, train_w, val_w, patience=10, max_epochs=50, seed=0):
        super().fit(train_w, val_w, patience, max_epochs, seed)
        X = train_w.reshape(len(train_w), -1)
        if len(X) > MAX_TRAIN:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), size=MAX_TRAIN, replace=False)
            X = X[idx]
        self.clf = OneClassSVM(nu=self.nu, gamma=self.gamma).fit(X)
        return self

    def score_windows(self, W):
        return -self.clf.decision_function(W.reshape(len(W), -1))

    def count_params(self):
        n = self.clf.support_vectors_.shape[0]
        d = self.clf.support_vectors_.shape[1]
        return int(n * d)
