"""模型注册表（协议 models 列表在此对位）。

统一接口（设计书 4.2：一行一个实验，含全部指标与硬件耗时）：
- fit(train_w, val_w, ...) 早停训练
- score_windows(W) -> (m,) 异常分数（越大越异常）
- count_params / serialize / probe_input 支撑效率测量
"""
from __future__ import annotations

import pickle

import numpy as np

MODEL_REGISTRY: dict[str, type] = {}


def register(cls):
    MODEL_REGISTRY[cls.name] = cls
    return cls


def get_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"未知模型 {name}；可用：{sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


class BaseModel:
    name = "base"
    input_shape: tuple[int, int] | None = None  # (w, f)

    def fit(self, train_w, val_w, patience=10, max_epochs=50, seed=0):
        self.input_shape = (train_w.shape[1], train_w.shape[2])
        return self

    def score_windows(self, W: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def count_params(self) -> int:
        return 0

    def serialize(self) -> bytes:
        return pickle.dumps(self)

    def probe_input(self, k: int) -> np.ndarray:
        """随机窗口批，仅用于延迟测量的数据准备（不计时）。"""
        w, f = self.input_shape
        return np.random.default_rng(0).standard_normal((k, w, f)).astype(np.float32)

    # ---- 神经模型共享：Adam + 小批量 + 早停 ----
    @staticmethod
    def _batches(X, batch, rng):
        idx = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            yield X[idx[i : i + batch]]

    @staticmethod
    def _adam(grads, params, m, v, t, lr=1e-3):
        b1, b2, eps = 0.9, 0.999, 1e-8
        out = []
        for g, p, mm, vv in zip(grads, params, m, v):
            mm[:] = b1 * mm + (1 - b1) * g
            vv[:] = b2 * vv + (1 - b2) * g * g
            mhat = mm / (1 - b1**t)
            vhat = vv / (1 - b2**t)
            p -= lr * mhat / (np.sqrt(vhat) + eps)
            out.append(p)
        return out

# 导入全部模型模块以触发注册（注册表完整性 = 协议 models 列表可对位）
from . import ae_mlp, lstm_ad, pca_recon, iforest, ocsvm, knn_dist, deep_svdd, distill_ae, ablation, tranad_base, usad_base  # noqa: E402,F401
