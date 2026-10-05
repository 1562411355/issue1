"""消融模型：教师单体 AE、无蒸馏学生、λ 敏感性（协议 distill_ablation）。

与 distill_ae 一致：训练窗 >MAX_TRAIN 时固定种子抽样，保证可比与可跑。
主网格 8 模型不在此处改动。
"""
from __future__ import annotations

import numpy as np

from . import register
from .ae_mlp import AEMLP
from .distill_ae import MAX_TRAIN, DistillAE


def _subsample(train_w, seed):
    if len(train_w) > MAX_TRAIN:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train_w), size=MAX_TRAIN, replace=False)
        return train_w[idx]
    return train_w


@register
class AELTeacher(AEMLP):
    """教师结构单体 AE：h1=128, z=32（消融：不用蒸馏的“大模型”上限）。"""

    name = "ae_l_teacher"

    def __init__(self):
        super().__init__(h1=128, z=32)

    def fit(self, train_w, val_w, patience=10, max_epochs=50, seed=0, lr=1e-3, batch=64):
        train_w = _subsample(train_w, seed)
        return super().fit(train_w, val_w, patience=patience, max_epochs=max_epochs,
                           seed=seed, lr=lr, batch=batch)


@register
class AESNoDistill(AEMLP):
    """学生结构单体 AE：h1=16, z=4，仅重构（等价 λ=0，无教师）。"""

    name = "ae_s_nodistill"

    def __init__(self):
        super().__init__(h1=16, z=4)

    def fit(self, train_w, val_w, patience=10, max_epochs=50, seed=0, lr=1e-3, batch=64):
        train_w = _subsample(train_w, seed)
        return super().fit(train_w, val_w, patience=patience, max_epochs=max_epochs,
                           seed=seed, lr=lr, batch=batch)


@register
class DistillAELam01(DistillAE):
    name = "distill_ae_lam01"

    def __init__(self):
        super().__init__(teacher_h1=128, teacher_z=32, student_h1=16, student_z=4, lam=0.1)


@register
class DistillAELam05(DistillAE):
    name = "distill_ae_lam05"

    def __init__(self):
        super().__init__(teacher_h1=128, teacher_z=32, student_h1=16, student_z=4, lam=0.5)
