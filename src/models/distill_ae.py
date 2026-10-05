"""蒸馏学生模型（课题轻量化核心）：大教师 AE → 微型学生 AE，分数蒸馏。

教师 = AE-L（宽瓶颈），训练后冻结；学生 = AE-S（约 1/10 参数）。总损失
L = L_recon + λ·L_distill，其中 L_distill = MSE(zs, zt)，zs/zt 为批内
z 归一化的学生/教师逐窗口异常分数。对 zs 的 z 归一化耦合求精确解析梯度
（Jacobian 向量化展开），非近似。
"""
from __future__ import annotations

import numpy as np

from . import BaseModel, register
from .ae_mlp import AEMLP

# 大训练集（SMD 的 13.5 万窗）上双网蒸馏会远超可跑时长；固定种子抽样以保证可行性
MAX_TRAIN = 15000


@register
class DistillAE(BaseModel):
    name = "distill_ae"

    def __init__(self, teacher_h1=128, teacher_z=32, student_h1=16, student_z=4, lam=0.3):
        self.teacher = AEMLP(h1=teacher_h1, z=teacher_z)
        self.student = AEMLP(h1=student_h1, z=student_z)
        self.lam = lam

    def fit(self, train_w, val_w, patience=10, max_epochs=50, seed=0, lr=1e-3, batch=64):
        super().fit(train_w, val_w, patience, max_epochs, seed)
        if len(train_w) > MAX_TRAIN:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(train_w), size=MAX_TRAIN, replace=False)
            train_w = train_w[idx]
        # 1) 训练教师（冻结）
        self.teacher.fit(train_w, val_w, patience=patience, max_epochs=max_epochs,
                         seed=seed, lr=lr, batch=batch)
        # 2) 训练学生：重构 + 分数蒸馏（z 归一化匹配的精确梯度）
        s = self.student
        s.fit(train_w, val_w, patience=patience, max_epochs=max_epochs,
              seed=seed, lr=lr, batch=batch)  # 常规预训练初始化
        rng = np.random.default_rng(seed)
        P = [s.W1, s.b1, s.W2, s.b2, s.W3, s.b3, s.W4, s.b4]
        m = [np.zeros_like(p) for p in P]
        v = [np.zeros_like(p) for p in P]
        best, wait, best_P = np.inf, 0, [p.copy() for p in P]
        t = 0
        for _ in range(max_epochs):
            for xb in self._batches(train_w, batch, rng):
                t += 1
                X2, a1, a2, a3, out = s._forward(xb)
                B, d = X2.shape
                t_sc = self.teacher.score_windows(xb)          # 冻结教师分数
                s_sc = np.mean((out - X2) ** 2, axis=1)         # 学生分数
                sd_s = s_sc.std() + 1e-8
                zs = (s_sc - s_sc.mean()) / sd_s
                zt = (t_sc - t_sc.mean()) / (t_sc.std() + 1e-8)
                # L_distill 对 zs 的梯度 -> 对 s_sc 的精确回传
                dl_dzs = 2.0 * (zs - zt) / B
                c0 = dl_dzs.sum()
                c1 = float((dl_dzs * zs).sum())
                dl_ds = self.lam * (dl_dzs - c0 / B - c1 * zs / B) / sd_s
                # 合并到输出层梯度：d s_i / d out_ij = 2(out_ij - X2_ij)/d
                dOut = 2.0 * (out - X2) / (B * d) + dl_ds[:, None] * (2.0 * (out - X2) / d)
                gW4 = a3.T @ dOut; gb4 = dOut.sum(0)
                da3 = dOut @ s.W4.T * (1 - a3**2)
                gW3 = a2.T @ da3; gb3 = da3.sum(0)
                da2 = da3 @ s.W3.T * (1 - a2**2)
                gW2 = a1.T @ da2; gb2 = da2.sum(0)
                da1 = da2 @ s.W2.T * (1 - a1**2)
                gW1 = X2.T @ da1; gb1 = da1.sum(0)
                self._adam([gW1, gb1, gW2, gb2, gW3, gb3, gW4, gb4], P, m, v, t, lr)
            tv = self.teacher.score_windows(val_w)
            sv = s.score_windows(val_w)
            val_loss = float(np.mean((s._forward(val_w)[4] - val_w.reshape(len(val_w), -1)) ** 2)) \
                - 0.1 * _spearman(tv, sv)
            if val_loss < best - 1e-6:
                best, wait, best_P = val_loss, 0, [p.copy() for p in P]
            else:
                wait += 1
                if wait >= patience:
                    break
        for p, bp in zip(P, best_P):
            p[...] = bp
        return self

    def score_windows(self, W):
        return self.student.score_windows(W)

    def count_params(self):
        return int(self.student.count_params())

    def serialize(self) -> bytes:
        """部署体积 = 仅学生（教师只在训练期存在，不进边缘端）。"""
        return self.student.serialize()

    def teacher_params(self):
        return int(self.teacher.count_params())


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra = (ra - ra.mean()) / (ra.std() + 1e-9)
    rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return float(np.mean(ra * rb))
