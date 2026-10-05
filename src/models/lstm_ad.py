"""LSTM-AD：LSTM 预测器，预测误差作异常分数（numpy 手写 LSTM + BPTT）。

输入窗口 (w, f)，模型逐步预测下一时间点，窗口分数 = 平均预测误差。
"""
from __future__ import annotations

import numpy as np

from . import BaseModel, register


@register
class LSTMAD(BaseModel):
    name = "lstm_ad"

    def __init__(self, hidden=32):
        self.hidden = hidden

    def _init(self, f, seed):
        rng = np.random.default_rng(seed)
        s = lambda a, b: rng.normal(0, np.sqrt(2.0 / (a + b)), (a, b)).astype(np.float32)
        self.Wx = s(f, 4 * self.hidden)          # 输入 -> 4 门
        self.Wh = s(self.hidden, 4 * self.hidden)  # 隐状态 -> 4 门
        self.b = np.zeros(4 * self.hidden, np.float32)
        self.Wy = s(self.hidden, f)
        self.by = np.zeros(f, np.float32)
        # 遗忘门偏置 +1（惯例，利于长程记忆）
        self.b[1 * self.hidden : 2 * self.hidden] = 1.0

    def _forward_seq(self, X):
        """X:(B, T, F)。逐时间步，返回预测与全部缓存。"""
        B, T, F = X.shape
        H = self.hidden
        h = np.zeros((B, H), np.float32)
        c = np.zeros((B, H), np.float32)
        hs, cs, ig, fg, og, gg, preds = [], [], [], [], [], [], []
        for t in range(T):
            a = X[:, t] @ self.Wx + h @ self.Wh + self.b
            i = 1.0 / (1.0 + np.exp(-a[:, :H]))
            f = 1.0 / (1.0 + np.exp(-a[:, H : 2 * H]))
            o = 1.0 / (1.0 + np.exp(-a[:, 2 * H : 3 * H]))
            g = np.tanh(a[:, 3 * H :])
            c = f * c + i * g
            c = np.clip(c, -5.0, 5.0)
            h = o * np.tanh(c)
            hs.append(h); cs.append(c)
            ig.append(i); fg.append(f); og.append(o); gg.append(g)
            preds.append(h @ self.Wy + self.by)
        cache = dict(hs=hs, cs=cs, ig=ig, fg=fg, og=og, gg=gg)
        return np.stack(preds, axis=1), cache  # (B, T, F)

    def fit(self, train_w, val_w, patience=10, max_epochs=50, seed=0, lr=1e-3, batch=64):
        super().fit(train_w, val_w, patience, max_epochs, seed)
        F = train_w.shape[2]
        self._init(F, seed)
        rng = np.random.default_rng(seed)

        def loss_and_grads(Xb):
            B, T, Fd = Xb.shape
            H = self.hidden
            preds, cache = self._forward_seq(Xb)
            Y = np.concatenate([Xb[:, 1:], Xb[:, -1:, :]], axis=1)  # 预测下一点
            dP = 2.0 * (preds - Y) / (B * T * Fd)
            gWy = np.zeros_like(self.Wy); gby = np.zeros_like(self.by)
            gWx = np.zeros_like(self.Wx); gWh = np.zeros_like(self.Wh)
            gb = np.zeros_like(self.b)
            dH_next = np.zeros((B, H), np.float32)
            dC_next = np.zeros((B, H), np.float32)
            for t in reversed(range(T)):
                h = cache["hs"][t]; c = cache["cs"][t]
                c_prev = cache["cs"][t - 1] if t > 0 else np.zeros_like(c)
                h_prev = cache["hs"][t - 1] if t > 0 else np.zeros((B, H), np.float32)
                # 输出层梯度
                gWy += h.T @ dP[:, t]; gby += dP[:, t].sum(0)
                dH = dP[:, t] @ self.Wy.T + dH_next
                tanh_c = np.tanh(c)
                dO = dH * tanh_c
                dC = dH * cache["og"][t] * (1 - tanh_c**2) + dC_next
                dF = dC * c_prev
                dI = dC * cache["gg"][t]
                dG = dC * cache["ig"][t]
                i, f, o, g = cache["ig"][t], cache["fg"][t], cache["og"][t], cache["gg"][t]
                dI = dI * i * (1 - i)
                dF = dF * f * (1 - f)
                dO = dO * o * (1 - o)
                dG = dG * (1 - g * g)
                dA = np.concatenate([dI, dF, dO, dG], axis=1)  # (B, 4H)
                gWx += Xb[:, t].T @ dA
                gWh += h_prev.T @ dA
                gb += dA.sum(0)
                dH_next = dA @ self.Wh.T
                dC_next = dC * cache["fg"][t]
            # 梯度裁剪
            for g in (gWx, gWh, gWy):
                n = np.linalg.norm(g)
                if n > 5.0:
                    g *= 5.0 / n
            return float(np.mean((preds - Y) ** 2)), [gWx, gWh, gb, gWy, gby]

        P = [self.Wx, self.Wh, self.b, self.Wy, self.by]
        m = [np.zeros_like(p) for p in P]
        v = [np.zeros_like(p) for p in P]
        best, wait, best_P = np.inf, 0, [p.copy() for p in P]
        t = 0
        for epoch in range(max_epochs):
            for xb in self._batches(train_w, batch, rng):
                t += 1
                _, grads = loss_and_grads(xb)
                self._adam(grads, P, m, v, t, lr)
            val_loss = float(np.mean((self._forward_seq(val_w)[0]
                                      - np.concatenate([val_w[:, 1:], val_w[:, -1:, :]], 1)) ** 2))
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
        preds, _ = self._forward_seq(W)
        Y = np.concatenate([W[:, 1:], W[:, -1:, :]], axis=1)
        return np.mean((preds - Y) ** 2, axis=(1, 2))

    def count_params(self):
        return int(self.Wx.size + self.Wh.size + self.b.size + self.Wy.size + self.by.size)
