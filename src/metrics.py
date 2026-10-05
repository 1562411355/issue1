"""评估指标：PA（点调整）与非 PA 双报告 + AUC + 效率测量。

协议第 4 节：窗口分数 → 点分数（被覆盖窗口的中位数）；阈值在验证集搜索
（PA-F1 最优）；测试集同阈值双报告。
"""
from __future__ import annotations

import time

import numpy as np


def window_scores_to_point(scores: np.ndarray, w: int, stride: int, n_points: int) -> np.ndarray:
    """窗口分数映射回点级：取覆盖该点的窗口分数中位数（协议 §4）。"""
    m = len(scores)
    if m == 0:
        return np.zeros(n_points, dtype=np.float64)
    buckets: list[list[float]] = [[] for _ in range(n_points)]
    for i in range(m):
        lo = i * stride
        hi = min(lo + w, n_points)
        si = float(scores[i])
        for j in range(lo, hi):
            buckets[j].append(si)
    last = float(scores[-1])
    point = np.empty(n_points, dtype=np.float64)
    for j in range(n_points):
        b = buckets[j]
        point[j] = float(np.median(b)) if b else last
    return point


def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """标准 Point-Adjust（PA）：命中段内任一点 → 整段记为命中。

    Kim et al., AAAI 2019 "PA: Accurate Anomaly Detection through
    Point Adjustment" 的实现。
    """
    adj = y_pred.copy()
    n = len(y_true)
    i = 0
    while i < n:
        if y_true[i] == 1:
            j = i
            while j < n and y_true[j] == 1:
                j += 1
            if adj[i:j].sum() > 0:      # 段内至少一个检出
                adj[i:j] = 1            # 整段记命中
            i = j
        else:
            i += 1
    return adj


def prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """点级双报告 + AUC。"""
    y_pred = (scores > threshold).astype(np.int8)
    p, r, f = prf(y_true, y_pred)
    adj_pred = point_adjust(y_true, y_pred)
    pa_p, pa_r, pa_f = prf(y_true, adj_pred)
    from sklearn.metrics import average_precision_score, roc_auc_score

    auc_roc = roc_auc_score(y_true, scores) if 0 < y_true.sum() < len(y_true) else float("nan")
    auc_pr = average_precision_score(y_true, scores) if 0 < y_true.sum() < len(y_true) else float("nan")
    return {
        "precision": p, "recall": r, "f1": f,
        "precision_pa": pa_p, "recall_pa": pa_r, "f1_pa": pa_f,
        "auc_roc": auc_roc, "auc_pr": auc_pr,
    }


def search_threshold(y_val: np.ndarray, s_val: np.ndarray, quantiles: list[float]) -> tuple[float, dict]:
    """协议 4：验证集上按分位网格搜索，PA-F1 最优。"""
    best_tau, best = s_val.max() + 1e-9, {"f1_pa": -1.0}
    for q in quantiles:
        tau = float(np.quantile(s_val, q))
        ev = evaluate(y_val, s_val, tau)
        if ev["f1_pa"] > best["f1_pa"]:
            best_tau, best = tau, ev
    return best_tau, best


def efficiency(model, n_windows: int) -> dict:
    """参数量 / 体积 / CPU 单窗口推理延迟中位数（真机实测，非编造）。

    计时对象 = score_windows 对单个真实输入窗口的前向；数据准备不计入。
    """
    params = int(model.count_params())
    kb = round(len(model.serialize()) / 1024.0, 2)
    x1 = model.probe_input(1)
    for _ in range(5):  # 预热
        model.score_windows(x1)
    times = []
    for _ in range(n_windows):
        t0 = time.perf_counter()
        model.score_windows(x1)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {"params": params, "model_kb": kb, "latency_ms_cpu": round(float(np.median(times)), 3)}
