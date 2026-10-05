"""数据加载与合成数据（仅供管线自检）。

设计书红线（第 1.1 节 / 第 6 节）：真实实验必须真实运行。
- 全量运行：data/<name>/ 下没有真实数据 → 硬报错退出，绝不用合成顶替。
- --smoke 自检：允许合成数据，但返回值永久携带 data_source="synthetic"，
  results.csv 会记录，verify.py 看到即 FAIL（论文不可能误用）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"


@dataclass
class Dataset:
    name: str
    X: np.ndarray            # (n, features) float32
    y: np.ndarray            # (n,) int8, 1=异常
    data_source: str         # "real" 或 "synthetic"
    train_len: int | None = None  # 若给出：[:train_len] 为官方 train 段


def _load_real(name: str) -> tuple[np.ndarray, np.ndarray, int | None] | None:
    """真实数据格式：data/<name>/<任意名>.npz，含 X(float) 与 y(0/1)。

    可选标量 train_len：官方 train 段长度（SMD/MSL 的 train/test 划分）。
    SKAB 原始 csv 可用 scripts/convert_skab.py 转换。
    """
    d = DATA_ROOT / name
    if not d.is_dir():
        return None
    npz_files = sorted(d.glob("*.npz"))
    if not npz_files:
        return None
    xs, ys = [], []
    train_lens = []
    for f in npz_files:
        z = np.load(f)
        xs.append(np.asarray(z["X"], dtype=np.float32))
        ys.append(np.asarray(z["y"], dtype=np.int8))
        if "train_len" in z:
            train_lens.append(int(z["train_len"]))
        else:
            train_lens.append(None)
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError(f"{f} 中 X/y 形状不合法")
    # 多文件拼接时 train_len 必须全员提供，否则视为无官方划分
    if train_lens and all(t is not None for t in train_lens):
        # 各文件 train 段在拼接后的累计偏移
        total = 0
        offset = 0
        for arr_len, tl in zip([len(a) for a in xs], train_lens):
            if tl is not None:
                offset = total + tl
            total += arr_len
        train_len = offset
    else:
        train_len = None
    return X, y, train_len


def _make_synthetic(name: str, n: int = 2400, dim: int = 6, seed: int = 7) -> Dataset:
    """合成工业风格时序（仅供 --smoke 管线自检，结果永不入论文）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 10.0
    X = np.stack(
        [
            np.sin(t * (0.5 + 0.1 * k)) + 0.05 * rng.standard_normal(n)
            for k in range(dim)
        ],
        axis=1,
    ).astype(np.float32)
    y = np.zeros(n, dtype=np.int8)
    for start, length in [(int(n * 0.35), 80), (int(n * 0.55), 120), (int(n * 0.8), 60)]:
        end = min(start + length, n)
        y[start:end] = 1
        X[start:end] += rng.uniform(2.0, 4.0)  # 明显偏移，保证信号可学
    return Dataset(name, X, y, data_source="synthetic")


def load_dataset(name: str, allow_synthetic: bool = False) -> Dataset:
    real = _load_real(name)
    if real is not None:
        X, y, train_len = real
        return Dataset(name, X, y, data_source="real", train_len=train_len)
    if allow_synthetic:
        print(f"[data] 警告：{name} 无真实数据，--smoke 使用合成数据（禁入论文）")
        return _make_synthetic(name)
    raise FileNotFoundError(
        f"data/{name}/ 下没有 .npz 真实数据。全量运行禁止合成数据顶替——"
        f"请先按 data/README.md 放入真实数据（SKAB 可用 scripts/convert_skab.py 转换）。"
    )


# ---------------------------------------------------------------------------
# 划分与窗口化（协议第 2/3 节）
# ---------------------------------------------------------------------------
@dataclass
class Splits:
    train_normal: np.ndarray   # 仅正常，训练
    val_normal: np.ndarray     # 正常，早停
    val_mixed: np.ndarray      # 含异常，阈值搜索（y 同步切片）
    val_mixed_y: np.ndarray
    test: np.ndarray           # 测试（含异常）
    test_y: np.ndarray
    mu: np.ndarray             # z-score 统计量只来自训练段
    sigma: np.ndarray
    data_source: str


def split_and_normalize(ds: Dataset, val_ratio: float, test_ratio: float) -> Splits:
    """时间顺序划分（协议第 2 节）。

    有官方 train 段（train_len，SMD/MSL）时：
        |--- train_len（训练，SMD/MSL 通常全正常）---|--- 验证 ---|--- 测试 ---|
        验证/测试在余下段内按 val:test 切；训练段若混入异常则剔除。
    无 train_len（SKAB）时用比例时间切分：
        |--- 训练区(1-test-val) ---|--- 验证区 ---|--- 测试区 ---|
    z-score 统计量只来自训练段。
    """
    X, y = ds.X, ds.y
    n = len(X)
    if ds.train_len is not None:
        train_end = int(ds.train_len)
        # 余下段（官方 test）做分层 50/50：保证 val/test 都含正常+异常
        # （MSL 时间尾部全是 attack，简单对半切会得到单类测试集）
        idx_rest = np.arange(train_end, n)
        rest_norm = idx_rest[y[idx_rest] == 0]
        rest_anom = idx_rest[y[idx_rest] == 1]
        vn, va = len(rest_norm) // 2, len(rest_anom) // 2
        val_idx = np.sort(np.concatenate([rest_norm[:vn], rest_anom[:va]]))
        test_idx = np.sort(np.concatenate([rest_norm[vn:], rest_anom[va:]]))
        if len(val_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"{ds.name} 余下段无法分成含双类的验证/测试")
        val_start = int(val_idx[0])  # 仅用于兼容占位，不再按时间比例切
        test_start = int(test_idx[0])
    else:
        test_start = int(n * (1 - test_ratio))
        val_start = int(test_start * (1 - val_ratio / (1 - test_ratio)))
        train_end = val_start

    region = np.arange(n)
    train_idx = region[(region < (train_end if ds.train_len is not None else val_start)) & (y == 0)]
    if ds.train_len is not None:
        val_norm_idx = val_idx[y[val_idx] == 0]
        val_mixed_idx, test_idx_arr = val_idx, test_idx
        mu = X[train_idx].mean(axis=0)
        sigma = X[train_idx].std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        return Splits(
            train_normal=Xn[train_idx],
            val_normal=Xn[val_norm_idx],
            val_mixed=Xn[val_mixed_idx],
            val_mixed_y=y[val_mixed_idx],
            test=Xn[test_idx_arr],
            test_y=y[test_idx_arr],
            mu=mu, sigma=sigma, data_source=ds.data_source,
        )

    val_norm_idx = region[(region >= val_start) & (region < test_start) & (y == 0)]
    mu = X[train_idx].mean(axis=0)
    sigma = X[train_idx].std(axis=0) + 1e-8
    Xn = (X - mu) / sigma
    return Splits(
        train_normal=Xn[train_idx],
        val_normal=Xn[val_norm_idx],
        val_mixed=Xn[val_start:test_start],
        val_mixed_y=y[val_start:test_start],
        test=Xn[test_start:],
        test_y=y[test_start:],
        mu=mu, sigma=sigma, data_source=ds.data_source,
    )


def sliding_windows(x: np.ndarray, w: int, stride: int) -> np.ndarray:
    """(n, f) -> (m, w, f)；长度不足 w 时返回空批。"""
    n = len(x)
    if n < w:
        return np.empty((0, w, x.shape[1]), dtype=x.dtype)
    m = 1 + (n - w) // stride
    idx = np.arange(w)[None, :] + stride * np.arange(m)[:, None]
    return x[idx]


def windows_of(X: np.ndarray, y: np.ndarray, w: int, stride: int):
    """同时返回窗口与窗口标签（窗口内含异常点即为异常窗口，仅诊断用）。"""
    W = sliding_windows(X, w, stride)
    yl = sliding_windows(y[:, None].astype(np.float32), w, stride)[:, :, 0]
    return W, (yl.max(axis=1) > 0.5).astype(np.int8)
