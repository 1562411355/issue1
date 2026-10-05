#!/usr/bin/env python3
"""SMD (ServerMachineDataset) 原始目录 → data/smd/*.npz。

布局（OmniAnomaly）：
  train/<machine>       仅正常段，逗号分隔特征，无标签
  test/<machine>        含异常的测试段
  test_label/<machine>  与 test 等长的 0/1 标签

每台机器输出一个 npz：X=(n,f) float32，y=(n,) int8，train+test 按时间拼接。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_xy(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)


def convert(src: Path, out: Path) -> None:
    train_dir = src / "train"
    test_dir = src / "test"
    label_dir = src / "test_label"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise SystemExit(f"[convert_smd] {src} 下缺少 train/ 或 test/")
    out.mkdir(parents=True, exist_ok=True)
    machines = sorted(p.name for p in train_dir.iterdir() if p.is_file())
    if not machines:
        raise SystemExit(f"[convert_smd] train/ 为空")
    for m in machines:
        X_tr = load_xy(train_dir / m)
        X_te = load_xy(test_dir / m)
        lab_f = label_dir / m
        if not lab_f.exists():
            raise SystemExit(f"[convert_smd] 缺标签 {lab_f}")
        y_te = np.loadtxt(lab_f, delimiter=",", dtype=np.int8, ndmin=1)
        if len(y_te) != len(X_te):
            raise SystemExit(f"[convert_smd] {m} 测试点数 {len(X_te)} 与标签 {len(y_te)} 不一致")
        if X_tr.shape[1] != X_te.shape[1]:
            raise SystemExit(f"[convert_smd] {m} train/test 特征数不一致")
        X = np.concatenate([X_tr, X_te], axis=0)
        y = np.concatenate([np.zeros(len(X_tr), dtype=np.int8), y_te], axis=0)
        if not np.isfinite(X).all():
            bad = int((~np.isfinite(X)).sum())
            print(f"[convert_smd] 警告：{m} 含 {bad} 个非有限值，已置 0")
            X = np.nan_to_num(X)
        np.savez_compressed(out / f"{m}.npz", X=X, y=y, train_len=len(X_tr))
        print(f"[convert_smd] {m}.npz: {len(y)} 点（异常 {int(y.sum())}，"
              f"{X.shape[1]} 特征，train {len(X_tr)}+test {len(X_te)}）")
    print("[convert_smd] 完成。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="ServerMachineDataset 目录")
    ap.add_argument("--out", default="data/smd")
    a = ap.parse_args()
    convert(Path(a.src), Path(a.out))
