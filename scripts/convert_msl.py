#!/usr/bin/env python3
"""MSL (GDN 版 csv) → data/msl/*.npz。

布局：
  train.csv   无标签（全正常），列：timestamp + 传感器
  test.csv    末列 attack 为 0/1 标签，其余同 train
  list.txt    通道名（仅供核对）

输出 data/msl/msl.npz：X=train+test 特征，y=train 全 0 + test 的 attack。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def convert(src: Path, out: Path) -> None:
    train_f = src / "train.csv"
    test_f = src / "test.csv"
    if not train_f.exists() or not test_f.exists():
        raise SystemExit(f"[convert_msl] {src} 下缺 train.csv / test.csv")
    out.mkdir(parents=True, exist_ok=True)
    df_tr = pd.read_csv(train_f)
    df_te = pd.read_csv(test_f)

    def split_xy(df: pd.DataFrame, has_y: bool):
        cols = [c for c in df.columns if c.lower() not in ("timestamp", "attack", "anomaly", "label")]
        X = df[cols].to_numpy(dtype=np.float32)
        if has_y:
            ycol = [c for c in df.columns if c.lower() in ("attack", "anomaly", "label")]
            if not ycol:
                raise SystemExit("[convert_msl] test.csv 缺 attack/label 列")
            y = df[ycol[0]].to_numpy(dtype=np.int8)
        else:
            y = np.zeros(len(df), dtype=np.int8)
        return X, y

    X_tr, y_tr = split_xy(df_tr, has_y=False)
    X_te, y_te = split_xy(df_te, has_y=True)
    if X_tr.shape[1] != X_te.shape[1]:
        raise SystemExit(f"[convert_msl] train/test 特征数不一致：{X_tr.shape[1]} vs {X_te.shape[1]}")
    X = np.concatenate([X_tr, X_te], axis=0)
    y = np.concatenate([y_tr, y_te], axis=0)
    if not np.isfinite(X).all():
        bad = int((~np.isfinite(X)).sum())
        print(f"[convert_msl] 警告：含 {bad} 个非有限值，已置 0")
        X = np.nan_to_num(X)
    np.savez_compressed(out / "msl.npz", X=X, y=y, train_len=len(X_tr))
    print(f"[convert_msl] msl.npz: {len(y)} 点（异常 {int(y.sum())}，"
          f"{X.shape[1]} 特征，train {len(X_tr)}+test {len(X_te)}）")
    print("[convert_msl] 完成。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="MSL csv 目录（含 train.csv/test.csv）")
    ap.add_argument("--out", default="data/msl")
    a = ap.parse_args()
    convert(Path(a.src), Path(a.out))
