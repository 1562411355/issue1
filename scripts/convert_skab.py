#!/usr/bin/env python3
"""SKAB 原始 csv → data/skab/*.npz 转换器。

SKAB（工业测试台异常检测基准）官方仓库提供多个 csv，每行为一个时间点，
列为传感器读数，末列 anomaly 为 0/1 标签（另有 changepoint 列，忽略）。

用法：
  python3 scripts/convert_skab.py --src /path/to/SKAB/master/ --out data/skab/

src 目录下所有 *.csv 逐一转为同名 .npz（X + y），按文件名顺序拼接。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def convert(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    csvs = sorted(src.rglob("*.csv"))
    if not csvs:
        raise SystemExit(f"[convert] {src} 下没有 csv 文件")
    for f in csvs:
        # SKAB 官方 csv 为分号分隔，首列 datetime
        df = pd.read_csv(f, sep=None, engine="python")
        drop = {"anomaly", "changepoint", "datetime", "timestamp", "datetime;"}
        cols = [c for c in df.columns if c.lower().strip() not in drop]
        y_cols = [c for c in df.columns if c.lower().strip() == "anomaly"]
        X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        if y_cols:
            y = pd.to_numeric(df[y_cols[0]], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
        else:
            # anomaly-free 等无标签序列：全部视为正常
            y = np.zeros(len(df), dtype=np.int8)
            print(f"[convert] {f.name} 无 anomaly 列，y 全部置 0")
        if not np.isfinite(X).all():
            bad = int((~np.isfinite(X)).sum())
            print(f"[convert] 警告：{f.name} 含 {bad} 个非有限值，已置 0")
            X = np.nan_to_num(X)
        rel = f.relative_to(src).as_posix().replace("/", "_").replace(".csv", "")
        name = rel + ".npz"
        np.savez_compressed(out / name, X=X, y=y)
        n_anom = int(y.sum())
        print(f"[convert] {name}: {len(y)} 点（异常 {n_anom}，"
              f"{X.shape[1]} 特征）")
    print(f"[convert] 完成。data_source 将自动标记为 real。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="SKAB 原始目录")
    ap.add_argument("--out", default="data/skab")
    a = ap.parse_args()
    convert(Path(a.src), Path(a.out))
