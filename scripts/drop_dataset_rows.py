# -*- coding: utf-8 -*-
"""删除 results.csv 中指定 dataset 的行（重跑前清理）。

用法：python scripts/drop_dataset_rows.py --dataset msl
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "results" / "results.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--yes", action="store_true", help="跳过确认")
    args = ap.parse_args()

    if not CSV_PATH.exists():
        print(f"[drop] {CSV_PATH} 不存在")
        return 0

    with CSV_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows if r.get("dataset") != args.dataset]
    drop_n = len(rows) - len(kept)
    print(f"[drop] dataset={args.dataset} 将删除 {drop_n} 行，保留 {len(kept)} 行")
    if drop_n == 0:
        return 0
    if not args.yes:
        ans = input("确认删除？[y/N] ").strip().lower()
        if ans != "y":
            print("[drop] 已取消")
            return 1

    fieldnames = list(rows[0].keys()) if rows else []
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)
    print(f"[drop] 已写回 {CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
