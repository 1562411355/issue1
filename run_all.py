#!/usr/bin/env python3
"""链路 B：实验批处理入口（CP2 检查点：运行一条命令，确认首行日志后离开）。

设计书 4.2：读取 configs/protocol.md 已锁定的协议（窗口 60、步长 10、
早停 patience 10、3 种子、验证集搜阈值、PA/非 PA 双报告），遍历
模型×数据集×种子全组合，单卡顺序排队。失败自动重试 1 次，再失败则
记录并跳过；断点续跑（results.csv 已有组合自动跳过）。

用法：
  python3 run_all.py                 # 全量（协议须已锁定，即 CP1 已批准）
  python3 run_all.py --smoke         # 管线自检：合成数据、8 模型×1 数据集×1 种子
  python3 run_all.py --models ae_mlp,distill_ae --datasets skab --seeds 0
  python3 run_all.py --allow-unlocked   # 调试：协议未锁定也强制跑（不推荐）
"""
from __future__ import annotations

import argparse
import csv
import os
import time
import traceback
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)
from datetime import datetime
from pathlib import Path

import numpy as np

from src import protocol as proto
from src.data import load_dataset, sliding_windows, split_and_normalize
from src.metrics import efficiency, evaluate, search_threshold, window_scores_to_point
from src.models import get_model

REPO = Path(__file__).resolve().parent
RESULTS_CSV = REPO / "results" / "results.csv"
FAILED_LOG = REPO / "results" / "failed_runs.log"

CSV_COLUMNS = [
    "timestamp", "protocol_version", "model", "dataset", "seed", "data_source",
    "threshold", "precision", "recall", "f1",
    "precision_pa", "recall_pa", "f1_pa", "auc_roc", "auc_pr",
    "params", "model_kb", "latency_ms_cpu", "latency_ms_rpi", "teacher_params",
    "train_seconds", "epochs_run", "retries", "notes",
]

METRIC_COLS = ["threshold", "precision", "recall", "f1", "precision_pa", "recall_pa",
               "f1_pa", "auc_roc", "auc_pr", "latency_ms_cpu"]


def load_rpi_latencies(cfg) -> dict[tuple[str, str], float]:
    """树莓派真机延迟（可选，人工导入，绝不编造）。"""
    f = REPO / cfg.get("rpi_latency_file", "results/rpi_latency.csv")
    out = {}
    if f.exists():
        for row in csv.DictReader(f.open(encoding="utf-8")):
            try:
                out[(row["model"], row["dataset"])] = float(row["latency_ms_rpi"])
            except (KeyError, ValueError):
                continue
    return out


# 进程内数据缓存：同一 worker 跑同数据集的多个组合时只加载一次
_DATA_CACHE: dict = {}


def _get_windows(dataset_name: str, cfg: dict, allow_synthetic: bool):
    key = (dataset_name, cfg["window_length"], cfg["stride"],
           cfg["val_ratio"], cfg["test_ratio"], allow_synthetic)
    if key not in _DATA_CACHE:
        ds = load_dataset(dataset_name, allow_synthetic=allow_synthetic)
        sp = split_and_normalize(ds, cfg["val_ratio"], cfg["test_ratio"])
        w, stride = cfg["window_length"], cfg["stride"]
        _DATA_CACHE[key] = (
            sliding_windows(sp.train_normal, w, stride),
            sliding_windows(sp.val_normal, w, stride),
            sliding_windows(sp.val_mixed, w, stride),
            sliding_windows(sp.test, w, stride),
            sp.val_mixed_y, sp.test_y, sp.data_source,
        )
    return _DATA_CACHE[key]


def run_single(model_name: str, dataset_name: str, seed: int, cfg: dict,
               allow_synthetic: bool = False) -> dict:
    """训练 + 阈值搜索 + 测试评估 + 效率测量，返回一行结果。"""
    t_start = time.time()
    w, stride = cfg["window_length"], cfg["stride"]
    train_w, val_w, val_mix_w, test_w, val_y, test_y, data_source = _get_windows(
        dataset_name, cfg, allow_synthetic)
    if len(train_w) < 8:
        raise ValueError(f"训练窗口过少（{len(train_w)}），数据量不足以训练")

    model = get_model(model_name)
    model.fit(train_w, val_w, patience=cfg["early_stop_patience"],
              max_epochs=cfg["max_epochs"], seed=seed)
    # 阈值：验证集（含异常段）上分位网格搜索，PA-F1 最优
    s_val = model.score_windows(val_mix_w)
    s_val_pt = window_scores_to_point(s_val, w, stride, len(val_y))
    tau, _ = search_threshold(val_y, s_val_pt, cfg["threshold_search"]["quantile_grid"])
    # 测试集：同阈值双报告
    s_test = model.score_windows(test_w)
    s_test_pt = window_scores_to_point(s_test, w, stride, len(test_y))
    ev = evaluate(test_y, s_test_pt, tau)
    eff = efficiency(model, cfg["latency_windows"])
    epochs = getattr(model, "_epochs_run", "n/a")
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "protocol_version": str(cfg["protocol_version"]),
        "model": model_name, "dataset": dataset_name, "seed": seed,
        "data_source": data_source,
        **{k: ev.get(k) for k in ["precision", "recall", "f1", "precision_pa",
                                  "recall_pa", "f1_pa", "auc_roc", "auc_pr"]},
        "threshold": tau,
        **eff,
        "teacher_params": (model.teacher_params() if hasattr(model, "teacher_params") else ""),
        "train_seconds": round(time.time() - t_start, 1),
        "epochs_run": epochs, "retries": 0,
        "notes": "",
    }
    return row


def _job(args: tuple) -> tuple:
    """并行 worker：跑一个组合（重试 1 次），返回 ('ok', row) 或 ('fail', msg, tb, m, d, s)。"""
    m, d, s, cfg, allow_synthetic = args
    last_err = None
    for attempt in (0, 1):
        try:
            row = run_single(m, d, s, cfg, allow_synthetic=allow_synthetic)
            row["retries"] = attempt
            return ("ok", row, m, d, s, attempt)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt == 0:
                continue
    return ("fail", f"{last_err}", traceback.format_exc(), m, d, s, 1)


# 单任务超时（秒）。超时记 failed 并继续，避免 O(n²) 模型堵死整批。
JOB_TIMEOUT = float(os.environ.get("RUN_ALL_JOB_TIMEOUT", "10800"))  # 默认 3 小时


def fmt(v) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"          # 4 位小数：verify.py 字符串级对账的精度契约
    return str(v)


def append_row(row: dict) -> None:
    new_file = not RESULTS_CSV.exists()
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            wtr.writeheader()
        wtr.writerow({k: fmt(row.get(k, "")) for k in CSV_COLUMNS})


def existing_keys() -> set[tuple[str, str, int]]:
    """断点续跑的已完成集合。synthetic（smoke）行不计入——否则会把真实组合误跳。"""
    if not RESULTS_CSV.exists():
        return set()
    keys = set()
    for row in csv.DictReader(RESULTS_CSV.open(encoding="utf-8")):
        if row.get("data_source") == "synthetic":
            continue
        try:
            keys.add((row["model"], row["dataset"], int(row["seed"])))
        except (KeyError, ValueError):
            continue
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(description="链路 B：实验批处理（CP2）")
    ap.add_argument("--smoke", action="store_true",
                    help="管线自检：合成数据 + 全部模型 × 1 数据集 × 1 种子，不受协议锁限制")
    ap.add_argument("--models", type=str, default=None, help="逗号分隔的模型子集")
    ap.add_argument("--datasets", type=str, default=None, help="逗号分隔的数据集子集")
    ap.add_argument("--seeds", type=str, default=None, help="逗号分隔的种子子集")
    ap.add_argument("--allow-unlocked", action="store_true", help="协议未锁定时仍运行（调试）")
    ap.add_argument("--force", action="store_true", help="忽略断点续跑，重跑已存在组合")
    ap.add_argument("--workers", type=int, default=0,
                    help="并行 worker 数（默认 min(4, CPU 核数)，1=顺序）")
    args = ap.parse_args()

    cfg = proto.load_protocol()
    print(f"[run_all] protocol v{cfg['protocol_version']} locked={cfg.get('locked')}"
          f" smoke={args.smoke}")
    print("[run_all] CP2：确认本行日志正常后即可离开，批处理自动进行。")

    if args.smoke:
        cfg = dict(cfg)
        cfg["max_epochs"] = 2
        cfg["latency_windows"] = 10
        allow_synthetic = True
        models = cfg["models"]
        datasets = [cfg["datasets"][0]]
        seeds = [cfg["random_seeds"][0]]
    else:
        proto.assert_lock(cfg, allow_unlocked=args.allow_unlocked)
        allow_synthetic = False
        models = args.models.split(",") if args.models else cfg["models"]
        datasets = args.datasets.split(",") if args.datasets else cfg["datasets"]
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["random_seeds"]

    rpi = load_rpi_latencies(cfg)
    if not args.smoke and RESULTS_CSV.exists():
        with RESULTS_CSV.open(encoding="utf-8") as f:
            n_synth = sum(1 for row in csv.DictReader(f) if row.get("data_source") == "synthetic")
        if n_synth:
            print(f"[run_all] 警告：results.csv 含 {n_synth} 行 synthetic（smoke 残留）。"
                  "不会占用续跑名额，但 verify 会拦截；正式出稿前请清掉 results.csv。")
    done = set() if args.force else existing_keys()
    combos = [(m, d, s) for m in models for d in datasets for s in seeds]
    todo = [c for c in combos if c not in done]
    print(f"[run_all] 待运行 {len(todo)}/{len(combos)} 个组合"
          f"（跳过已完成 {len(combos) - len(todo)}）")

    ok, skipped = 0, 0
    t0 = time.time()
    total = len(todo)
    already = len(combos) - total
    workers = args.workers if args.workers > 0 else min(4, os.cpu_count() or 2)
    workers = max(1, min(workers, total or 1))

    # 调度：轻任务优先（msl/skab 先出数），避免 4 个 worker 同时卡死在 SMD 长任务上
    _DCOST = {"msl": 0, "skab": 1, "smd": 3}
    _MCOST = {"iforest": 0, "ocsvm": 0, "pca_recon": 1, "ae_mlp": 1,
              "deep_svdd": 2, "knn_dist": 2, "distill_ae": 3, "lstm_ad": 3}
    todo.sort(key=lambda c: (_DCOST.get(c[1], 2), _MCOST.get(c[0], 2), c[0], c[1], c[2]))

    print(f"[run_all] 进度条 = 总进度（含已完成 {already}）；Ctrl+C 可中断，续跑自动跳过。")
    print(f"[run_all] 并行 worker × {workers}；轻任务优先（msl/skab → smd）")
    if total:
        print(f"[run_all] 本场队列前 5：{' | '.join(f'{m}×{d}×{s}' for m, d, s in todo[:5])}\n")

    def _fmt_dur(sec: float) -> str:
        sec = max(0, int(sec))
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"

    def _bar(done_n: int) -> str:
        overall = already + done_n
        n_all = len(combos)
        width = 20
        filled = int(width * overall / n_all) if n_all else width
        pct = 100.0 * overall / n_all if n_all else 100.0
        return (f"[{'#'*filled}{'-'*(width-filled)}] "
                f"总 {overall}/{n_all} ({pct:5.1f}%)  本场 {done_n}/{total}")

    jobs = [(m, d, s, cfg, allow_synthetic) for m, d, s in todo]
    done_n = 0

    def _finish_ok(row, m, d, s, attempt):
        nonlocal ok, done_n
        if (m, d) in rpi:
            row["latency_ms_rpi"] = rpi[(m, d)]
        append_row(row)
        ok += 1
        done_n += 1
        elapsed = time.time() - t0
        eta = (elapsed / done_n) * (total - done_n)
        print(f"{_bar(done_n)}  ✓ {m} × {d} × seed{s}"
              f"  f1_pa={row['f1_pa']:.4f}  {row['train_seconds']}s"
              f"  [累计 {already + done_n}/{len(combos)}]"
              f"  耗时 {_fmt_dur(elapsed)}  ETA {_fmt_dur(eta)}"
              + (f"  retry={attempt}" if attempt else ""), flush=True)

    def _finish_fail(msg, tb, m, d, s):
        nonlocal skipped, done_n
        skipped += 1
        done_n += 1
        FAILED_LOG.parent.mkdir(exist_ok=True)
        with FAILED_LOG.open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%F %T} | {m} × {d} × seed{s} | 重试后仍失败 | {msg}\n{tb}\n")
        elapsed = time.time() - t0
        eta = (elapsed / done_n) * (total - done_n)
        print(f"{_bar(done_n)}  ✗ 跳过 {m}×{d}×{s}  ETA {_fmt_dur(eta)}  | {msg}", flush=True)

    if workers == 1:
        for i, (m, d, s) in enumerate(todo, 1):
            print(f"{_bar(i - 1)}  ▶ {m} × {d} × seed{s}  …", flush=True)
            kind, *rest = _job((m, d, s, cfg, allow_synthetic))
            if kind == "ok":
                _, row, _m, _d, _s, attempt = (kind, *rest)
                _finish_ok(row, _m, _d, _s, attempt)
            else:
                _, msg, tb, _m, _d, _s, _ = (kind, *rest)
                _finish_fail(msg, tb, _m, _d, _s)
    else:
        # 组合级并行：每个任务独立，结果按完成顺序落盘（断点续跑安全）
        import threading

        inflight: dict = {}   # future -> 标签
        stop_beat = threading.Event()

        def _heartbeat():
            while not stop_beat.wait(60):
                labels = list(inflight.values())
                elapsed = time.time() - t0
                print(f"    … 心跳 {_fmt_dur(elapsed)}  已完成 {done_n}/{total}"
                      f"  在跑 {len(labels)} 个: {' | '.join(labels[:6])}", flush=True)

        print(f"{_bar(0)}  ▶ 启动 {total} 个任务 × {workers} workers …", flush=True)
        beat = threading.Thread(target=_heartbeat, daemon=True)
        beat.start()
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {}
                fut_t0 = {}
                for j in jobs:
                    fut = ex.submit(_job, j)
                    futs[fut] = j
                    fut_t0[fut] = time.time()
                    inflight[fut] = f"{j[0]}×{j[1]}×s{j[2]}"
                pending = set(futs)
                while pending:
                    finished, pending = wait(pending, timeout=1.0,
                                             return_when=FIRST_COMPLETED)
                    now = time.time()
                    for fut in finished:
                        inflight.pop(fut, None)
                        try:
                            kind, *rest = fut.result(timeout=0)
                        except Exception as e:  # noqa: BLE001
                            j = futs[fut]
                            _finish_fail(str(e), traceback.format_exc(), j[0], j[1], j[2])
                            continue
                        if kind == "ok":
                            _, row, m, d, s, attempt = (kind, *rest)
                            _finish_ok(row, m, d, s, attempt)
                        else:
                            _, msg, tb, m, d, s, _ = (kind, *rest)
                            _finish_fail(msg, tb, m, d, s)
                    for fut in list(pending):
                        if now - fut_t0[fut] > JOB_TIMEOUT:
                            pending.discard(fut)
                            inflight.pop(fut, None)
                            fut.cancel()
                            j = futs[fut]
                            _finish_fail(f"单任务超时 {JOB_TIMEOUT:.0f}s（已跳过）",
                                         "", j[0], j[1], j[2])
        finally:
            stop_beat.set()

    mins = (time.time() - t0) / 60
    print(f"[run_all] 完成：成功 {ok}，跳过 {skipped}，耗时 {mins:.1f} 分钟。"
          f"结果：{RESULTS_CSV}")
    if skipped:
        print(f"[run_all] 失败明细见 {FAILED_LOG}")
    return 0 if (ok or not todo) else 1


if __name__ == "__main__":
    raise SystemExit(main())
