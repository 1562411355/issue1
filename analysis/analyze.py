#!/usr/bin/env python3
"""链路 C：自动分析（设计书 5.3：B 一完成它就接手，你只需要看）。

读取 results/results.csv，输出 analysis/report.md + 图表：
- 排行榜（种子均值±标准差，95% CI，t 分布）
- 显著性检验：各模型 vs 基线 pca_recon 的配对 Wilcoxon 符号秩检验
  （配对单元 = 数据集×种子），Holm 校正，α=0.05
- 轻量化/蒸馏对比（参数量与延迟压缩比）
- 机器复核：合成数据拦截、网格完整性、种子方差异常
图表使用英文标签（规避 matplotlib 中文字体缺字问题，亦符合期刊惯例）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO / "results" / "results.csv"
OUT_DIR = REPO / "analysis"
BASELINE = "pca_recon"
ALPHA = 0.05


def _ms(x, prec=4) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a"
    return f"{x:.{prec}f}"


def load_df() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        sys.exit(f"[analyze] 未找到 {RESULTS_CSV}，请先运行 run_all.py")
    df = pd.read_csv(RESULTS_CSV)
    df = df.drop_duplicates(subset=["model", "dataset", "seed"], keep="last")
    return df


def leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("model")
    lb = pd.DataFrame({
        "f1_pa_mean": g["f1_pa"].mean(), "f1_pa_std": g["f1_pa"].std(ddof=1),
        "f1_mean": g["f1"].mean(), "f1_std": g["f1"].std(ddof=1),
        "auc_pr_mean": g["auc_pr"].mean(), "auc_pr_std": g["auc_pr"].std(ddof=1),
        "auc_roc_mean": g["auc_roc"].mean(),
        "params_mean": g["params"].mean(),
        "latency_ms": g["latency_ms_cpu"].median(),
        "runs": g.size(),
    })
    n = lb["runs"].iloc[0] if len(lb) else 0
    lb["ci95_half"] = stats.t.ppf(0.975, df=max(n - 1, 1)) * lb["f1_pa_std"] / np.sqrt(np.maximum(lb["runs"], 1))
    return lb.sort_values("f1_pa_mean", ascending=False)


def significance(df: pd.DataFrame) -> pd.DataFrame:
    """配对 Wilcoxon：每个模型 vs 基线，配对单元 = (dataset, seed)。"""
    base = df[df.model == BASELINE].set_index(["dataset", "seed"])["f1_pa"]
    rows = []
    for m in sorted(df.model.unique()):
        if m == BASELINE:
            continue
        cur = df[df.model == m].set_index(["dataset", "seed"])["f1_pa"]
        common = base.index.intersection(cur.index)
        if len(common) < 5:
            rows.append({"model": m, "n_pairs": len(common), "p_raw": np.nan,
                         "p_holm": np.nan, "significant": "", "median_delta": np.nan})
            continue
        d = (cur.loc[common] - base.loc[common]).values
        try:
            p = stats.wilcoxon(d).pvalue
        except ValueError:
            p = 1.0
        rows.append({"model": m, "n_pairs": len(common), "p_raw": p,
                     "p_holm": np.nan, "significant": "", "median_delta": float(np.median(d))})
    # Holm 校正
    sig = pd.DataFrame(rows)
    valid = sig["p_raw"].notna()
    m = int(valid.sum())
    order = sig.loc[valid, "p_raw"].sort_values().index
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * float(sig.at[idx, "p_raw"]))
        adj = max(adj, prev)  # 单调性
        prev = adj
        sig.at[idx, "p_holm"] = adj
        sig.at[idx, "significant"] = "yes" if adj < ALPHA else "no"
    return sig


def charts(lb: pd.DataFrame) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[analyze] matplotlib 不可用，跳过图表：{e}")
        return []
    figdir = OUT_DIR / "figs"
    figdir.mkdir(exist_ok=True)
    made = []
    # 1) F1-PA 排行榜（误差棒 = 种子标准差）
    fig, ax = plt.subplots(figsize=(7, 4.2))
    order = lb.sort_values("f1_pa_mean")
    ax.barh(order.index, order["f1_pa_mean"], xerr=order["f1_pa_std"],
            color="#7fa8c9", edgecolor="#2c5d8f")
    ax.set_xlabel("F1 (PA), mean over seeds")
    ax.set_xlim(0, 1)
    ax.set_title("Leaderboard (higher = better)")
    fig.tight_layout()
    p1 = figdir / "fig_leaderboard_f1pa.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    made.append(p1)
    # 2) 参数量 vs F1-PA（轻量化权衡图）
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(lb["params_mean"], lb["f1_pa_mean"], s=70, color="#c96f7f", edgecolor="#8f2c44")
    for name, r in lb.iterrows():
        ax.annotate(name, (r["params_mean"], r["f1_pa_mean"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale, lower = lighter)")
    ax.set_ylabel("F1 (PA)")
    ax.set_title("Accuracy vs. Model Size")
    fig.tight_layout()
    p2 = figdir / "fig_params_vs_f1.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    made.append(p2)
    return made


def main() -> int:
    df = load_df()
    lb = leaderboard(df)
    sig = significance(df)
    figs = charts(lb)

    warn = []
    if (df["data_source"] == "synthetic").any():
        warn.append("results.csv 含 **synthetic** 数据（--smoke 产物）。此表**不可用于论文**，仅验证管线。")
    expect = df.groupby("model")["seed"].nunique()
    if (expect < 3).any():  # 协议要求 3 种子
        bad = expect[expect < 3]
        warn.append("种子数不足 3 的模型：" + "、".join(f"{m}({n})" for m, n in bad.items()))
    hi_std = lb[lb["f1_pa_std"] > 0.10]
    if len(hi_std):
        warn.append("种子方差异常（std>0.10）：" + "、".join(hi_std.index))

    L: list[str] = []
    L.append("# 实验分析报告（链路 C 自动生成）\n")
    L.append(f"- 数据源：`results/results.csv`（{len(df)} 行，{df.model.nunique()} 模型 × "
             f"{df.dataset.nunique()} 数据集）")
    L.append(f"- 协议版本：{df.protocol_version.iloc[0] if len(df) else '?'}；基线：`{BASELINE}`\n")

    if warn:
        L.append("## ⚠ 机器复核警告\n")
        for w in warn:
            L.append(f"- {w}")
        L.append("")

    L.append("## 1. 总排行榜（种子均值 ± 标准差 [95% CI]）\n")
    L.append("| 模型 | F1-PA | F1(非PA) | AUC-PR | AUC-ROC | 参数量 | CPU延迟(ms) | 运行数 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, r in lb.iterrows():
        ci = f"±{_ms(r.f1_pa_std)}" if not np.isnan(r.f1_pa_std) else ""
        L.append(f"| `{name}` | **{_ms(r.f1_pa_mean)}** {ci} [±{_ms(r.ci95_half)}] "
                 f"| {_ms(r.f1_mean)} ± {_ms(r.f1_std)} | {_ms(r.auc_pr_mean)} ± {_ms(r.auc_pr_std)} "
                 f"| {_ms(r.auc_roc_mean)} | {int(r.params_mean):,} | {_ms(r.latency_ms, 3)} | {int(r.runs)} |")
    L.append("")

    L.append("## 2. 各数据集明细（F1-PA 均值）\n")
    piv = df.pivot_table(index="model", columns="dataset", values="f1_pa", aggfunc="mean")
    L.append("| 模型 | " + " | ".join(piv.columns) + " | 均值 |")
    L.append("|---" * (len(piv.columns) + 2) + "|")
    for name, r in piv.iterrows():
        vals = " | ".join(_ms(v) for v in r.values)
        L.append(f"| `{name}` | {vals} | {_ms(float(np.nanmean(r.values)))} |")
    L.append("")

    L.append("## 3. 显著性检验（vs 基线 `pca_recon`，配对 Wilcoxon + Holm）\n")
    L.append("| 模型 | 配对数 | 中位ΔF1-PA | p(raw) | p(Holm) | 显著(α=0.05) |")
    L.append("|---|---|---|---|---|---|")
    for _, r in sig.iterrows():
        p_r = _ms(r.p_raw, 3) if not np.isnan(r.p_raw) else "n/a"
        p_h = _ms(r.p_holm, 3) if not np.isnan(r.p_holm) else "n/a"
        d = _ms(r.median_delta) if not np.isnan(r.median_delta) else "n/a"
        L.append(f"| `{r.model}` | {r.n_pairs} | {d} | {p_r} | {p_h} | {r.significant or '样本不足'} |")
    L.append("")

    L.append("## 4. 轻量化与蒸馏对比\n")
    try:
        stu = lb.loc["distill_ae"]
        rows = []
        for m in ["ae_mlp", "lstm_ad", "deep_svdd", "ocsvm", "knn_dist"]:
            if m in lb.index:
                rows.append((m, lb.loc[m]))
        tp = pd.to_numeric(df[df.model == "distill_ae"]["teacher_params"], errors="coerce").dropna()
        tpar = int(tp.iloc[0]) if len(tp) else None
        L.append(f"- 学生模型 `distill_ae`：{int(stu.params_mean):,} 参数，"
                 f"F1-PA {_ms(stu.f1_pa_mean)}，CPU 延迟 {_ms(stu.latency_ms, 3)} ms。")
        if tpar:
            L.append(f"- 其教师 AE-L：{tpar:,} 参数 → 压缩到 **{tpar / max(stu.params_mean, 1):.1f}×** 更小。")
        for m, r in rows:
            ratio = r.params_mean / max(stu.params_mean, 1)
            size_txt = (f"学生参数量为它的 1/{ratio:.1f}" if ratio >= 1
                        else f"学生参数量是它的 {1 / ratio:.1f}×（更大）")
            L.append(f"- vs `{m}`（{int(r.params_mean):,} 参数）："
                     f"{size_txt}，"
                     f"延迟比 {_ms(r.latency_ms / max(stu.latency_ms, 1e-9), 1)}×，"
                     f"F1-PA 差 {_ms(r.f1_pa_mean - stu.f1_pa_mean)}。")
    except KeyError:
        L.append("- 本次运行未包含 `distill_ae`。")
    L.append("")

    if figs:
        L.append("## 5. 图表\n")
        for f in figs:
            L.append(f"- `analysis/figs/{f.name}`")
        L.append("")

    L.append("## 6. 下一步（自动衔接）\n")
    L.append("- 链路 D：`python3 draft/inject_tables.py` 将排行榜渲染进草稿表格。")
    L.append("- 链路 E：`python3 verify/verify.py` 对草稿做数字对账与引用检查。")
    L.append("- 一键串联：`python3 pipeline.py --stage all`。\n")

    out = OUT_DIR / "report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"[analyze] 报告已生成：{out}")
    if figs:
        print("[analyze] 图表：", ", ".join(str(f) for f in figs))
    for w in warn:
        print("[analyze][警告]", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
