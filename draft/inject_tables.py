#!/usr/bin/env python3
"""链路 D：数字注入器（设计书 6.4：正文数字一律由它从 results.csv 注入）。

- 渲染 draft/tables/ 下的论文表格（markdown + 主表 LaTeX 版）
- 扫描 draft/sections/*.md 中的 `{{table:名称}}` 占位并替换为真实表格
- 每张表带「来源行」：数据文件、协议版本、生成时间——verify.py 据此对账
- 事实契约：AI 在表格之间写叙述，但**任何数字都必须来自这些表格**，
  手输数字会被 verify.py 抓出（4 位小数精度对账）。
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO / "results" / "results.csv"
TABLES_DIR = REPO / "draft" / "tables"
SECTIONS_DIR = REPO / "draft" / "sections"


def _f(x, prec=4):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a"
    return f"{x:.{prec}f}"


def _protocol_version(df: pd.DataFrame) -> str:
    """版本戳：优先 configs/protocol.md 首行，否则取 csv 最大 protocol_version。"""
    proto = REPO / "configs" / "protocol.md"
    if proto.exists():
        m = re.search(r"v(\d+\.\d+)", proto.read_text(encoding="utf-8").splitlines()[0])
        if m:
            return m.group(1)
    if len(df):
        return str(df.protocol_version.max())
    return "?"


def provenance(df: pd.DataFrame) -> str:
    v = _protocol_version(df)
    return (f"\n> 来源：`results/results.csv`（协议 v{v}）· 本表由 `draft/inject_tables.py` "
            f"于 {datetime.now():%Y-%m-%d %H:%M} 自动生成 · 修改数字 = 伪造数据\n")


def load_df() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        sys.exit("[inject] 未找到 results/results.csv，请先运行 run_all.py")
    df = pd.read_csv(RESULTS_CSV).drop_duplicates(subset=["model", "dataset", "seed"], keep="last")
    if (df["data_source"] == "synthetic").any():
        print("[inject] 警告：results.csv 含合成数据（smoke 产物）。"
              "表格将带 SYNTHETIC 水印，正式论文前必须用真实数据重跑。")
    return df


def table_main(df: pd.DataFrame) -> str:
    g = df.groupby("model")
    rows = []
    for m, r in df.groupby("model"):
        rows.append({
            "model": m, "runs": len(r),
            "f1_pa": r.f1_pa.mean(), "f1_pa_std": r.f1_pa.std(ddof=1),
            "f1": r.f1.mean(), "f1_std": r.f1.std(ddof=1),
            "auc_pr": r.auc_pr.mean(), "auc_roc": r.auc_roc.mean(),
        })
    lb = pd.DataFrame(rows).sort_values("f1_pa", ascending=False)
    synth = (df["data_source"] == "synthetic").any()
    head = "### 表 1：主结果（种子均值 ± 标准差）" + (" **[SYNTHETIC-禁入论文]**" if synth else "") + "\n\n"
    L = [head,
         "| 模型 | F1-PA | F1（非 PA） | AUC-PR | AUC-ROC |",
         "|---|---|---|---|---|"]
    for _, r in lb.iterrows():
        s1 = f"±{_f(r.f1_pa_std, 3)}" if not np.isnan(r.f1_pa_std) else ""
        s2 = f"±{_f(r.f1_std, 3)}" if not np.isnan(r.f1_std) else ""
        L.append(f"| `{r.model}` | {_f(r.f1_pa)} {s1} | {_f(r.f1)} {s2} | {_f(r.auc_pr)} | {_f(r.auc_roc)} |")
    L.append(provenance(df))
    return "\n".join(L)


def table_main_tex(df: pd.DataFrame) -> str:
    lb = df.groupby("model").agg(f1_pa=("f1_pa", "mean"), f1=("f1", "mean"),
                                 auc_pr=("auc_pr", "mean")).sort_values("f1_pa", ascending=False)
    L = ["% Auto-generated from results/results.csv by draft/inject_tables.py",
         r"\begin{table}[t]", r"\centering",
         r"\caption{Main results (mean over seeds).}",
         r"\label{tab:main}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"Model & F1$_{PA}$ & F1 & AUC-PR \\",
         r"\midrule & \\[-6pt]"]
    for name, r in lb.iterrows():
        L.append(f"{name.replace('_', '\\_')} & {_f(r.f1_pa, 2)} & {_f(r.f1, 2)} & {_f(r.auc_pr, 2)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def table_efficiency(df: pd.DataFrame) -> str:
    rows = []
    for m, r in df.groupby("model"):
        tp = pd.to_numeric(r["teacher_params"], errors="coerce").dropna()
        rows.append({
            "model": m, "params": int(r.params.mean()),
            "kb": r.model_kb.mean(), "lat_cpu": r.latency_ms_cpu.median(),
            "lat_rpi": (r.latency_ms_rpi.dropna().median() if r.latency_ms_rpi.notna().any() else np.nan),
            "teacher": (int(tp.iloc[0]) if len(tp) else ""),
        })
    eff = pd.DataFrame(rows).sort_values("params")
    L = ["### 表 2：效率与轻量化（参数量 / 体积 / 单窗口延迟）\n",
         "| 模型 | 参数量 | 体积(KB) | CPU延迟(ms) | 树莓派延迟(ms) | 教师参数 |",
         "|---|---|---|---|---|---|"]
    for _, r in eff.iterrows():
        rpi = _f(r.lat_rpi, 3) if not (isinstance(r.lat_rpi, float) and np.isnan(r.lat_rpi)) else "（未实测）"
        L.append(f"| `{r.model}` | {r.params:,} | {_f(r.kb, 1)} | {_f(r.lat_cpu, 3)} | {rpi} | "
                 + (f"{r.teacher:,}" if r.teacher != "" else "—") + " |")
    L.append("\n> 注：树莓派延迟只能来自 `results/rpi_latency.csv` 真机实测（见 `scripts/rpi_latency.md`）；"
             "空 = 未实测，论文中不得给出估算值。")
    L.append(provenance(df))
    return "\n".join(L)


def table_datasets(df: pd.DataFrame) -> str:
    piv = df.pivot_table(index="model", columns="dataset", values="f1_pa", aggfunc="mean")
    piv["mean"] = piv.mean(axis=1)
    piv = piv.sort_values("mean", ascending=False)
    L = ["### 表 3：各数据集 F1-PA 明细\n",
         "| 模型 | " + " | ".join(str(c) for c in piv.columns) + " |",
         "|---" * (len(piv.columns) + 1) + "|"]
    for name, r in piv.iterrows():
        L.append(f"| `{name}` | " + " | ".join(_f(v) for v in r.values) + " |")
    L.append(provenance(df))
    return "\n".join(L)


def substitute_in_sections(tables: dict[str, str]) -> int:
    """把 sections 中的 {{table:名称}} 替换为渲染后的表格。

    注入结果包在 <!--inject:名称-->…<!--/inject:名称--> 里，再次运行时
    会整块刷新——否则占位符被替换掉后旧数字会永久冻在草稿中。
    """
    n = 0
    for f in sorted(SECTIONS_DIR.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        for key, body in tables.items():
            wrapped = f"<!--inject:{key}-->\n{body.strip()}\n<!--/inject:{key}-->"
            marker = "{{table:" + key + "}}"
            if marker in text:
                text = text.replace(marker, wrapped)
                n += 1
                continue
            text, k = re.subn(
                rf"<!--inject:{re.escape(key)}-->.*?<!--/inject:{re.escape(key)}-->",
                wrapped, text, flags=re.S,
            )
            n += k
        f.write_text(text, encoding="utf-8")
    return n


def main() -> int:
    df = load_df()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "table_main": table_main(df),
        "table_efficiency": table_efficiency(df),
        "table_datasets": table_datasets(df),
    }
    for name, body in tables.items():
        (TABLES_DIR / f"{name}.md").write_text(body, encoding="utf-8")
        print(f"[inject] draft/tables/{name}.md")
    (TABLES_DIR / "table_main.tex").write_text(table_main_tex(df), encoding="utf-8")
    print("[inject] draft/tables/table_main.tex")
    # 派生数值清单：本表渲染的全部数字（多种子均值等派生量）逐个登记，
    # verify.py 据此放行；手改表格数字将不在清单中 → 对账 FAIL。
    derived = set()
    for body in list(tables.values()) + [table_main_tex(df)]:
        for m in re.finditer(r"\d+\.\d+|\d{1,3}(?:,\d{3})+|\b\d+\b", body):
            derived.add(m.group().replace(",", ""))
    (TABLES_DIR / "derived_numbers.txt").write_text(
        "\n".join(sorted(derived, key=float)) + "\n", encoding="utf-8")
    n = substitute_in_sections(tables)
    print(f"[inject] 已向草稿小节注入 {n} 处表格；AI 叙述只在表格之间写，不得改动数字。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
