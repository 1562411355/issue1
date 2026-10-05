#!/usr/bin/env python3
"""链路 E：校验闸门（设计书 7：投稿包链 F 启动前的最后一道机器闸门）。

检查项（FAIL 拦截 / WARN 提醒）：
1. 数字对账：草稿（sections + tables）中的全部实验数字必须能在
   results.csv / analysis/report.md / configs/protocol.md 的数值集合中
   找到（4 位小数精度容差）。手输/编造数字 → FAIL。
2. 合成数据拦截：results.csv 任一行 data_source=synthetic → FAIL（论文禁用）。
3. 【待填】/【叙述 残留 → --final 模式 FAIL（投稿前必须清零），默认 WARN。
4. 引用核对：正文 [n] 引用与 references.md 条目互相缺失 → WARN；
   条目残缺【待核】→ WARN。
5. Results 段落缺依据行（> 依据：）→ WARN。
6. 树莓派延迟：正文出现树莓派延迟数字但 results/rpi_latency.csv 缺失
   或不含该值 → FAIL（防编造真机数据）。

用法：
  python3 verify/verify.py            # 日常（CP3 数据抽查配套）
  python3 verify/verify.py --final    # CP4 终审 / 投稿前（残留即拦截）
退出码：0 通过（允许 WARN）；1 拦截。pipeline.py 据此阻止链路 F。
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import csv as _csv
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO / "results" / "results.csv"
RPI_CSV = REPO / "results" / "rpi_latency.csv"
SECTIONS = REPO / "draft" / "sections"
TABLES = REPO / "draft" / "tables"
REPORT = REPO / "analysis" / "report.md"
PROTOCOL = REPO / "configs" / "protocol.md"
OUT = REPO / "verify" / "verify_report.txt"

TOL = 5.1e-5          # 4 位小数量化半步
SKIP_LINE = re.compile(r"^(#|\>|$|\|[-: ]+\|?\s*$|\[\d+\])")


def collect_allowed_numbers() -> tuple[set[float], set[int], set[float]]:
    """机器产物数值全集：csv + report + protocol（含树莓派真机表）。"""
    floats: set[float] = set()
    ints: set[int] = set()
    rpi: set[float] = set()

    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
        for c in df.columns:
            for v in pd.to_numeric(df[c], errors="coerce").dropna():
                if abs(v - round(v)) < 1e-9 and abs(v) >= 1:
                    ints.add(int(round(v)))
                floats.add(float(v))
    for f in (REPORT, PROTOCOL):
        if f.exists():
            for m in re.finditer(r"\d+\.\d+|\b\d{2,}\b", f.read_text(encoding="utf-8")):
                floats.add(float(m.group()))
    derived = REPO / "draft" / "tables" / "derived_numbers.txt"
    if derived.exists():  # 注入器登记的派生量（均值/标准差等）
        for m in re.finditer(r"\d+\.\d+|\b\d+\b", derived.read_text(encoding="utf-8")):
            v = float(m.group())
            floats.add(v)
            if abs(v - round(v)) < 1e-9 and abs(v) >= 1:
                ints.add(int(round(v)))
    if RPI_CSV.exists():
        with RPI_CSV.open(encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                try:
                    rpi.add(float(row["latency_ms_rpi"]))
                except (KeyError, ValueError):
                    pass
    return floats, ints, rpi


def extract_draft_numbers():
    """逐文件逐行抽取待检数字：文本行中的浮点 + 带量纲整数。

    WRITING_RULES.md 是写作规范文档（非论文内容），不参与扫描。
    """
    out = []  # (file, line_no, kind, value, line)
    files = sorted(SECTIONS.glob("*.md")) + sorted(TABLES.glob("*.md"))
    files = [f for f in files if f.name != "WRITING_RULES.md"]
    for f in files:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if SKIP_LINE.match(line):
                continue
            if re.search(r"树莓派|Raspberry|raspberry", line, re.I):
                # 只检十进制数值（如 3.2、0.05 ms）；列表编号等裸整数不算延迟
                for m in re.finditer(r"\d+\.\d+", line):
                    out.append((f, i, "rpi", float(m.group()), line))
                continue
            for m in re.finditer(r"\d+\.\d+", line):
                out.append((f, i, "float", float(m.group()), line))
            for m in re.finditer(r"(\d[\d,]*)\s*(ms|KB|MB|参数|params)", line, re.I):
                out.append((f, i, "int", float(m.group(1).replace(",", "")), line))
            for m in re.finditer(r"\b\d{1,3}(?:,\d{3})+\b", line):
                out.append((f, i, "int", float(m.group().replace(",", "")), line))
    return out


def check_numbers(nums, allowed_f, allowed_i, rpi_vals):
    fails = []
    for f, i, kind, v, line in nums:
        if kind == "rpi":
            if not any(abs(v - r) <= TOL for r in rpi_vals):
                fails.append((f, i, v, line, "树莓派数字无真机记录"))
            continue
        ok = any(abs(v - a) <= TOL for a in allowed_f) or (
            kind == "int" and any(abs(v - a) < 0.5 for a in allowed_i))
        if not ok:
            fails.append((f, i, v, line, "未在 csv/report/protocol 中找到"))
    return fails


def check_references():
    warns = []
    refs = {}
    ref_file = SECTIONS / "references.md"
    text_files = [f for f in sorted(SECTIONS.glob("*.md"))
                  if f.name not in ("references.md", "WRITING_RULES.md")]
    if ref_file.exists():
        for m in re.finditer(r"^\[(\d+)\]\s*(.+)$", ref_file.read_text(encoding="utf-8"), re.M):
            refs[int(m.group(1))] = m.group(2).strip()
    else:
        warns.append("references.md 缺失")
    cited = set()
    for f in text_files:
        for m in re.finditer(r"\[(\d+(?:,\s*\d+)*)\]", f.read_text(encoding="utf-8")):
            cited.update(int(x) for x in m.group(1).split(","))
    for n in sorted(cited - set(refs)):
        warns.append(f"正文引用 [{n}] 在 references.md 无条目")
    for n in sorted(set(refs) - cited):
        warns.append(f"条目 [{n}] 未被正文引用")
    for n, e in refs.items():
        if "【待核" in e or "【待填" in e:
            warns.append(f"条目 [{n}] 未核实：【待核】残留")
        elif not re.search(r"\b(19|20)\d{2}\b", e):
            warns.append(f"条目 [{n}] 疑似缺年份/残缺")
    return warns, len(cited), len(refs)


def check_evidence_lines():
    """Results 小节：长段落应以 > 依据： 收尾（写作规范 2）。"""
    warns = []
    f = SECTIONS / "04-experiments.md"
    if not f.exists():
        return warns
    paras, cur = [], []
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cur.append(line)
        else:
            if cur:
                paras.append(cur)
                cur = []
    if cur:
        paras.append(cur)
    for p in paras:
        body = "\n".join(p)
        if len(body) > 120 and not re.match(r"^#|\|", p[0]):
            if not any(l.lstrip("> ").startswith("依据") for l in p):
                warns.append(f"04-experiments.md 段落缺依据行：「{p[0][:30]}…」")
    return warns


def check_placeholders(final_mode: bool):
    issues = []
    files = [f for f in sorted(SECTIONS.glob("*.md")) if f.name != "WRITING_RULES.md"]
    for f in files:
        text = f.read_text(encoding="utf-8")
        for token in ("【待填", "【待核", "【叙述"):
            n = text.count(token)
            if n:
                issues.append((final_mode, f"{f.name} 残留 {token}…】 ×{n}"))
    return issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="CP4/投稿前模式：残留即 FAIL")
    args = ap.parse_args()

    L = [f"verify.py 校验报告 — {datetime.now():%Y-%m-%d %H:%M:%S}"
         + ("（--final 投稿前模式）" if args.final else "") + "\n"]
    fails, warns = [], []

    if not RESULTS_CSV.exists():
        print("[verify] 未找到 results/results.csv")
        return 1

    # 1. 合成数据拦截
    df = pd.read_csv(RESULTS_CSV)
    n_synth = int((df["data_source"] == "synthetic").sum())
    if n_synth:
        fails.append(f"results.csv 含 {n_synth} 行 synthetic 数据（--smoke 产物）。"
                     "论文实验必须真实数据真实运行：放入 data/ 后重跑 run_all.py。")

    # 2. 数字对账
    allowed_f, allowed_i, rpi_vals = collect_allowed_numbers()
    nums = extract_draft_numbers()
    bad = check_numbers(nums, allowed_f, allowed_i, rpi_vals)
    L.append(f"1. 数字对账：抽取 {len(nums)} 个数字（sections+tables），"
             f"容差 {TOL:.0e}，未对上 {len(bad)} 个。")
    for f, i, v, line, why in bad[:30]:
        fails.append(f"数字对账失败 {f.name}:{i} 值={v}（{why}）：{line.strip()[:70]}")
        L.append(f"   ✗ {f.name}:{i}  {v}  {why}")
    if not bad:
        L.append("   ✓ 全部数字可追溯（csv / report / protocol）。")

    # 3. 树莓派
    if RPI_CSV.exists():
        L.append(f"2. 树莓派延迟：results/rpi_latency.csv 存在（{len(rpi_vals)} 条真机记录）。")
    else:
        L.append("2. 树莓派延迟：无真机记录文件（正文不得出现树莓派延迟数字）。")

    # 4. 占位符
    ph = check_placeholders(args.final)
    L.append(f"3. 占位符残留：{len(ph)} 处。")
    for final_mode, msg in ph:
        (fails if final_mode else warns).append(msg)
        L.append(f"   {'✗' if final_mode else '⚠'} {msg}")

    # 5. 引用
    ref_warns, n_cited, n_refs = check_references()
    L.append(f"4. 引用核对：正文引用 {n_cited} 个编号，条目 {n_refs} 条。")
    warns += ref_warns
    for w in ref_warns:
        L.append(f"   ⚠ {w}")

    # 6. 依据行
    ev = check_evidence_lines()
    L.append(f"5. Results 依据行：缺失 {len(ev)} 段。")
    warns += ev
    for w in ev:
        L.append(f"   ⚠ {w}")

    L.append("")
    L.append(f"结论：FAIL {len(fails)} 项，WARN {len(warns)} 项 → "
             + ("闸门拦截（链路 F 不得启动）" if fails else "放行（含警告）"))
    for msg in fails:
        L.append(f"[FAIL] {msg}")
    for msg in warns:
        L.append(f"[WARN] {msg}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:14]))
    if len(L) > 14:
        print(f"…（完整报告：{OUT}）")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
