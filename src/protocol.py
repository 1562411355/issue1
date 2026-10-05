"""protocol.md 解析器：读取 YAML 机读区，执行 CP1 锁定约束。

设计书第 2.2 节「机器管机器」：锁定状态由机器强制，不靠自觉。
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = REPO_ROOT / "configs" / "protocol.md"


class ProtocolNotLockedError(RuntimeError):
    """协议未锁定即尝试全量运行。"""


def load_protocol(path: Path = PROTOCOL_PATH) -> dict:
    """从 protocol.md 提取 ```yaml ... ``` 代码块并解析。"""
    text = path.read_text(encoding="utf-8")
    m = re.search(r"```yaml\n(.*?)```", text, re.DOTALL)
    if not m:
        raise ValueError(f"{path} 中未找到 yaml 机读区")
    cfg = yaml.safe_load(m.group(1))
    if not isinstance(cfg, dict) or "models" not in cfg:
        raise ValueError(f"{path} 的 yaml 机读区不完整")
    return cfg


def assert_lock(cfg: dict, allow_unlocked: bool = False) -> None:
    """全量运行前置检查：协议必须已锁定（CP1 批准）。"""
    if cfg.get("locked", False):
        return
    if allow_unlocked:
        print("[protocol] 警告：协议未锁定，按 --allow-unlocked 继续（仅限调试）")
        return
    raise ProtocolNotLockedError(
        "configs/protocol.md 尚未锁定（locked: false）。"
        "请先完成 CP1：逐条审阅协议，批准后把 locked 改为 true 并签名。"
        "管线自检请用 run_all.py --smoke。"
    )


def grid(cfg: dict) -> list[tuple[str, str, int]]:
    """生成 (model, dataset, seed) 全组合网格。"""
    return [
        (m, d, s)
        for m in cfg["models"]
        for d in cfg["datasets"]
        for s in cfg["random_seeds"]
    ]
