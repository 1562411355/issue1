# -*- coding: utf-8 -*-
"""以分离进程启动 run_all.py，本脚本立刻退出。"""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
log = open(root / "results" / "run_all.log", "ab")
err = open(root / "results" / "run_all.err", "ab")
p = subprocess.Popen(
    [sys.executable, "run_all.py"],
    cwd=str(root),
    stdout=log,
    stderr=err,
    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
    close_fds=True,
)
print(f"launched run_all.py pid={p.pid}")
