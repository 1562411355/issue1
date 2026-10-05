# -*- coding: utf-8 -*-
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
log = open(root / "results" / "usad_fill.log", "ab")
err = open(root / "results" / "usad_fill.err", "ab")
p = subprocess.Popen(
    [sys.executable, "run_all.py", "--models", "usad", "--datasets", "smd", "--workers", "3"],
    cwd=str(root),
    stdout=log,
    stderr=err,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
)
print(f"launched usad fill pid={p.pid}")
