# -*- coding: utf-8 -*-
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
log = open(root / "results" / "deep_baselines.log", "ab")
err = open(root / "results" / "deep_baselines.err", "ab")
cmd = (
    f'cd /d "{root}" && '
    ".venv\\Scripts\\python.exe run_all.py --models tranad,usad --datasets msl,skab,smd --workers 4 && "
    ".venv\\Scripts\\python.exe analysis\\analyze.py && "
    ".venv\\Scripts\\python.exe verify\\verify.py"
)
p = subprocess.Popen(
    ["cmd", "/c", cmd],
    stdout=log,
    stderr=err,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
)
print(f"launched deep baselines pid={p.pid}")
