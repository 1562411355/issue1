# -*- coding: utf-8 -*-
"""Detached launch: 56 ablation runs + analyze/verify."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
log = open(root / "results" / "ablation.log", "ab")
err = open(root / "results" / "ablation.err", "ab")

cmd = (
    "cd /d " + str(root).replace("&", "^&") +
    " && .venv\\Scripts\\python.exe run_all.py --models ae_s_nodistill,ae_l_teacher "
    "--datasets msl,skab,smd --workers 4"
    " && .venv\\Scripts\\python.exe run_all.py --models distill_ae_lam01,distill_ae_lam05 "
    "--datasets skab --workers 4"
    " && .venv\\Scripts\\python.exe analysis\\analyze.py"
    " && .venv\\Scripts\\python.exe verify\\verify.py"
)
p = subprocess.Popen(
    ["cmd", "/c", cmd],
    cwd=str(root),
    stdout=log,
    stderr=err,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
)
print(f"launched ablation pid={p.pid}")
