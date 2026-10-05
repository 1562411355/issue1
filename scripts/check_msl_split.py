# -*- coding: utf-8 -*-
"""校验 MSL 划分：val/test 必须都含正常与异常。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data import load_dataset, split_and_normalize

sp = split_and_normalize(load_dataset("msl"), 0.2, 0.2)
v = np.unique(sp.val_mixed_y, return_counts=True)
t = np.unique(sp.test_y, return_counts=True)
print("val_mixed", v)
print("test", t)
if len(np.unique(sp.test_y)) < 2:
    raise SystemExit("FAIL: test set is single-class")
if len(np.unique(sp.val_mixed_y)) < 2:
    raise SystemExit("FAIL: val_mixed is single-class")
print("MSL split OK")
