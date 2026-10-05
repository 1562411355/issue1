# 树莓派真机延迟测量流程（人工执行，约 30 分钟/轮）

verify.py 会拦截一切没有真机记录的树莓派延迟数字。流程：

1. 把模型序列化文件（results/<model>_<dataset>_s<seed>.npz 运行时产物）
   与 `scripts/rpi_benchmark.py`（待你按型号编写，模板见下）拷到树莓派。
2. 每模型 × 数据集测 200 个单窗口前向，取中位数（ms）。
3. 结果写入 `results/rpi_latency.csv`：

```csv
model,dataset,latency_ms_rpi
ae_mlp,skab,3.21
distill_ae,skab,0.85
```

4. 重跑 `python3 run_all.py --rerun` 或手工合并——run_all.py 启动时会
   自动 join 该文件到 results.csv 的 latency_ms_rpi 列。

## rpi_benchmark.py 模板（按树莓派型号补全）

import csv, time, numpy as np
from src.models import get_model
# 1. 载入序列化模型；2. 生成同形状随机窗口；3. 计时 200 次取中位数
# （单窗口前向，数据准备不计入——与 src/metrics.efficiency 相同口径）
