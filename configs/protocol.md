# 实验协议（Protocol）v0.10（已锁定）

> **状态：已锁定（CP1 完成，2026-09-30；v0.7 为 OCSVM/kNN 子采样可行性修复）**。改动任何
> 数值必须升版本号并重新批准。`run_all.py --smoke` 不受锁限制（仅用于管线自检）。
>
> 版本历史：v0.1 → v0.2 CP1 预演 → v0.3 预演落地 → v0.4 补登记 → v0.5 作者改参
> （窗口 120、7 种子）→ **v0.6 数据划分修正**（SMD/MSL 用官方 train/test 边界，
> 修复 val_normal=0 崩溃；实验主网格与 v0.5 相同）。
> → v0.7 子采样可行性 → v0.8 分层划分 → **v0.9 新增消融行与新基线行，协议数值不变**。
> → **v0.10 登记 TranAD/USAD 深度基线超参，主网格与协议数值不变**。

```yaml
# ============ 机器可读区：run_all.py 读取 ============
protocol_version: 0.10
locked: true             # CP1 已批准（2026-09-30）；v0.9 消融登记（2026-10-05）
random_seeds: [0, 1, 2, 3, 4, 5, 6]  # 7 个随机种子
window_length: 120       # 滑动窗口长度（SKAB 100Hz 下约 1.2s）
stride: 10               # 窗口步长（约 0.1s）
early_stop_patience: 10  # 早停耐心值（验证损失）
max_epochs: 50           # 神经模型最大训练轮数（smoke 模式覆盖为 2）
val_ratio: 0.2           # 正常数据中划给验证的比例
test_ratio: 0.2          # 含异常数据的测试段比例
threshold_search:        # 阈值在验证集搜索
  enabled: true
  quantile_grid: [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999]
  criterion: f1_pa       # 以验证集 PA-F1 最优为准
report_pa_and_nonpa: true  # PA 与非 PA 双报告
datasets: [skab, smd, msl]  # 3 个数据集（data/ 目录，格式见 data/README.md）
models:                     # 8 个模型（src/models/ 注册表）
  - ae_mlp      # MLP 自编码器（重构误差）
  - lstm_ad     # LSTM 预测器（预测误差，numpy 实现）
  - pca_recon   # PCA 重构误差
  - iforest     # 孤立森林
  - ocsvm       # 单类 SVM
  - knn_dist    # k 近邻距离
  - deep_svdd   # Deep SVDD（MLP 超球面）
  - distill_ae  # 蒸馏学生模型（teacher=AE-L，课题轻量化核心）
baseline_model: pca_recon   # 显著性检验的对照基线
# ---- 超参唯一登记处（与 src/models/*.py 默认值逐字一致，改动须升版本）----
hyperparams:
  ae_mlp: {h1: 64, z: 16}
  lstm_ad: {hidden: 32}
  pca_recon: {n_components: 0.95}
  iforest: {n_estimators: 100, contamination: 0.02}
  ocsvm: {nu: 0.05, gamma: scale, max_train: 5000}
  knn_dist: {k: 5, max_train: 5000}
  deep_svdd: {h1: 64, z: 8}
  distill_ae: {teacher_h1: 128, teacher_z: 32, student_h1: 16, student_z: 4, lam: 0.3, max_train: 15000}
  # ---- v0.9 消融登记（主网格 8 模型数值不变）----
  ae_l_teacher: {h1: 128, z: 32, max_train: 15000}      # 教师结构单体 AE
  ae_s_nodistill: {h1: 16, z: 4, max_train: 15000}      # 学生结构仅重构（λ=0）
  distill_ae_lam01: {teacher_h1: 128, teacher_z: 32, student_h1: 16, student_z: 4, lam: 0.1, max_train: 15000}
  distill_ae_lam05: {teacher_h1: 128, teacher_z: 32, student_h1: 16, student_z: 4, lam: 0.5, max_train: 15000}
  # ---- v0.10 深度基线（imperial-qore/TranAD；manigalati/usad）----
  tranad: {batch: 128, hidden: 8, n_window: 120, epochs: 5, lr: 0.001, max_train: 15000, score: focus+adv_0.5/0.5}
  usad: {z_size: 32, batch: 128, lr: 0.001, epochs: 100, max_train: 15000, score: recon+adv_0.5/0.5}
# 子采样披露（v0.7）：OCSVM/kNN 精确解 O(n²)+，SMD 的 13.5 万窗上不可行。
# 训练窗 >5000 时随机抽 5000（seed 固定）再拟合；Limitations 必须写明。
distill_ablation:            # λ 消融：不进 72 组主网格
  lam_grid: [0.1, 0.3, 0.5]
  main_lam: 0.3              # 主网格固定
  ablate_datasets: [skab]    # 仅 SKAB 做消融（附录表）
metrics:
  point: [precision, recall, f1]        # 非 PA（点级）
  pa: [precision_pa, recall_pa, f1_pa]  # PA（点调整）
  ranking: [auc_roc, auc_pr]            # 阈值无关
  efficiency: [params, model_kb, latency_ms_cpu]  # 硬件开销
latency_windows: 100        # CPU 延迟测量的窗口数（取中位数）
rpi_latency_file: results/rpi_latency.csv  # 树莓派真机实测文件（可选，人工导入）
# ====================================================
```

## 1 课题与对象

工业多变量时序异常检测的**轻量化评测**：在统一协议下比较 8 种检测器，并验证
「大教师 → 微型学生」的分数蒸馏能否以约 1/10 参数量保持检测精度（对应规划书
的评测+蒸馏双主线）。

## 2 数据集与划分

- 数据集 3 个：SKAB（工业测试台）、SMD（服务器机组）、MSL（航天器遥测）。
  落盘格式见 `data/README.md`；**没有落盘真实数据时，全量运行直接报错退出，
  绝不用合成数据顶替**（`--smoke` 管线自检除外，且合成结果在 results.csv 中
  永久标记 `data_source=synthetic`，verify.py 见到即 FAIL）。
- 划分（与 `src/data.py` 逐字对应）：
  - **SMD / MSL（有官方 train 段）**：`train_len` 之前的官方 train 作训练
    （剔除其中异常点）；**余下段（官方 test）按类别分层 50/50** 切验证与测试，
    保证两边都含正常+异常（v0.8；v0.7 对半时间切导致 MSL 测试全为异常、AUC 无效）。
    验证段用于阈值搜索，测试段只做最终评估。
  - **SKAB（无官方 train 段）**：时间顺序切分——前 60% 的仅正常作训练；
    正常验证段（20%）用于早停；异常验证段（20%）用于阈值搜索；测试段
    （20%）只做最终评估。
  - 归一化统计量只来自训练段；阈值、早停、超参选择均不得读测试段。
  - **（v0.6）** 合并 train+test 后再按比例切，会在 SMD/MSL 尾部全异常段
    切出空 `val_normal` 导致崩溃，故小/尾部异常密集数据集必须用官方划分。

## 3 训练协议（神经模型）

- 窗口 120、步长 10。依据：SKAB 采样率 100 Hz，120 点 ≈ 1.2 s、步长 ≈ 0.1 s，
  覆盖更长上下文以刻画工业过程的持续性异常（v0.5 作者改参）。
- 输入为按训练段统计的 z-score 标准化（统计量只来自训练段）。
- 优化器 Adam(lr=1e-3, batch=64)；早停 patience=10，监控**正常验证段**损失
  （样本来源见第 2 节第 2 条，与 `src/data.py` 一致）；最大 50 epoch
  （smoke 模式 2 epoch）。
- 每个模型 × 数据集 × 种子组合独立训练，共 **8×3×7=168 次**运行，单卡顺序排队。
- **蒸馏超参**：主网格 λ=0.3，教师 AE-L（h1=128, z=32）冻结，学生 AE-S
  （h1=16, z=4）。λ ∈ {0.1, 0.3, 0.5} 消融**仅在 SKAB** 上补跑（附录表，
  不进主网格，见 `distill_ablation`）。全部超参以本文件 `hyperparams` 为
  唯一登记处，与代码默认值逐字一致。

## 4 评估协议

- 异常分数统一为「窗口 → 点」映射：一个点被覆盖的窗口的分数中位数。
- 阈值：仅在验证集上按分位网格搜索，以 PA-F1 最优选定；测试集用同一阈值
  报告 PA 与非 PA 两套 precision/recall/F1（**主表两套并列，缺一无效**），
  同时报告阈值无关的 AUC-ROC/AUC-PR。
- 硬件开销：参数量、序列化体积（KB）、本机 CPU 单窗口推理延迟中位数（ms）。
  树莓派延迟**只能**来自 `results/rpi_latency.csv` 的真机实测（人工在树莓派上
  按 `scripts/rpi_latency.md` 流程导入）；文件不存在时该列留空，论文不得引用。

## 5 统计与报告

- 排行榜按种子均值±标准差；95% 置信区间（t 分布，**n=7 种子**）。
- 显著性：对每个模型 vs 基线（pca_recon）做配对 Wilcoxon 符号秩检验
  （配对单元 = 数据集×种子，3×7=21 对），Holm 校正，α=0.05。
- **阈值选择偏差披露**：论文须写明阈值在 10 分位网格上按 PA-F1 优选，
  存在多重比较效应；敏感性分析不进主文，留作 rebuttal 补充实验位。
- analyze.py 输出 `analysis/report.md`；所有数字四舍五入到 4 位小数，verify.py
  以该精度做字符串级对账。

## 6 失败处理与防伪造

- 单任务失败自动重试 1 次；再失败则记录 `results/failed_runs.log`（含堆栈）
  并跳过，不阻塞整批。**每次运行的 `retries` 计数写入 results.csv**。
  断点续跑：results.csv 已存在的组合自动跳过（synthetic 行不占用名额）。
- 正文数字一律由 `draft/inject_tables.py` 从 results.csv 注入；AI 不得手输数字。
- verify.py 逐数字对账 + 引用检查 + 【待填】检查 + 合成数据拦截 +
  树莓派延迟来源核查，FAIL 则投稿包链（F）不得启动。

## 批准记录（CP1）

| 项目 | 内容 |
|------|------|
| 审阅人 | 作者（本会话确认 v0.5 改参并批准） |
| 日期 | 2026-09-30 |
| 结论 | **同意锁定**（v0.5） |
| 作者 | 2026-10-05 · 同意 v0.9 消融登记（新增模型行，主网格数值不变） |
| 作者 | 2026-10-05 · 同意 v0.10 登记 TranAD/USAD（协议数值不变） |
