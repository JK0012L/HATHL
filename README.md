# HATHL Code Usage Guide

> Experimental code for the paper: **"HATHL: A heterogeneity-aware hybrid learning framework for dynamic job shop scheduling with multi-source heterogeneity"**

---

## 1. Environment Dependencies

```bash
pip install numpy torch scipy simpy pandas matplotlib seaborn openpyxl scikit-posthocs pymoo
```

- Python 3.8+
- PyTorch 1.12+ (GPU version recommended)

---

## 2. Directory Structure

```
multi-object-omdjsp/
│
├── RL model training.py                            # ① RL model training entry point
├── Comparison with Dispatching Rules.py          # ② Comparison with scheduling rules
├── Comparison with Evolutionary Algorithms.py    # ③ Comparison with multi-objective evolutionary algorithms
├── Comparison with DRL.py                        # ④ Comparison with DRL algorithms
├── Effects of Heterogeneity.py                   # ⑤ Heterogeneity sensitivity analysis
├── Ablation.py                               # ⑥ Ablation study
│
├── agent/                    # Shop floor simulation agents (Job Creator / Machine / Worker)
├── algorithm/RL/             # DRL algorithms (HATHL / A2C / DDPG / SAC / TD3)
├── algorithm/SI/             # Static evolutionary algorithms (NSGA-II / MOEA-DD / RVEA / SMS-EMOA)
├── common/                   # Common modules (scheduling utils / scenario definitions / network base / analysis)
├── sequencing_models/        # Pre-trained model weights (.pt) and preference vectors (.pkl)
├── test_result/              # Experiment results and visualization reports
└── train_result/             # Training process logs
```

---

## 3. Core File Descriptions

### Common Library (`common/`)

| File | Description |
|:---|:---|
| `base_brain.py` | Scheduling brain base class + Dual-path gated fusion network |
| `cfunctions.py` | Scheduling helper functions (15-dim state feature extraction / Gantt chart / experience storage) |
| `experiment_scene.py` | 12 orthogonal experiment scenario parameter definitions |
| `experiment_sensitivity.py` | Heterogeneity sensitivity analysis scenario definitions |
| `experimental_analysis.py` | Experiment data analysis (HV / IGD / Spread metrics computation and visualization) |
| `multiobject.py` | Multi-objective manager (external archive maintenance / preference vector generation / importance matrix) |
| `scenario_freeze.py` | Scenario freezer (dynamic simulation → static instance conversion) |
| `sensitivity_analysis.py` | Heterogeneity sensitivity statistical analyzer |
| `sequencing.py` | 5 classic scheduling rules |
| `shop_floor.py` | Shop floor simulation main controller |
| `static_scheduling.py` | Static scheduling decoding and evaluation library (for multi-objective evolutionary algorithms) |

### Model Files (`sequencing_models/`)

Naming convention: `{Algorithm}_{Scenario_ID}.pt` + `{Algorithm}_{Scenario_ID}_preferences.pkl`

For example:
- `HATHL_EXP-1.pt` / `HATHL_EXP-1_preferences.pkl` — HATHL model weights and preference vectors for Scenario 1
- `A2C_EXP-1.pt` — A2C model weights for Scenario 1
- `HATHL_Ablation1_EXP-1.pt` — Ablation 1 model weights for Scenario 1

---

## 4. Running Experiments

### 4.1 Training RL Models

```bash
python RL model training.py
```

- Trains all ablation variants by default (Ablation1/2/3)
- Modify the `benchmark` list in the script to switch between training HATHL or other RL algorithms
- Trained models are automatically saved to `sequencing_models/`

**Key parameters:** 10000 training jobs, 12 scenarios trained independently

### 4.2 Comparison with Scheduling Rules

```bash
python Comparison with Dispatching Rules.py
```

- Compares HATHL against SPT / LWKR / WINQ / SRO / NPT
- 30 independent repeats per scenario, 50 jobs
- DRL models evaluated with 5 random preference vectors

**Output location:** `test_result/experimental_analysis/Rule/`

### 4.3 Comparison with Multi-Objective Evolutionary Algorithms

```bash
python Comparison with Evolutionary Algorithms.py
```

- Compares HATHL against NSGA-II / MOEA/DD / RVEA / SMS-EMOA
- Uses scenario freezer to convert dynamic instances to static instances for fair comparison
- Multi-objective algorithms use population size 50, 100 generations

**Output location:** `test_result/experimental_analysis/DMO/`

### 4.4 Comparison with RL Algorithms

```bash
python Comparison with DRL.py
```

- Compares HATHL against SAC / TD3 / A2C / DDPG
- All algorithms load pre-trained models and test on identical scenarios

**Output location:** `test_result/experimental_analysis/Ref_Learning/`

### 4.5 Heterogeneity Sensitivity Analysis

```bash
python Effects of Heterogeneity.py
```

- Independently tests 9 levels each for machine / worker / repair heterogeneity
- Fixed baseline scenario (M=10, W=5, U=0.95)
- When testing one heterogeneity type, the other two are fixed at baseline values

**Output location:** `test_result/experimental_analysis/Sensitivity/`

### 4.6 Ablation Study

```bash
python Ablation.py
```

- Ablation1: Remove Chebyshev scoring → random selection + random reward
- Ablation2: Remove mathematical programming optimization layer
- Ablation3: Replace Pareto-front-based preference vector generation with random sampling

**Output location:** `test_result/experimental_analysis/Ablation/`

---

## 5. Key Adjustable Parameters

| Parameter | Location | Default | Description |
|:---|:---|:---|:---|
| `run_num` | Comparison scripts | 30 | Number of independent repeats per scenario |
| `job_num` | Comparison scripts | 50 | Number of test jobs |
| `drl_preference_runs` | Comparison scripts | 5 | Number of DRL preference vector samples |
| `pop_size` | algorithm/SI/*.py | 50 | Population size for multi-objective evolutionary algorithms |
| `max_gen` | algorithm/SI/*.py | 100 | Maximum generations for multi-objective evolutionary algorithms |

> **Note:** The training job count in `RL model training.py` is fixed at 10000 and should not be modified.

---

## 6. Viewing Results

After each comparison script completes, `data_analysis_report()` is automatically invoked to generate:

- **`statistical_tables.xlsx`** — Friedman test / Nemenyi post-hoc test / Algorithm ranking
- **`detailed_metrics.xlsx`** — Fine-grained scenario metrics (mean ± std with significance annotation)
- **`win_rate_statistics.xlsx`** — Algorithm win rate statistics
- **`analysis_report.txt`** — Comprehensive text analysis report
- **`*.png`** — Boxplots / Radar charts / Heatmaps / Significance matrices

All outputs are located under `test_result/experimental_analysis/{subdirectory}/`.

---

## 7. Citation

If this code is helpful to your research, please cite our paper:

```bibtex
@article{YourPaper2025,
  title     = {Heterogeneity-Aware Transferable Hypervolume-based Learning for Multi-Objective Dynamic Job Shop Scheduling},
  author    = {...},
  journal   = {...},
  year      = {2025}
}
```

---

**For academic research and paper reproduction only.**

---

---

# HATHL 代码使用说明

> 论文 **"HATHL: A heterogeneity-aware hybrid learning framework for dynamic job shop scheduling with multi-source heterogeneity"** 的实验代码

---

## 1. 环境依赖

```bash
pip install numpy torch scipy simpy pandas matplotlib seaborn openpyxl scikit-posthocs pymoo
```

- Python 3.8+
- PyTorch 1.12+（推荐 GPU 版本）

---

## 2. 目录结构

```
multi-object-omdjsp/
│
├── 训练强化学习模型.py              # ① RL模型训练入口
├── 对比-调度规则.py                 # ② 与调度规则对比实验
├── 对比-多目标算法.py               # ③ 与多目标进化算法对比实验
├── 对比-强化学习算法.py              # ④ 与RL算法对比实验
├── 对比-敏感性实验.py               # ⑤ 异质性敏感性分析
├── 对比-消融实验.py                 # ⑥ 消融实验
│
├── agent/                          # 车间仿真Agent（工件生成器/设备/工人）
├── algorithm/RL/                   # 强化学习算法（HATHL/A2C/DDPG/SAC/TD3）
├── algorithm/SI/                   # 静态多目标进化算法（NSGA-II/MOEA-DD/RVEA/SMS-EMOA）
├── common/                         # 公共模块（调度函数库/场景定义/网络基类/实验分析）
├── sequencing_models/              # 预训练模型权重（.pt）及偏好向量（.pkl）
├── test_result/                    # 实验结果与可视化分析报告
└── train_result/                   # 训练过程记录
```

---

## 3. 核心文件说明

### 公共库 (`common/`)

| 文件 | 功能 |
|:---|:---|
| `base_brain.py` | 调度大脑基类 + 双路径门控融合网络 |
| `cfunctions.py` | 调度辅助函数（状态特征提取(15维)/甘特图绘制/经验存储） |
| `experiment_scene.py` | 12个正交实验场景参数定义 |
| `experiment_sensitivity.py` | 异质性敏感性分析场景定义 |
| `experimental_analysis.py` | 实验数据分析（HV/IGD/Spread 指标计算与可视化） |
| `multiobject.py` | 多目标管理器（外部分档维护 / 偏好向量生成 / 重要性矩阵） |
| `scenario_freeze.py` | 场景冻结器（动态仿真→静态实例转换） |
| `sensitivity_analysis.py` | 异质性敏感性统计分析器 |
| `sequencing.py` | 5条经典调度规则 |
| `shop_floor.py` | 车间仿真主控类 |
| `static_scheduling.py` | 静态调度解码与评估函数库（供多目标进化算法调用） |

### 模型文件 (`sequencing_models/`)

命名规则：`{算法名}_{场景ID}.pt` + `{算法名}_{场景ID}_preferences.pkl`

例如：
- `HATHL_EXP-1.pt` / `HATHL_EXP-1_preferences.pkl` —— 场景1下 HATHL 的模型权重和偏好向量
- `A2C_EXP-1.pt` —— 场景1下 A2C 的模型权重
- `HATHL_Ablation1_EXP-1.pt` —— 场景1下消融实验1的模型权重

---

## 4. 实验运行

### 4.1 训练强化学习模型

```bash
python 训练强化学习模型.py
```

- 默认训练所有消融实验变体（Ablation1/2/3）
- 修改脚本中 `benchmark` 列表可切换训练 HATHL 或其他 RL 算法
- 训练完成后模型自动保存至 `sequencing_models/`

**关键参数：** 训练工件数 10000，12 个场景独立训练

### 4.2 与调度规则对比

```bash
python 对比-调度规则.py
```

- 对比 HATHL 与 SPT / LWKR / WINQ / SRO / NPT 五条经典规则
- 每组 30 次重复实验，50 个工件
- DRL 模型使用 5 组随机偏好向量评估

**输出位置：** `test_result/experimental_analysis/Rule/`

### 4.3 与多目标进化算法对比

```bash
python 对比-多目标算法.py
```

- 对比 HATHL 与 NSGA-II / MOEA/DD / RVEA / SMS-EMOA
- 使用场景冻结器将动态实例转为静态实例，保证对比公平性
- 多目标算法使用 50 个种群个体、100 代进化

**输出位置：** `test_result/experimental_analysis/DMO/`

### 4.4 与强化学习算法对比

```bash
python 对比-强化学习算法.py
```

- 对比 HATHL 与 SAC / TD3 / A2C / DDPG
- 各算法均加载预训练模型在相同场景下测试

**输出位置：** `test_result/experimental_analysis/Ref_Learning/`

### 4.5 异质性敏感性分析

```bash
python 对比-敏感性实验.py
```

- 分别对设备、工人、维修三种异质性进行 9 个水平的独立测试
- 固定基准场景（M=10, W=5, U=0.95）
- 每种异质性类型下其他两种异质性固定为基准值

**输出位置：** `test_result/experimental_analysis/Sensitivity/`

### 4.6 消融实验

```bash
python 对比-消融实验.py
```

- Ablation1：移除切比雪夫评分 → 随机选择 + 随机奖励
- Ablation2：移除数学规划优化层
- Ablation3：偏好向量改为随机采样（而非基于 Pareto 前沿）

**输出位置：** `test_result/experimental_analysis/Ablation/`

---

## 5. 关键可调参数

| 参数 | 位置 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `run_num` | 各对比脚本 | 30 | 每场景独立重复次数 |
| `job_num` | 各对比脚本 | 50 | 测试集工件数 |
| `drl_preference_runs` | 各对比脚本 | 5 | DRL 偏好向量采样数 |
| `pop_size` | algorithm/SI/*.py | 50 | 多目标进化算法的种群大小 |
| `max_gen` | algorithm/SI/*.py | 100 | 多目标进化算法的最大代数 |

> **注意：** `训练强化学习模型.py` 中训练工件数固定为 10000，不建议修改。

---

## 6. 实验结果查看

每个对比脚本运行完毕后，自动调用 `data_analysis_report()` 生成以下文件：

- **`statistical_tables.xlsx`** —— Friedman 检验 / Nemenyi 事后检验 / 算法排名
- **`detailed_metrics.xlsx`** —— 各场景细粒度指标（含均值±标准差及显著性标注）
- **`win_rate_statistics.xlsx`** —— 算法胜率统计
- **`analysis_report.txt`** —— 文本格式综合分析报告
- **`*.png`** —— 箱线图 / 雷达图 / 热力图 / 显著性矩阵等可视化图表

所有输出位于 `test_result/experimental_analysis/{子目录名}/`。

---

## 7. 引用

如果本代码对您的研究有帮助，请引用我们的论文：

```bibtex
@article{YourPaper2025,
  title     = {面向多源异质性的动态多目标作业车间调度——基于超体积学习的异质性感知可迁移框架},
  author    = {...},
  journal   = {...},
  year      = {2025}
}
```

---

**仅供学术研究与论文复现使用。**