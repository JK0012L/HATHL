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

---

**For academic research and paper reproduction only.**

---
