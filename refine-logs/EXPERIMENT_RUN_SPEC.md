# EXPERIMENT_RUN_SPEC

**日期**：2026-05-21  
**状态**：implementation spec v0  
**作用**：把 `EXPERIMENT_PLAN.md` 和 `EXPERIMENT_TRACKER.md` 转换为可执行 run contract，定义 run id、输入、输出、配置、命令、成功标准和产物。

## 0. 文档边界

`EXPERIMENT_PLAN.md` 定义实验设计和科学证据链；`EXPERIMENT_TRACKER.md` 定义 run id 和状态；本文档定义每类 run 在工程实现中必须如何执行、保存和验收。

权威关系：

```text
run id / status:
  EXPERIMENT_TRACKER.md

experiment intent / success criteria:
  EXPERIMENT_PLAN.md

data schema:
  DATA_SCHEMA_SPEC.md

teacher simulator API:
  TEACHER_SIMULATOR_SPEC.md

student model API:
  STUDENT_MODEL_SPEC.md
```

## 1. Runtime Environment Contract

所有实验活动必须在 Miniforge / conda 虚拟环境中执行，包括：

```text
teacher simulator 生成
dataset export / validation
unit test / smoke test
student model training / evaluation
baseline training / evaluation
report / metrics aggregation
```

默认环境文件：

```text
environment.yml
```

默认环境名：

```text
deep-sim
```

首次创建环境：

```bash
conda env create -f environment.yml
conda activate deep-sim
python -m pip install -r requirements.txt
```

后续所有命令必须在已激活的 `deep-sim` 环境中运行，或显式使用：

```bash
conda run -n deep-sim python -m experiments.run --config configs/runs/R000.yaml
```

禁止直接使用系统 Python 执行实验命令。自动化 runner / queue / remote backend 在启动前必须记录并检查：

```text
CONDA_DEFAULT_ENV == deep-sim
CONDA_PREFIX is not empty
python executable belongs to the conda environment
package versions are captured in env.txt or equivalent environment snapshot
```

如果 backend 无法激活指定虚拟环境，应将 run 标记为 `blocked`，而不是退回系统 Python 静默执行。

## 2. Run Directory Contract

每个 run 必须创建独立目录：

```text
runs/{run_id}_{short_name}/
  config.yaml
  resolved_config.yaml
  command.txt
  env.txt
  git_status.txt
  metrics.jsonl
  summary.json
  artifacts/
  logs/
  checkpoints/
```

Required `summary.json` fields:

```yaml
run_id:
milestone:
experiment_block:
config_name:
status: success | failed | blocked | skipped
start_time:
end_time:
dataset_id:
train_split:
val_split:
test_split:
seed:
primary_metric:
primary_metric_value:
success_criteria_met: true | false
notes:
```

`env.txt` 必须至少记录：

```text
CONDA_DEFAULT_ENV
CONDA_PREFIX
python executable path
python version
installed package snapshot
platform / CUDA / GPU info if available
```

## 3. Configuration Contract

Every run config must include:

```yaml
run:
  id:
  name:
  milestone:
  experiment_block:
  seed:

data:
  dataset_id:
  dataset_path:
  schema_version:
  split_id:
  train_filter:
  val_filter:
  test_filter:

model:
  system_name:
  student_config:
  baseline_config:

training:
  optimizer:
  learning_rate:
  batch_size:
  rollout_horizon:
  max_steps:
  early_stopping_metric:
  early_stopping_patience:

evaluation:
  horizons: [1s, 5s, 10s]
  metrics:
  report_splits:

logging:
  output_dir:
  save_checkpoints:
  save_diagnostics:
```

`resolved_config.yaml` must contain all inherited defaults after expansion.

## 4. Required CLI

Recommended commands:

```bash
conda activate deep-sim
python -m experiments.run --config configs/runs/R000.yaml
python -m experiments.evaluate --checkpoint runs/R009_base/checkpoints/best.pt --config configs/eval/base.yaml
python -m experiments.report --runs runs/R009_base runs/R010_seen runs/R011_road --out reports/B3_base.md
```

Non-interactive / queue commands should use:

```bash
conda run -n deep-sim python -m experiments.run --config configs/runs/R000.yaml
```

All commands must write `command.txt` and `resolved_config.yaml`.

## 5. Milestone And Run Mapping

| Milestone | Runs | Purpose | Gate |
|---|---|---|---|
| `M0` | `R000-R000d` | teacher simulator v0 | minimal simulator, tire/load, road, sensor/actuator and export work |
| `M1` | `R000e-R000h` | DS1 scenario/data generation | multi-vehicle, multi-scenario, splits, QA |
| `M2` | `R001-R004b` | data/physics sanity | schema, timing, derived features, tiny learnability, physics smoke |
| `M3` | `R004c-R004e` | sim-to-real proxy | perturbation profiles and target windows |
| `M4` | `R005-R008` | baselines | physics-only and black-box baselines train/evaluate |
| `M5` | `R009-R014` | base hybrid | base training/evaluation/audit/seed replication |
| `M6` | `R015-R033` | component ablation suite | E/T/F/S/M/V/U ablations complete |
| `M7` | `R034-R036` | cross-vehicle/config generalization | held-out vehicle/config evidence |
| `M8` | `R037` | final single model freeze | final E/T/F/S/M/V/U selected |
| `M9` | `R038-R045` | target fine-tune | FT0-FT6 × FTD0-FTD5 data efficiency |
| `M10` | `R046+` | DS2 extreme + MoE | optional second phase |

## 6. Stage A: Teacher And Data Runs

### R000-R000d: Teacher Simulator Minimal Implementation

Required inputs:

```text
TEACHER_SIMULATOR_SPEC.md
DATA_SCHEMA_SPEC.md
configs/teacher/ds0_minimal.yaml
```

Required outputs:

```text
teacher simulator package importable
DS0 smoke episodes generated
schema validation report
unit test report
```

Success criteria:

```text
no NaN/Inf
required observable and teacher_aux_labels exported
metadata complete
braking/turning/Split-μ signs pass
dataloader leakage test passes
```

### R000e-R000h: DS1 Data Generation

Required outputs:

```text
DS1 manifest
episode files
split manifest
scenario coverage report
dataset QA report
```

Success criteria:

```text
CG-SINGLE / CG-SPLIT / CG-TRANSITION generated
multi-vehicle / multi-config present
held-out road μ and held-out vehicle/config splits present
FTD0-FTD5 target windows reproducible
vehicle_internal_params_hash available and excluded from student input
```

## 7. Stage B Sanity Runs

### R001-R004b: Data And Physics Sanity

Required checks:

```text
schema / field role check
teacher simulator physical consistency
timestamp / dt / alignment check
student-derived slip / steering / wheel dynamics check
tiny black-box + tiny base hybrid overfit
physics-only rollout smoke test
```

Success criteria:

```text
schema validator passes
teacher-only leakage test passes
tiny models overfit 5-10 short sequences
physics-only rollout does not immediately diverge on smoke cases
```

## 8. Stage B Baseline Runs

### R005: Physics-Only Baseline

Model:

```text
student physics backbone only
no neural residual
no VehicleParamAdapter
```

Report:

```text
1s / 5s / 10s rollout RMSE
constraint violations
failure mode examples
```

### R006: Black-Box Baseline

Required baseline variants:

```text
BB-MLP:
  flattened short history -> state increment

BB-GRU:
  observable/control history -> autoregressive state increment

BB-TCN:
  causal history -> autoregressive state increment

BB-NBEATSx:
  observable/control history + exogenous context -> direct multi-horizon trajectory
```

Fairness requirements:

```text
same student-visible inputs only
same train/val/test split
matched or reported parameter count
matched or reported training budget
autoregressive and direct multi-horizon results reported separately
no physics intermediate labels as input
```

### R007-R008: Baseline Audit And Report

Must report:

```text
physics-only vs black-box comparison
seen vehicle/config
held-out road μ
held-out vehicle/config
rollout stability
yaw drift
constraint violations for physics-based models
```

## 9. Stage B Base Hybrid Runs

### R009: Base Hybrid Training

Config:

```text
Base = E1 + T1 + F1 + S1 + M1a + V1 + U0
VehicleParamAdapter disabled
```

Required outputs:

```text
best checkpoint
training curves
validation rollout metrics
constraint diagnostics
residual magnitude diagnostics
```

### R010-R014: Base Evaluation And Audit

Required reports:

```text
R010: seen-config evaluation
R011: held-out road evaluation
R012: held-out vehicle/config evaluation
R013: residual/constraint audit
R014: key table 3-seed replication
```

Success criteria:

```text
base improves main rollout metrics over physics-only
base is at least not worse than black-box on 10s stability / yaw drift / physical plausibility
residual magnitudes remain bounded and interpretable
```

## 10. Stage B Component Ablations

General rule:

```text
all ablation configs train from scratch
same split, budget, optimizer family and early stopping rule
only one family differs from Base
VehicleParamAdapter disabled
```

### R015-R018: Tire Residual Ablation

Configs:

```text
T0
T1
T1-no-proj
T2
```

Decision:

```text
choose force-level T1 unless T2 matches performance with better smoothness/stability
T1-no-proj must show whether projection is necessary
```

### R019-R021: Fz Residual Ablation

Configs:

```text
F0
F1
F2
```

Decision:

```text
F1 must improve downstream rollout or constraints to justify keeping FzResidualNN
F2 cannot be the only working version for real-data mainline
```

### R022-R023: Steering Ablation

Configs:

```text
S0
S1
optional implementation check: S1-min vs S1-skip
```

Decision:

```text
S1 kept only if steering-dynamic scenes improve with bounded smooth residual
```

### R024-R027: MuHead Ablation

Configs:

```text
M0-fixed
M1a
M1b
M2-oracle
```

Decision:

```text
M1a/M1b must improve low-μ, Split-μ or transition cases over M0-fixed
M2-oracle gap must be interpretable
```

### R027a-R027c: Shared Encoder Ablation

Configs:

```text
E1
E2
E3
```

Decision:

```text
replace E1 only if E2/E3 improves held-out/transition/long-rollout without worse residual attribution or cost
```

### R028-R031: VehicleResidualNN Ablation

Configs:

```text
V0
V1
V1-large
V2-small
```

Decision:

```text
V1 kept if it improves rollout with controlled residual magnitude
V1-large diagnostic only, not default deployable model
```

### R032-R033: Uncertainty Ablation

Configs:

```text
U0
U1 K=3 ensemble
```

Decision:

```text
U1 kept only if calibration / coverage / OOD metrics improve enough to justify cost
```

## 11. Cross-Vehicle / Cross-Config Runs

### R034-R036

Systems:

```text
physics-only baseline
best black-box baseline from R006
Base
final single model candidate from B4
```

Required splits:

```text
seen vehicle/config
held-out road μ
held-out vehicle/config
held-out vehicle + held-out road μ
```

Success criteria:

```text
final single model outperforms physics-only and black-box on held-out vehicle/config
held-out gap quantified
uncertainty rises reasonably on held-out config
residual magnitude does not become uncontrolled
```

## 12. Final Single Model Freeze

### R037

Required artifact:

```yaml
final_single_model:
  checkpoint_path:
  config:
    E:
    T:
    F:
    S:
    M:
    V:
    U:
  selection_evidence:
    base_metrics:
    ablation_summary:
    held_out_summary:
    known_limitations:
```

This checkpoint is the initialization source for B6 / FT0-FT6.

## 13. Stage C Fine-Tune Runs

### R038-R045

Required grid:

```text
FT0 @ FTD0
FT1-FT6 @ FTD1, FTD2, FTD3, FTD4, FTD5
```

Budget-saving partial grid:

```text
FTD1 / FTD3 / FTD5 first
fill FTD2 / FTD4 after trend is confirmed
```

Rules:

```text
all FT configs start from R037 final single model checkpoint
FT1-FT5 train exactly one specified module/adapter
FT6 full model fine-tune
train/validation/test target windows must not overlap
early stopping uses validation target window
```

Required metrics:

```text
FT1-FT5 vs FT0@FTD0 improvement
FT1-FT5 vs FT6 gap
performance vs fine-tune data amount
adapter parameter drift
residual magnitude drift
held-out target window performance
small-data overfitting signs
```

## 14. DS2 / MoE Runs

### R046+

Entry criteria:

```text
DS1 single model stable
T0/T1/T2 decision complete
normal DS1 regression tests available
DS2 extreme handling data generated
```

MoE tire residual is kept only if:

```text
large-slip / emergency maneuver rollout improves over single-expert T1/T2
normal DS1 test does not degrade materially
additional cost is justified
```

## 15. Metrics Schema

`metrics.jsonl` each row:

```yaml
step:
split:
horizon:
metric_name:
value:
state_channel:
scenario_group:
vehicle_group:
seed:
timestamp:
```

Required metric groups:

```text
rollout_rmse
one_step_rmse
constraint_violation
Fz_budget_error
friction_ellipse_violation
residual_magnitude
residual_to_physics_ratio
uncertainty_nll
coverage
calibration_error
OOD_AUC
adapter_drift
```

## 16. Report Contract

Each experiment block must produce a Markdown report:

```text
reports/B0_teacher.md
reports/B1_sanity.md
reports/B3_base.md
reports/B4_ablation.md
reports/B5_generalization.md
reports/B6_finetune.md
reports/B7_moe.md
```

Report sections:

```text
run ids included
configs compared
dataset splits
primary result table
constraint / residual diagnostics
failure cases
decision
next action
```

## 17. Failure And Blocked Status

Use `blocked` when:

```text
required dataset or teacher field unavailable
schema validation fails before training
teacher simulator cannot generate required scenario
config requests forbidden teacher label
hardware budget insufficient for required run
```

Use `failed` when:

```text
run executes but numerical instability or metrics fail success criteria
training diverges
outputs contain NaN/Inf
leakage test fails
```

`blocked` runs must include `blocker` and `required_fix` in `summary.json`.

## 18. Reproducibility Requirements

Every run must record:

```text
git status
git commit if available
Python version
package versions
CUDA/GPU info if used
random seed
dataset manifest hash
resolved config hash
```

Random seed policy:

```text
screening runs may use 1 seed
key B3/B4/B5 tables require 3 seeds
U1 ensemble requires K=3 independent members
```
