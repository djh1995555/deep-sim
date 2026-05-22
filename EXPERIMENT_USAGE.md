# 手动实验使用指南

本指南说明如何在当前仓库手动运行、监控和汇总实验。默认工作目录：

```bash
cd /home/mi/vibe/research/deep_sim/codex
```

所有命令都应使用 `deep-sim` conda 环境：

```bash
conda run -n deep-sim <command>
```

不要直接用系统 Python 运行实验。

如果只是想手动运行一个指定工况的闭环车辆仿真 episode，而不是跑完整实验或数据集生成流程，见根目录 `SIMULATOR_USAGE.md`。

## 输出目录约定

以后运行产生的文件统一写入 `output/`：

| 路径 | 内容 |
| --- | --- |
| `output/training/` | `experiments.run`、批量队列、checkpoint、metrics、post-rollout eval 和训练汇总报告。 |
| `output/simulation/` | `simulator.cli` 单次闭环仿真、debug trace、Plotly HTML 和仿真 episode。 |

`data/` 只保存训练输入数据；不要把新实验产物写进 `data/`。

## 1. 运行前检查

确认代码状态：

```bash
git status --short
```

确认环境和 GPU：

```bash
conda run -n deep-sim python -m unittest
conda run -n deep-sim python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

确认 canonical dataset 存在并可读：

```bash
conda run -n deep-sim python -m experiments.materialize_data
```

当前 canonical 数据是 `data/` 下的真实目录，不再是指向历史运行输出目录的 symlink。需要把数据复制到其他位置时使用：

```bash
conda run -n deep-sim python -m experiments.materialize_data --mode copy --data-root /path/to/target_data
```

## 2. 运行单个实验

统一入口：

```bash
conda run -n deep-sim python -m experiments.run --config configs/experiments/p6_model_dev/p6_1_pytorch_base_model_small_training.yaml
```

常用 smoke：

```bash
conda run -n deep-sim python -m experiments.run --config configs/experiments/p3_smoke/p3_1_pytorch_data_loader_smoke.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/p4_gpu_smoke/p4_1_pytorch_gpu_forward_smoke.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/p6_model_dev/p6_1_pytorch_base_model_small_training.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/p6_model_dev/p6_3_pytorch_model_variant_smoke.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/p7_adapter_ensemble/p7_1_pytorch_fine_tune_adapter_smoke.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/p7_adapter_ensemble/p7_2_pytorch_deep_ensemble_smoke.yaml
```

DS2 / MoE 开发入口：

```bash
conda run -n deep-sim python -m experiments.run --config configs/experiments/b7_extreme_moe/b7_1_ds2_extreme_dataset_smoke.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/b7_extreme_moe/b7_2_pytorch_ds2_moe_tire_smoke.yaml
```

每个 run 会写到：

```text
output/training/{run_id}_{run_name}/
  summary.json
  metrics.jsonl
  artifacts/
  checkpoints/
  config.yaml
  resolved_config.yaml
  env.txt
  git_status.txt
```

当前仓库已经清理过历史运行产物。canonical 数据已经移动到 `data/ds1_v1` 和 `data/ds1_proxy_ft_v1`；重新执行实验时，runner 会按配置重新创建对应的 `output/training/` 输出目录。

重新运行某个实验后，优先看：

```bash
cat output/training/R111_pytorch_base_model_small_training/summary.json
cat output/training/R111_pytorch_base_model_small_training/artifacts/validation_report.json
```

未重新运行时，使用 `output/training/reports/` 和 `refine-logs/EXPERIMENT_RESULTS.md` 查看已完成阶段的结果摘要；根目录 `reports/` 不再作为新产物目录维护。

## 3. 运行完整 PyTorch 矩阵

先生成或刷新矩阵配置：

```bash
conda run -n deep-sim python -m experiments.torch_config_matrix --write
```

检查 dry-run：

```bash
conda run -n deep-sim python -m experiments.experiment_queue \
  --manifest configs/experiments/matrix/MANIFEST.json \
  --run-ids R200 R201 \
  --dry-run \
  --reset-state \
  --state-path output/training/queue_state_dryrun.json
```

运行 R200-R216 ablation：

```bash
conda run -n deep-sim python -m experiments.experiment_queue \
  --manifest configs/experiments/matrix/MANIFEST.json \
  --run-ids R200 R201 R202 R203 R204 R205 R206 R207 R208 R209 R210 R211 R212 R213 R214 R215 R216 \
  --max-retries 1 \
  --skip-success \
  --rollout-eval \
  --state-path output/training/queue_state_ablation.json \
  --log-dir output/training/_queue_logs_ablation
```

运行 R300-R334 fine-tune matrix：

```bash
conda run -n deep-sim python -m experiments.experiment_queue \
  --manifest configs/experiments/matrix/MANIFEST.json \
  --run-ids R300 R301 R302 R303 R304 R305 R306 R307 R308 R309 R310 R311 R312 R313 R314 R315 R316 R317 R318 R319 R320 R321 R322 R323 R324 R325 R326 R327 R328 R329 R330 R331 R332 R333 R334 \
  --max-retries 1 \
  --skip-success \
  --rollout-eval \
  --state-path output/training/queue_state_finetune.json \
  --log-dir output/training/_queue_logs_finetune
```

查看队列状态：

```bash
cat output/training/queue_state_ablation.json
ls output/training/_queue_logs_ablation
```

队列参数说明：

| 参数 | 用途 |
|---|---|
| `--manifest` | 从 `configs/experiments/matrix/MANIFEST.json` 读取 run 列表。 |
| `--configs` | 直接指定一个或多个 config 文件。 |
| `--run-ids` | 只运行指定 run id。 |
| `--dry-run` | 只写计划和状态，不实际训练。 |
| `--skip-success` | 已成功的 run 不重复训练。 |
| `--rollout-eval` | 训练成功后自动跑 post-rollout eval。 |
| `--max-retries` | 失败后重试次数。 |
| `--stop-on-failure` | 遇到失败立刻停止队列。 |
| `--state-path` | 队列状态 JSON 输出路径。 |
| `--log-dir` | 每次尝试的 stdout/stderr 日志目录。 |

## 4. 汇总结果

刷新 PyTorch development report：

```bash
conda run -n deep-sim python -m experiments.torch_dev_report
```

刷新完整矩阵报告：

```bash
conda run -n deep-sim python -m experiments.matrix_report
```

主要报告位置：

```text
output/training/reports/PYTORCH_DEV_REPORT.md
output/training/reports/PYTORCH_MATRIX_REPORT.md
output/training/reports/B0_teacher.md
refine-logs/EXPERIMENT_RESULTS.md
refine-logs/EXPERIMENT_TRACKER.md
```

当前 `PYTORCH_MATRIX_REPORT` 在未运行 R200-R334 前会显示 52 个 pending，这是正常状态。

## 5. 使用实车 CSV 数据

先把单个 CSV episode 转成 canonical dataset：

```bash
conda run -n deep-sim python -m experiments.real_data_adapter \
  --input-csv path/to/episode.csv \
  --output-dir data/real_v0 \
  --dataset-id REAL_V0 \
  --episode-id real_ep_000 \
  --fixed-context-json path/to/fixed_context.json \
  --nominal-prior-json path/to/nominal_prior.json \
  --field-map-json path/to/field_map.json \
  --metadata-json path/to/metadata.json
```

输出：

```text
data/real_v0/manifest.json
data/real_v0/episodes/real_ep_000.npz
data/real_v0/episodes/real_ep_000.json
data/real_v0/adapter_summary.json
```

之后复制一个现有 run config，修改：

```yaml
data:
  dataset_id: REAL_V0
  dataset_path: data/real_v0
dataset_source: existing
```

然后按普通 run 方式执行。

## 6. 修改配置时看哪里

常见入口：

| 目标 | 文件 |
|---|---|
| 改单个实验配置 | `configs/experiments/<实验块>/<语义化实验名>.yaml` |
| 改 DS1 数据生成 | `configs/datasets/ds1_v1.yaml` |
| 改 DS2 extreme 数据生成 | `configs/datasets/ds2_extreme_v0.yaml` |
| 改模型模块结构 | `student_model/torch_model.py` |
| 改训练 loop / loss / eval | `experiments/torch_training.py` |
| 改矩阵生成规则 | `experiments/torch_config_matrix.py` |
| 改队列行为 | `experiments/experiment_queue.py` |
| 改报告汇总 | `experiments/matrix_report.py` 或 `experiments/torch_dev_report.py` |

改完后至少运行：

```bash
conda run -n deep-sim python -m unittest
conda run -n deep-sim python -m compileall simulator experiments student_model tests
git diff --check
```

## 7. 解释结果时的边界

当前已完成的是工程可运行性和开发级 smoke。`R112` 显示当前小规模 hybrid 在 raw one-step MSE 和 rollout RMSE 上仍落后 best black-box；不要用 R100-R115 或 R046-R047 声称最终模型优于 black-box。

正式结论至少需要：

```text
R200-R216 full ablation matrix
R300-R334 fine-tune data-efficiency matrix
post-rollout eval
matrix report
必要时再做 R048+ DS2 MoE full training/eval
```

`R048+` 是 DS2/MoE 的可选第二阶段训练评估，不是当前第一版 base model 的前置条件。
