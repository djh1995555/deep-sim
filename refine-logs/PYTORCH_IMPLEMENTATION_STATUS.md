# PyTorch Implementation Status

**Date**: 2026-05-21 20:43 CST

## Scope

This file tracks the transition from scaffold experiments to the training-grade PyTorch implementation.

Current completed step:

```text
P1: canonical data entry points
P2: PyTorch Student Model v0 skeleton
P3: PyTorch smoke runner scaffold
P4: R100-R104 CPU smoke passed
P5: R105 GPU smoke passed
P6: R106 GPU backward/optimizer smoke passed
P7: R107-R111 training loop development passed
P8: R112-R115 fair comparison, variant, fine-tune, and ensemble development passed
P9: experiment queue, matrix report, real-data adapter, and training governance completed
P10: DS2 extreme scaffold and T3-MoE tire residual forward path completed
```

## P1 Canonical Data

Canonical dataset paths now exist under `data/`:

```text
data/ds1_v1
data/ds1_proxy_ft_v1
```

They are symlinks to the already generated scaffold artifacts:

```text
data/ds1_v1 -> runs/R000g_dataset_split_generation/artifacts/ds1
data/ds1_proxy_ft_v1 -> runs/R038_finetune_FT0/artifacts/ds1_proxy
```

Materialization command:

```bash
conda run -n deep-sim python -m experiments.materialize_data --mode symlink
```

Use `--mode copy` only when a remote backend cannot follow symlinks.

## P2 Student Model v0 Skeleton

Implemented package:

```text
student_model/
```

Current modules:

| File | Purpose |
| --- | --- |
| `student_model/constants.py` | Shared state/control/context key definitions. |
| `student_model/data.py` | Canonical dataset manifest loader, episode array loader, context vector encoder, Torch transition dataset, teacher aux labels, fine-tune bucket/window filters. |
| `student_model/torch_model.py` | PyTorch hybrid student model with E1/E2/E3 encoders, Fz/Tire/Mu/Steer/Vehicle residual switches, VehicleParamAdapter trainability, and U0-compatible uncertainty output. |

The current model implements the first training target selected by scaffold plus development switches for ablation:

```text
E1 small GRU encoder
E2 causal TCN encoder
E3 causal Transformer encoder
T0 disabled tire residual
T1 force-level tire residual with friction ellipse projection
T1-no-proj force-level tire residual without projection
T2 parameter-level tire residual proxy
F1 bounded Fz residual
S1 bounded steering residual
M0-fixed fixed μ
M1a learned MuHead
V2-small Δx after integration residual
U0 single-model log variance head
FT0-FT6 trainability matrix for VehicleParamAdapter / MuHead / Fz / Tire / Steering / full model
```

## P3 PyTorch Smoke Runner Scaffold

Implemented files:

| File | Purpose |
| --- | --- |
| `experiments/torch_training.py` | Import-safe PyTorch smoke runner for data loader, forward/loss, tiny overfit, rollout, and checkpoint save/load checks. |
| `configs/runs/R100.yaml` | Data loader smoke on canonical DS1. |
| `configs/runs/R101.yaml` | One-step forward/loss smoke. |
| `configs/runs/R102.yaml` | Tiny overfit on a small train subset. |
| `configs/runs/R103.yaml` | Short rollout smoke. |
| `configs/runs/R104.yaml` | Checkpoint save/load smoke. |
| `configs/runs/R105.yaml` | CUDA-required one-step forward/loss smoke. |
| `configs/runs/R106.yaml` | CUDA-required tiny-overfit smoke with backward and optimizer steps. |
| `configs/runs/R107.yaml` | CUDA one-step train/validation/checkpoint run. |
| `configs/runs/R108.yaml` | Rollout evaluation from the R107 checkpoint. |
| `configs/runs/R109.yaml` | Checkpoint resume + eval-only smoke. |
| `configs/runs/R110.yaml` | PyTorch direct TCN black-box baseline small training. |
| `configs/runs/R111.yaml` | Small base hybrid training run with more samples/steps. |
| `configs/runs/R112.yaml` | Matched-budget fair small comparison: hybrid vs direct TCN/GRU/N-BEATS. |
| `configs/runs/R113.yaml` | Model/component variant forward smoke for E/T/F/S/M/V and black-box variants. |
| `configs/runs/R114.yaml` | FT0-FT6 fine-tune adapter smoke over two data-size buckets. |
| `configs/runs/R115.yaml` | U1 K=3 deep ensemble smoke. |
| `configs/teacher/ds2_extreme_v0.yaml` | DS2 extreme handling scaffold data config. |
| `configs/runs/R046.yaml` | DS2 emergency/fishhook/lane-change dataset smoke. |
| `configs/runs/R047.yaml` | DS2 T1/T2/T3-MoE tire residual forward smoke. |
| `experiments/torch_config_matrix.py` | Generates full PyTorch ablation and fine-tune config matrix under `configs/torch_matrix/`. |
| `experiments/torch_dev_report.py` | Aggregates R112-R115 results into `reports/PYTORCH_DEV_REPORT.md/json`. |
| `experiments/experiment_queue.py` | Local queue runner for direct config lists or `configs/torch_matrix/MANIFEST.json`, with dry-run, retry, skip-success, queue state, logs, and optional post-rollout eval. |
| `experiments/matrix_report.py` | Aggregates matrix status, validation metrics, and post-rollout metrics into `reports/PYTORCH_MATRIX_REPORT.md/json`. |
| `experiments/real_data_adapter.py` | Converts CSV real-vehicle episodes into canonical datasets with manifest, episode arrays, sidecars, and validator summary. |
| `tests/test_torch_training.py` | Runner/config regression tests with a blocked path when PyTorch is absent. |
| `tests/test_experiment_engineering.py` | Regression tests for queue state writing, skip-success rollout path, matrix manifest aggregation, and CSV adapter output. |
| `AGENTS.md` | Local run-experiment environment notes, GPU status, and smoke run commands. |

The new runner branch supports:

```text
dataset_source: existing
```

This lets R100-R104 reuse `data/ds1_v1` instead of regenerating DS1 every time.

## P9 Experiment Engineering

Training governance added to `experiments/torch_training.py`:

```text
best checkpoint saving
early_stopping_patience / early_stopping_min_delta
lr_scheduler: none / cosine / step
lr_step_size / lr_gamma
nonfinite loss detection
completed-step and final-LR metrics
rollout eval filters for fine_tune_buckets, target_window_role, scenario_groups, vehicle_config_ids
```

Queue smoke command:

```bash
conda run -n deep-sim python -m experiments.experiment_queue \
  --configs configs/runs/R111.yaml \
  --limit 1 \
  --max-retries 0 \
  --skip-success \
  --rollout-eval \
  --rollout-steps 8 \
  --rollout-max-episodes 2 \
  --state-path runs/queue_state_smoke.json \
  --log-dir runs/_queue_logs_smoke
```

Queue smoke result:

```text
R111 post-rollout via queue: pass
rollout RMSE = 5.861144
episodes = 2
steps = 8
constraint violation rate = 0.0
```

Matrix report command:

```bash
conda run -n deep-sim python -m experiments.matrix_report
```

Current matrix report:

```text
reports/PYTORCH_MATRIX_REPORT.md
reports/PYTORCH_MATRIX_REPORT.json
status_counts.pending = 52
R200-R216: pending
R300-R334: pending
```

Real-data adapter entry point:

```bash
conda run -n deep-sim python -m experiments.real_data_adapter \
  --input-csv path/to/episode.csv \
  --output-dir data/real_v0 \
  --dataset-id REAL_V0 \
  --episode-id real_ep_000 \
  --fixed-context-json path/to/fixed_context.json \
  --nominal-prior-json path/to/nominal_prior.json \
  --field-map-json path/to/field_map.json
```

## P10 DS2 / MoE Development Entry

Added optional B7 engineering support:

```text
DS2_EXTREME_V0 dataset scaffold
R046 DS2 extreme dataset smoke
R047 DS2 T1/T2/T3-MoE forward smoke
T3-MoE tire residual branch with exposed tire_moe_weights
FT4 trainability coverage includes the MoE tire residual heads
```

R046/R047 commands:

```bash
conda run -n deep-sim python -m experiments.run --config configs/runs/R046.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R047.yaml
```

R046/R047 results:

```text
R046: pass, DS2_EXTREME_V0 generated 18 episodes from 6 extreme matrix entries
R047: pass, T1/T2/T3-MoE CUDA forward checks passed on DS2
```

Scope boundary:

```text
T3-MoE is now an implemented optional path, not a selected first-version component.
It does not change Base or the R200-R216 DS1 single-factor matrix.
MoE value still requires later DS2 training/eval against T1/T2.
```

## Verification

Commands run:

```bash
conda run -n deep-sim python -m experiments.materialize_data --mode symlink
conda run -n deep-sim python -m compileall experiments student_model tests
conda run -n deep-sim python -m unittest tests.test_canonical_data tests.test_student_model
conda run -n deep-sim python -m unittest
conda run -n deep-sim python -m experiments.run --config configs/runs/R100.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R101.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R102.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R103.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R104.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R105.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R106.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R107.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R108.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R109.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R110.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R111.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R112.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R113.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R114.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R115.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R046.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R047.yaml
conda run -n deep-sim python -m experiments.torch_config_matrix --write
conda run -n deep-sim python -m experiments.torch_dev_report
conda run -n deep-sim python -m experiments.experiment_queue --configs configs/runs/R111.yaml --limit 1 --max-retries 0 --skip-success --rollout-eval --rollout-steps 8 --rollout-max-episodes 2 --state-path runs/queue_state_smoke.json --log-dir runs/_queue_logs_smoke
conda run -n deep-sim python -m experiments.matrix_report
conda run -n deep-sim python -m compileall experiments student_model tests
conda run -n deep-sim python -m unittest tests.test_experiment_engineering tests.test_torch_training
conda run -n deep-sim python -m unittest tests.test_teacher_simulator tests.test_student_model
git diff --check
```

Result:

```text
canonical data test: pass
student model smoke test: pass
torch training runner tests: pass
compileall: pass
R100-R104 runner invocations: pass
R105 runner invocation: pass on CUDA
R106 runner invocation: pass on CUDA with backward and optimizer steps
R107-R111 runner invocations: pass on CUDA
R112-R115 runner invocations: pass on CUDA
diff check: pass
```

## Dependency Status

`environment.yml` and `requirements.txt` now declare PyTorch.

Current local `deep-sim` environment now has CUDA-enabled PyTorch installed through:

```bash
mamba install -n deep-sim -c pytorch -c nvidia -c conda-forge 'pytorch::pytorch=2.5.1' pytorch-cuda=12.1 -y
```

Observed PyTorch status:

```text
torch.__version__ = 2.5.1
torch.version.cuda = 12.1
torch.cuda.is_available() = True
torch.cuda.device_count() = 1
torch.cuda.get_device_name(0) = NVIDIA RTX A4500
```

Observed driver status:

```text
nvidia-smi driver = 535.309.01
GPU = NVIDIA RTX A4500
memory = 20470 MiB
```

## Next Runs

Current smoke run family:

```text
R100: PyTorch data loader smoke
R101: one-step forward/loss smoke
R102: tiny overfit on 5-10 short episodes
R103: short rollout smoke
R104: checkpoint save/load smoke
R105: CUDA-required forward/loss smoke
R106: CUDA-required tiny-overfit smoke
R107: one-step train/validation/checkpoint
R108: rollout eval from R107 checkpoint
R109: checkpoint resume + eval-only smoke
R110: black-box direct TCN baseline small training
R111: base hybrid small training
R112: matched-budget fair small comparison
R113: component/model variant forward smoke
R114: FT0-FT6 fine-tune adapter smoke
R115: U1 K=3 deep ensemble smoke
```

Current local status:

```text
R100: pass
R101: pass
R102: pass, tiny-overfit loss ratio = 0.850854
R103: pass, 8-step rollout finite fraction = 1.0
R104: pass, checkpoint save/load max abs diff = 0.0
R105: pass, torch_device = cuda
R106: pass, torch_device = cuda, tiny-overfit loss ratio = 0.850852
R107: pass, train loss ratio = 0.987559, validation MSE = 5.879991
R108: pass, rollout RMSE = 9.982708 over 4 validation episodes
R109: pass, resumed step 60 -> 70, validation MSE 5.879991 -> 5.781471
R110: pass, direct TCN black-box train loss ratio = 0.934590, validation MSE = 1.482733
R111: pass, base hybrid train loss ratio = 0.984881, validation MSE = 5.890477
R112: pass, hybrid raw MSE / best black-box raw MSE = 4.447071, rollout ratio = 1.666144
R113: pass, 15 component/model variants forward on CUDA including optional T3-MoE
R114: pass, FT0-FT6 trainability and two fine-tune bucket cells validated
R115: pass, K=3 ensemble MSE = 6.010266, predictive variance = 0.085524
R111 post-rollout queue smoke: pass, rollout RMSE = 5.861144 over 2 episodes x 8 steps
matrix report: generated, 52 pending matrix configs
R046: pass, DS2 extreme dataset smoke generated 18 episodes
R047: pass, DS2 T1/T2/T3-MoE forward smoke
```

Local GPU training scaffolding is now ready through one-step training, rollout eval, resume/eval-only, PyTorch black-box baselines, matched fair comparison, model/component variant smoke, fine-tune adapter smoke, and K=3 ensemble training. These are still development-scale gates, not final training-quality results.

Generated full-matrix config entry point:

```bash
conda run -n deep-sim python -m experiments.torch_config_matrix --write
```

Current generated matrix:

```text
configs/torch_matrix/MANIFEST.json
17 PyTorch ablation configs: R200-R216
35 PyTorch fine-tune configs: R300-R334
```

Recommended next execution block:

```bash
conda run -n deep-sim python -m experiments.experiment_queue \
  --manifest configs/torch_matrix/MANIFEST.json \
  --run-ids R200 R201 R202 R203 R204 R205 R206 R207 R208 R209 R210 R211 R212 R213 R214 R215 R216 \
  --max-retries 1 \
  --skip-success \
  --rollout-eval \
  --state-path runs/queue_state_ablation.json
```

After R200-R216 finishes, regenerate `reports/PYTORCH_MATRIX_REPORT.md/json`, then run R300-R334 fine-tune matrix.
