# PyTorch Implementation Status

**Date**: 2026-05-21 19:33 CST

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
| `experiments/torch_config_matrix.py` | Generates full PyTorch ablation and fine-tune config matrix under `configs/torch_matrix/`. |
| `experiments/torch_dev_report.py` | Aggregates R112-R115 results into `reports/PYTORCH_DEV_REPORT.md/json`. |
| `tests/test_torch_training.py` | Runner/config regression tests with a blocked path when PyTorch is absent. |
| `AGENTS.md` | Local run-experiment environment notes, GPU status, and smoke run commands. |

The new runner branch supports:

```text
dataset_source: existing
```

This lets R100-R104 reuse `data/ds1_v1` instead of regenerating DS1 every time.

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
conda run -n deep-sim python -m experiments.torch_config_matrix --write
conda run -n deep-sim python -m experiments.torch_dev_report
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
R113: pass, 14 component/model variants forward on CUDA
R114: pass, FT0-FT6 trainability and two fine-tune bucket cells validated
R115: pass, K=3 ensemble MSE = 6.010266, predictive variance = 0.085524
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
