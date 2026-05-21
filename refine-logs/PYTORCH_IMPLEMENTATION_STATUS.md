# PyTorch Implementation Status

**Date**: 2026-05-21 16:58 CST

## Scope

This file tracks the transition from scaffold experiments to the training-grade PyTorch implementation.

Current completed step:

```text
P1: canonical data entry points
P2: PyTorch Student Model v0 skeleton
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
| `student_model/data.py` | Canonical dataset manifest loader, episode array loader, context vector encoder, optional Torch dataset wrapper. |
| `student_model/torch_model.py` | PyTorch hybrid student skeleton for `E2 + T1 + F1 + S1 + M0-fixed + V2-small + U0`. |

The current model implements the first training target selected by scaffold:

```text
E2 causal TCN encoder
T1 force-level tire residual with friction ellipse projection
F1 bounded Fz residual
S1 bounded steering residual
M0-fixed fixed μ
V2-small Δx after integration residual
U0 single-model log variance head
```

Important limitation:

```text
This is only a forward-pass skeleton.
Training loss, optimizer, rollout training, checkpoint save/load, and report-compatible training runs are not implemented yet.
```

## Verification

Commands run:

```bash
conda run -n deep-sim python -m experiments.materialize_data --mode symlink
conda run -n deep-sim python -m compileall experiments student_model tests
conda run -n deep-sim python -m unittest tests.test_canonical_data tests.test_student_model
git diff --check
```

Result:

```text
canonical data test: pass
student model smoke test: skipped because PyTorch is not installed in the current deep-sim environment
compileall: pass
diff check: pass
```

## Dependency Status

`environment.yml` and `requirements.txt` now declare PyTorch.

Current local `deep-sim` environment does not yet have PyTorch installed. Before running R100 training smoke tests, update the environment:

```bash
conda env update -n deep-sim -f environment.yml
python -m pip install -r requirements.txt
```

or install the project-preferred CUDA/CPU PyTorch build manually.

## Next Runs

Recommended next run family:

```text
R100: PyTorch data loader smoke
R101: one-step forward/loss smoke
R102: tiny overfit on 5-10 short episodes
R103: short rollout smoke
R104: checkpoint save/load smoke
```

Only after R100-R104 pass should training-grade B3/B4/B5/B6 experiments start.
