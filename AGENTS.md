# Deep Sim Experiment Environment

## Local Environment

- Machine: local Linux workstation
- Conda env: `deep-sim`
- Activation command: `conda run -n deep-sim <command>`
- Python smoke command: `conda run -n deep-sim python -m unittest`
- Experiment command template: `conda run -n deep-sim python -m experiments.run --config <config.yaml>`
- Experiment queue template: `conda run -n deep-sim python -m experiments.experiment_queue --manifest configs/experiments/matrix/MANIFEST.json`
- Code dir: `/home/mi/vibe/research/deep_sim/codex`
- code_sync: local
- wandb: false
- Output root: `output/`
  - Training experiment outputs: `output/training/`
  - Closed-loop simulation outputs: `output/simulation/`

## GPU Status

- Local GPU backend is available.
- GPU: NVIDIA RTX A4500, 20 GB.
- Driver: `535.309.01`; `nvidia-smi` reports CUDA runtime compatibility `12.2`.
- Current `deep-sim` PyTorch CUDA status:
  - `torch.__version__ = 2.5.1`
  - `torch.version.cuda = 12.1`
  - `torch.cuda.is_available() = True`
  - `torch.cuda.device_count() = 1`
- Previous root cause observed on 2026-05-21 and fixed by reboot:
  - Loaded kernel module: `535.288.01` from `/proc/driver/nvidia/version`
  - Installed DKMS/user-space driver: `535.309.01`
  - `display-manager`, Xorg, GNOME, Chrome, VS Code, and other desktop processes held `/dev/nvidia*`
  - `/var/run/reboot-required` was present
- GPU smoke run:
  - `conda run -n deep-sim python -m experiments.run --config configs/experiments/p4_gpu_smoke/p4_1_pytorch_gpu_forward_smoke.yaml`
- GPU backward/optimizer smoke run:
  - `conda run -n deep-sim python -m experiments.run --config configs/experiments/p4_gpu_smoke/p4_2_pytorch_gpu_tiny_overfit.yaml`
- Small training dev runs:
  - `conda run -n deep-sim python -m experiments.run --config configs/experiments/p5_training_dev/p5_1_pytorch_one_step_training.yaml`
  - `conda run -n deep-sim python -m experiments.run --config configs/experiments/p5_training_dev/p5_2_pytorch_rollout_eval.yaml`
  - `conda run -n deep-sim python -m experiments.run --config configs/experiments/p5_training_dev/p5_3_pytorch_resume_eval_smoke.yaml`
  - `conda run -n deep-sim python -m experiments.run --config configs/experiments/p5_training_dev/p5_4_pytorch_black_box_tcn_baseline.yaml`
  - `conda run -n deep-sim python -m experiments.run --config configs/experiments/p6_model_dev/p6_1_pytorch_base_model_small_training.yaml`
  - `conda run -n deep-sim python -m experiments.run --config configs/experiments/p6_model_dev/p6_2_pytorch_fair_small_comparison.yaml`
  - `conda run -n deep-sim python -m experiments.run --config configs/experiments/p6_model_dev/p6_3_pytorch_model_variant_smoke.yaml`
  - `conda run -n deep-sim python -m experiments.run --config configs/experiments/p7_adapter_ensemble/p7_1_pytorch_fine_tune_adapter_smoke.yaml`
  - `conda run -n deep-sim python -m experiments.run --config configs/experiments/p7_adapter_ensemble/p7_2_pytorch_deep_ensemble_smoke.yaml`
- DS2 / MoE development smoke runs:
  - `conda run -n deep-sim python -m experiments.run --config configs/experiments/b7_extreme_moe/b7_1_ds2_extreme_dataset_smoke.yaml`
  - `conda run -n deep-sim python -m experiments.run --config configs/experiments/b7_extreme_moe/b7_2_pytorch_ds2_moe_tire_smoke.yaml`

## Ready Smoke Runs

```bash
conda run -n deep-sim python -m experiments.run --config configs/experiments/p3_smoke/p3_1_pytorch_data_loader_smoke.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/p3_smoke/p3_2_pytorch_forward_loss_smoke.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/p3_smoke/p3_3_pytorch_tiny_overfit.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/p3_smoke/p3_4_pytorch_rollout_smoke.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/p3_smoke/p3_5_pytorch_checkpoint_smoke.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/p4_gpu_smoke/p4_1_pytorch_gpu_forward_smoke.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/p4_gpu_smoke/p4_2_pytorch_gpu_tiny_overfit.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/p5_training_dev/p5_1_pytorch_one_step_training.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/p5_training_dev/p5_2_pytorch_rollout_eval.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/p5_training_dev/p5_3_pytorch_resume_eval_smoke.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/p5_training_dev/p5_4_pytorch_black_box_tcn_baseline.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/p6_model_dev/p6_1_pytorch_base_model_small_training.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/p6_model_dev/p6_2_pytorch_fair_small_comparison.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/p6_model_dev/p6_3_pytorch_model_variant_smoke.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/p7_adapter_ensemble/p7_1_pytorch_fine_tune_adapter_smoke.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/p7_adapter_ensemble/p7_2_pytorch_deep_ensemble_smoke.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/b7_extreme_moe/b7_1_ds2_extreme_dataset_smoke.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/b7_extreme_moe/b7_2_pytorch_ds2_moe_tire_smoke.yaml
```

## Generated Training Matrix

```bash
conda run -n deep-sim python -m experiments.torch_config_matrix --write
```

This writes trainable PyTorch ablation configs under `configs/experiments/ablation/`, fine-tune data-efficiency configs under `configs/experiments/finetune/`, and the queue manifest at `configs/experiments/matrix/MANIFEST.json`.

## Local Experiment Queue

Use the local queue for matrix runs or any run batch that should keep crash-safe state and per-attempt logs:

```bash
conda run -n deep-sim python -m experiments.experiment_queue \
  --manifest configs/experiments/matrix/MANIFEST.json \
  --run-ids R200 R201 R202 \
  --max-retries 1 \
  --skip-success \
  --rollout-eval \
  --state-path output/training/queue_state_ablation.json
```

Useful dry run:

```bash
conda run -n deep-sim python -m experiments.experiment_queue \
  --manifest configs/experiments/matrix/MANIFEST.json \
  --run-ids R200 R201 \
  --dry-run \
  --state-path output/training/queue_state_dryrun.json
```

The queue reads direct configs or `configs/experiments/matrix/MANIFEST.json`, supports `--skip-success`, retries, `--stop-on-failure`, and optional post-training rollout evaluation.

## Matrix Report

After queued runs finish, regenerate the matrix report:

```bash
conda run -n deep-sim python -m experiments.matrix_report
```

Outputs:

```text
output/training/reports/PYTORCH_MATRIX_REPORT.json
output/training/reports/PYTORCH_MATRIX_REPORT.md
```

Current generated report status: all `R200-R216` and `R300-R334` matrix configs are pending until full training is launched.

## Real Data Adapter

CSV real-vehicle episodes can be converted into the canonical dataset format with:

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

The adapter writes `manifest.json`, episode `.npz` / sidecar `.json`, and `adapter_summary.json`; the result is validated through the canonical dataset validator.

## Remote GPU

No remote GPU server, Vast.ai instance, or Modal backend is configured yet.
