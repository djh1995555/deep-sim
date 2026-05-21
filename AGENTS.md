# Deep Sim Experiment Environment

## Local Environment

- Machine: local Linux workstation
- Conda env: `deep-sim`
- Activation command: `conda run -n deep-sim <command>`
- Python smoke command: `conda run -n deep-sim python -m unittest`
- Experiment command template: `conda run -n deep-sim python -m experiments.run --config <config.yaml>`
- Experiment queue template: `conda run -n deep-sim python -m experiments.experiment_queue --manifest configs/torch_matrix/MANIFEST.json`
- Code dir: `/home/mi/vibe/research/deep_sim/codex`
- code_sync: local
- wandb: false

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
  - `conda run -n deep-sim python -m experiments.run --config configs/runs/R105.yaml`
- GPU backward/optimizer smoke run:
  - `conda run -n deep-sim python -m experiments.run --config configs/runs/R106.yaml`
- Small training dev runs:
  - `conda run -n deep-sim python -m experiments.run --config configs/runs/R107.yaml`
  - `conda run -n deep-sim python -m experiments.run --config configs/runs/R108.yaml`
  - `conda run -n deep-sim python -m experiments.run --config configs/runs/R109.yaml`
  - `conda run -n deep-sim python -m experiments.run --config configs/runs/R110.yaml`
  - `conda run -n deep-sim python -m experiments.run --config configs/runs/R111.yaml`
  - `conda run -n deep-sim python -m experiments.run --config configs/runs/R112.yaml`
  - `conda run -n deep-sim python -m experiments.run --config configs/runs/R113.yaml`
  - `conda run -n deep-sim python -m experiments.run --config configs/runs/R114.yaml`
  - `conda run -n deep-sim python -m experiments.run --config configs/runs/R115.yaml`

## Ready Smoke Runs

```bash
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
```

## Generated Training Matrix

```bash
conda run -n deep-sim python -m experiments.torch_config_matrix --write
```

This writes `configs/torch_matrix/R200-R216.yaml` for trainable PyTorch ablations and `configs/torch_matrix/R300-R334.yaml` for FT0-FT6 data-efficiency runs.

## Local Experiment Queue

Use the local queue for matrix runs or any run batch that should keep crash-safe state and per-attempt logs:

```bash
conda run -n deep-sim python -m experiments.experiment_queue \
  --manifest configs/torch_matrix/MANIFEST.json \
  --run-ids R200 R201 R202 \
  --max-retries 1 \
  --skip-success \
  --rollout-eval \
  --state-path runs/queue_state_ablation.json
```

Useful dry run:

```bash
conda run -n deep-sim python -m experiments.experiment_queue \
  --manifest configs/torch_matrix/MANIFEST.json \
  --run-ids R200 R201 \
  --dry-run \
  --state-path runs/queue_state_dryrun.json
```

The queue reads direct configs or `configs/torch_matrix/MANIFEST.json`, supports `--skip-success`, retries, `--stop-on-failure`, and optional post-training rollout evaluation.

## Matrix Report

After queued runs finish, regenerate the matrix report:

```bash
conda run -n deep-sim python -m experiments.matrix_report
```

Outputs:

```text
reports/PYTORCH_MATRIX_REPORT.json
reports/PYTORCH_MATRIX_REPORT.md
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
