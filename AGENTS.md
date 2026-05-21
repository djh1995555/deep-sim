# Deep Sim Experiment Environment

## Local Environment

- Machine: local Linux workstation
- Conda env: `deep-sim`
- Activation command: `conda run -n deep-sim <command>`
- Python smoke command: `conda run -n deep-sim python -m unittest`
- Experiment command template: `conda run -n deep-sim python -m experiments.run --config <config.yaml>`
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
```

## Remote GPU

No remote GPU server, Vast.ai instance, or Modal backend is configured yet.
