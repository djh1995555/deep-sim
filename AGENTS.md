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

- Local GPU backend is currently not available.
- `nvidia-smi` fails with `Failed to initialize NVML: Driver/library version mismatch`.
- Until the driver/library mismatch is fixed or a remote GPU target is configured, run only CPU smoke tests such as `R100-R104`.

## Ready Smoke Runs

```bash
conda run -n deep-sim python -m experiments.run --config configs/runs/R100.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R101.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R102.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R103.yaml
conda run -n deep-sim python -m experiments.run --config configs/runs/R104.yaml
```

## Remote GPU

No remote GPU server, Vast.ai instance, or Modal backend is configured yet.
