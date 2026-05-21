# Canonical Datasets

This directory contains stable dataset entry points for training code.

Current canonical paths:

| Path | Source | Purpose |
| --- | --- | --- |
| `data/ds1_v1` | `runs/R000g_dataset_split_generation/artifacts/ds1` | Main DS1 scaffold dataset for base training and held-out evaluation. |
| `data/ds1_proxy_ft_v1` | `runs/R038_finetune_FT0/artifacts/ds1_proxy` | Proxy target-window dataset for fine-tune scaffolds. |

The current entries are symlinks created by:

```bash
conda run -n deep-sim python -m experiments.materialize_data --mode symlink
```

Use `--mode copy` only when a backend cannot follow symlinks.
