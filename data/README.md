# Canonical Datasets

This directory contains stable dataset entry points for training code.

Current canonical paths:

| Path | Source | Purpose |
| --- | --- | --- |
| `data/ds1_v1` | local directory | Main DS1 scaffold dataset for base training and held-out evaluation. |
| `data/ds1_proxy_ft_v1` | local directory | Proxy target-window dataset for fine-tune scaffolds. |

These entries are real directories, not symlinks into historical run outputs. Validate them with:

```bash
conda run -n deep-sim python -m experiments.materialize_data
```

Use `--mode copy --data-root <target>` only when copying the canonical datasets to another location.
