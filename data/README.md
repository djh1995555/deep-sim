# Canonical Datasets

This directory contains stable dataset entry points for training code.
Dataset generation is owned by the external simulator project:

```text
/home/mi/vibe/research/simulator
```

Current canonical paths:

| Path | Source | Purpose |
| --- | --- | --- |
| `data/ds1_v1` | `/home/mi/vibe/research/simulator/configs/datasets/ds1_v1.yaml` | Main DS1 scaffold dataset for base training and held-out evaluation. |
| `data/ds1_proxy_ft_v1` | `/home/mi/vibe/research/simulator/configs/datasets/ds1_proxy_ft_v1.yaml` | Proxy target-window dataset for fine-tune scaffolds. |

These entries are real directories, not symlinks into historical run outputs.
Regenerate them with `deep-sim-generate` from the external simulator package, then validate them with:

```bash
conda run -n deep-sim python -m experiments.materialize_data
```

Use `--mode copy --data-root <target>` only when copying the canonical datasets to another location.
