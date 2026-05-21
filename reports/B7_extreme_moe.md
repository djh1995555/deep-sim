# B7 Extreme Handling And MoE Tire Residual Smoke

**Date**: 2026-05-21 20:43 CST

## Status

| Run | Purpose | Primary Metric | Value | Status |
|---|---|---:|---:|---|
| R046 | DS2 extreme dataset smoke | schema_checks_passed | 1 | success |
| R047 | DS2 T1/T2/T3-MoE forward smoke | torch_model_variant_smoke_passed | 1 | success |

## R046 DS2 Dataset

```text
dataset_id = DS2_EXTREME_V0
scenario_set = ds2_extreme
episode_count = 18
full_matrix_count = 6
scenario_group = DS2-EXTREME
split_roles = held-out, test, train, validation
road coverage = single, split, transition
```

The scaffold covers emergency braking, trail braking, power-on exit, fishhook, emergency lane change, and sine-sweep steering cases across dry/wet/snow/ice, split-mu, and transition-mu roads.

## R047 MoE Smoke

The CUDA forward smoke passed for:

```text
hybrid_T1_force
hybrid_T2_param
hybrid_T3_moe
```

`T3-MoE` exposes `tire_moe_weights` and remains a B7/DS2-only optional tire residual variant. It does not change the current first-version base model or the R200-R216 single-factor matrix.

## Interpretation

This is development evidence only. It proves DS2 extreme data generation and T3-MoE model plumbing are available. It does not prove MoE improves large-slip behavior; that requires full DS2 training and comparison against T1/T2 after the main R200-R216 and R300-R334 matrices are run.
