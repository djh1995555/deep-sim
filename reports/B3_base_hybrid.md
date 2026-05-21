# B3 Base Hybrid Scaffold Report

This report is produced by the M5 numpy scaffold for `Base = E1 + T1 + F1 + S1 + M1a + V1 + U0`.

## Success Gates

| Gate | Value |
|---|---:|
| base_hybrid_training_passed | 1 |
| base_seen_config_eval_passed | 1 |
| base_held_out_road_eval_passed | 1 |
| base_held_out_vehicle_eval_passed | 1 |
| base_residual_constraint_audit_passed | 1 |
| base_seed_replication_passed | 1 |

## 100-step RMSE Ratios

| Group | Base / Physics | Base / Black-box |
|---|---:|---:|
| validation | 0.665915 | 1.159502 |
| seen-config combined | 0.697045 | 0.919612 |
| test split | 1.063680 | 0.515253 |
| held-out-road | 0.793733 | 0.856830 |
| held-out-vehicle | 0.760209 | 1.405008 |
| test-window | 0.852502 | 0.932444 |

## Residual Audit

| Metric | Value |
|---|---:|
| residual_to_physics_ratio_mean | 0.967944 |
| residual_dot_smoothness_mean | 12.162744 |
| constraint_violation_rate_mean | 0.000000 |
