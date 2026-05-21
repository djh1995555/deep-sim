# Initial Experiment Results

**Date**: 2026-05-21 14:56:35 CST
**Plan**: `refine-logs/EXPERIMENT_PLAN.md`
**Tracker**: `refine-logs/EXPERIMENT_TRACKER.md`

## Results by Milestone

### M0: Teacher Simulator Sanity

| Run | System | Primary Metric | Value | Status |
|---|---|---:|---:|---|
| R000 | high-fidelity teacher simulator v0 | episodes_generated | 8 | DONE |
| R000a | teacher tire + suspension | sign_checks_passed | 1 | DONE |
| R000b | teacher road μ map | road_coverage_score | 3 | DONE |
| R000c | actuator + sensor model | sensor_actuator_checks_passed | 1 | DONE |
| R000d | dataset export + schema validator | leakage_checks_passed | 1 | DONE |

### M1: DS1 Scenario/Data Generation

| Run | System | Primary Metric | Value | Status |
|---|---|---:|---:|---|
| R000e | DS1 scenario matrix generator | full_matrix_count | 700 | DONE |
| R000f | vehicle/config randomization | vehicle_config_count | 4 | DONE |
| R000g | split manifest builder | split_roles_covered | 6 | DONE |
| R000h | dataset QA | dataset_qa_passed | 1 | DONE |

### M2: Data/Physics Sanity

| Run | System | Primary Metric | Value | Status |
|---|---|---:|---:|---|
| R001 | schema / field role check | role_schema_checks_passed | 1 | DONE |
| R002 | teacher physical consistency | physical_consistency_passed | 1 | DONE |
| R003 | timestamp / dt alignment | dt_alignment_passed | 1 | DONE |
| R004 | derived slip / steering / wheel checks | derived_quantities_passed | 1 | DONE |
| R004a | tiny learnability overfit proxy | tiny_learnability_passed | 1 | DONE |
| R004b | physics-only rollout smoke | physics_rollout_smoke_passed | 1 | DONE |

### M3: Sim-to-Real Proxy

| Run | System | Primary Metric | Value | Status |
|---|---|---:|---:|---|
| R004c | proxy perturbation profiles | proxy_profiles_passed | 1 | DONE |
| R004d | proxy target windows | proxy_target_windows_passed | 1 | DONE |
| R004e | proxy distribution sanity | proxy_distribution_passed | 1 | DONE |

### M4: Baseline Scaffolds

| Run | System | Primary Metric | Value | Status |
|---|---|---:|---:|---|
| R005 | physics-only baseline | physics_only_baseline_passed | 1 | DONE |
| R006 | black-box baseline | black_box_baseline_passed | 1 | DONE |
| R007 | baseline fairness audit | baseline_fairness_passed | 1 | DONE |
| R008 | baseline rollout report | baseline_report_passed | 1 | DONE |

Generated artifacts:

```text
runs/R000_teacher_simulator_minimal/
runs/R000a_tire_load_validation/
runs/R000b_road_scenario_generation/
runs/R000c_sensor_actuator_realism/
runs/R000d_dataset_export_split/
runs/R000e_scenario_matrix_v1/
runs/R000f_vehicle_parameter_randomization/
runs/R000g_dataset_split_generation/
runs/R000h_dataset_qa/
runs/R001_schema_field_role_check/
runs/R002_teacher_physical_consistency/
runs/R003_time_dt_alignment/
runs/R004_derived_physical_quantities/
runs/R004a_tiny_learnability/
runs/R004b_physics_rollout_smoke/
runs/R004c_proxy_perturbation_profiles/
runs/R004d_proxy_target_windows/
runs/R004e_proxy_distribution_sanity/
runs/R005_physics_only_baseline/
runs/R006_black_box_baseline/
runs/R007_baseline_fairness_audit/
runs/R008_baseline_rollout_report/
reports/B0_teacher.md
reports/B3_baselines.md
```

## Validation Summary

DS0 smoke dataset contains 8 representative episodes covering:

```text
single μ
split μ
transition μ
acceleration
braking
left/right steering
split-μ braking yaw sign
```

Validator results:

```text
schema_checks_passed = 1
leakage_checks_passed = 1
sign_checks_passed = 1
has_single_mu = 1
has_split_mu = 1
has_transition_mu = 1
```

DS1 scaffold results:

```text
full_matrix_count = 700
sampled_episode_count = 120
sampled_group_counts = CG-SINGLE 40 / CG-SPLIT 40 / CG-TRANSITION 40
duplicate_episode_count = 0
vehicle_config_count = 4
vehicle_family_count = 3
split_roles = train / validation / test / held-out / fine-tune / test-window
fine_tune_buckets = FTD1-FTD5
schema_checks_passed = 1
leakage_checks_passed = 1
dataset_qa_passed = 1
```

M2 sanity results:

```text
role_schema_checks_passed = 1
physical_consistency_passed = 1
dt_alignment_passed = 1
derived_quantities_passed = 1
tiny_learnability_passed = 1
physics_rollout_smoke_passed = 1
```

M3 proxy results:

```text
proxy_profile_count = 3
proxy_profile_min_abs_magnitude = 0.053618551195797684
proxy_profile_max_abs_magnitude = 0.14445175823877
proxy_target_window_count = 81
proxy_target_window_overlap_count = 0
target_window_role_counts = target_train 27 / target_validation 27 / target_test 27
proxy_distribution_shift_score = 0.08710456487873842
proxy_distribution_paired_episode_count = 81
```

M4 baseline scaffold results:

```text
physics_only_baseline_passed = 1
black_box_baseline_passed = 1
black_box_variant_count = 4
baseline_fairness_passed = 1
baseline_report_passed = 1
report_horizons = 25 / 50 / 100 steps
report_splits = train / validation / test / held-out / test-window
```

R008 representative rollout RMSE:

```text
physics-only test 100 steps = 0.251702
BB-MLP test 100 steps = 0.403985
BB-GRU test 100 steps = 0.752756
BB-TCN test 100 steps = 0.772581
BB-NBEATSx test 100 steps = 0.779350
```

Important implementation note:

```text
R001 found duplicate DS1 episode IDs in the first M1 scaffold. The DS1 balanced sampler was fixed to deduplicate matrix items selected by longitudinal and lateral coverage passes, then R000e-R000h and R001-R004b were regenerated successfully.
```

## Summary

`R000-R008` are complete and were run under the `deep-sim` Miniforge environment. The project is ready to proceed to `M5: R009-R014`, which should implement the base hybrid training/evaluation scaffold.
