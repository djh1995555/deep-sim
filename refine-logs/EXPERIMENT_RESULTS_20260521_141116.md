# Initial Experiment Results

**Date**: 2026-05-21 14:11:16 CST
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
reports/B0_teacher.md
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

Important implementation note:

```text
R001 found duplicate DS1 episode IDs in the first M1 scaffold. The DS1 balanced sampler was fixed to deduplicate matrix items selected by longitudinal and lateral coverage passes, then R000e-R000h and R001-R004b were regenerated successfully.
```

## Summary

`R000-R004b` are complete and were run under the `deep-sim` Miniforge environment. The project is ready to proceed to `M3: R004c-R004e`, which should add sim-to-real proxy perturbation profiles and target-window generation.
