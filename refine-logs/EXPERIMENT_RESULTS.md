# Initial Experiment Results

**Date**: 2026-05-21 20:43:11 CST
**Plan**: `refine-logs/EXPERIMENT_PLAN.md`
**Tracker**: `refine-logs/EXPERIMENT_TRACKER.md`

## Results by Milestone

### M0: External Simulator Data Sanity

| Run | System | Primary Metric | Value | Status |
|---|---|---:|---:|---|
| R000 | external simulator data generator | episodes_generated | 8 | DONE |
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

### M5: Base Hybrid Scaffold

| Run | System | Primary Metric | Value | Status |
|---|---|---:|---:|---|
| R009 | base hybrid training | base_hybrid_training_passed | 1 | DONE |
| R010 | base seen-config evaluation | base_seen_config_eval_passed | 1 | DONE |
| R011 | base held-out road evaluation | base_held_out_road_eval_passed | 1 | DONE |
| R012 | base held-out vehicle evaluation | base_held_out_vehicle_eval_passed | 1 | DONE |
| R013 | base residual/constraint audit | base_residual_constraint_audit_passed | 1 | DONE |
| R014 | base seed replication | base_seed_replication_passed | 1 | DONE |

### M6: Component Ablation Scaffold

| Run | System | Primary Metric | Value | Status |
|---|---|---:|---:|---|
| R015-R018 | tire residual ablation | ablation_run_passed | 1 | DONE |
| R019-R021 | Fz residual ablation | ablation_run_passed | 1 | DONE |
| R022-R023 | steering ablation | ablation_run_passed | 1 | DONE |
| R024-R027 | MuHead ablation | ablation_run_passed | 1 | DONE |
| R027a-R027c | shared encoder ablation | ablation_run_passed | 1 | DONE |
| R028-R031 | vehicle residual ablation | ablation_run_passed | 1 | DONE |
| R032-R033 | uncertainty ablation | ablation_run_passed | 1 | DONE |

### M7: Cross-Vehicle / Cross-Config Scaffold

| Run | System | Primary Metric | Value | Status |
|---|---|---:|---:|---|
| R034 | base cross-generalization reference | base_held_out_vehicle_eval_passed | 1 | DONE |
| R035 | selected final single scaffold | cross_generalization_passed | 1 | DONE |
| R036 | selected scaffold + U1 ensemble wrapper | cross_generalization_passed | 1 | DONE |

### M8: Final Single Model Freeze

| Run | System | Primary Metric | Value | Status |
|---|---|---:|---:|---|
| R037 | final single model checkpoint descriptor | final_single_model_freeze_passed | 1 | DONE |

### M9: Target Fine-Tune Data Efficiency Scaffold

| Run | System | Primary Metric | Value | Status |
|---|---|---:|---:|---|
| R038 | FT0 no fine-tune | fine_tune_run_passed | 1 | DONE |
| R039 | FT1 VehicleParamAdapter | fine_tune_run_passed | 1 | DONE |
| R040 | FT2 MuHead adapter | fine_tune_run_passed | 1 | DONE |
| R041 | FT3 FzResidual adapter | fine_tune_run_passed | 1 | DONE |
| R042 | FT4 TireResidual adapter | fine_tune_run_passed | 1 | DONE |
| R043 | FT5 steering residual adapter | fine_tune_run_passed | 1 | DONE |
| R044 | FT6 full model fine-tune scaffold | fine_tune_run_passed | 1 | DONE |
| R045 | B6 fine-tune summary | fine_tune_summary_passed | 1 | DONE |

### P3-P6: PyTorch Training Development Gates

| Run | System | Primary Metric | Value | Status |
|---|---|---:|---:|---|
| R100 | PyTorch data loader smoke | torch_data_loader_smoke_passed | 1 | DONE |
| R101 | PyTorch forward/loss smoke | torch_forward_loss_smoke_passed | 1 | DONE |
| R102 | PyTorch tiny overfit smoke | torch_tiny_overfit_passed | 1 | DONE |
| R103 | PyTorch rollout smoke | torch_rollout_smoke_passed | 1 | DONE |
| R104 | PyTorch checkpoint smoke | torch_checkpoint_smoke_passed | 1 | DONE |
| R105 | CUDA forward/loss smoke | torch_forward_loss_smoke_passed | 1 | DONE |
| R106 | CUDA backward/optimizer tiny overfit | torch_tiny_overfit_passed | 1 | DONE |
| R107 | CUDA one-step train/validation/checkpoint | torch_one_step_training_passed | 1 | DONE |
| R108 | Rollout eval from R107 checkpoint | torch_rollout_eval_passed | 1 | DONE |
| R109 | Resume + eval-only smoke | torch_resume_eval_passed | 1 | DONE |
| R110 | Direct TCN black-box PyTorch baseline | torch_black_box_training_passed | 1 | DONE |
| R111 | Base hybrid small PyTorch training | torch_one_step_training_passed | 1 | DONE |
| R112 | Matched fair small comparison | torch_fair_compare_passed | 1 | DONE |
| R113 | Component/model variant smoke | torch_model_variant_smoke_passed | 1 | DONE |
| R114 | Fine-tune adapter smoke | torch_fine_tune_smoke_passed | 1 | DONE |
| R115 | Deep ensemble smoke | torch_ensemble_training_passed | 1 | DONE |

### M10: DS2 Extreme / MoE Development Smoke

| Run | System | Primary Metric | Value | Status |
|---|---|---:|---:|---|
| R046 | DS2 extreme dataset smoke | schema_checks_passed | 1 | DONE |
| R047 | DS2 T1/T2/T3-MoE forward smoke | torch_model_variant_smoke_passed | 1 | DONE |

### P7-P10: Experiment Engineering Completion

| Item | System | Primary Metric | Value | Status |
|---|---|---:|---:|---|
| Q1 | local experiment queue | queue_smoke_passed | 1 | DONE |
| Q2 | R111 post-rollout via queue | torch_rollout_eval_passed | 1 | DONE |
| Q3 | PyTorch matrix report | pending_matrix_config_count | 52 | DONE |
| Q4 | real-data CSV adapter | canonical_adapter_test_passed | 1 | DONE |
| Q5 | training governance | best_checkpoint_early_stop_scheduler_supported | 1 | DONE |

Key R107-R111 development metrics:

```text
R107 hybrid one-step train loss ratio = 0.987559, validation MSE = 5.879991
R108 rollout RMSE = 9.982708 over 4 validation episodes
R109 resume step = 60 -> 70, validation MSE = 5.879991 -> 5.781471
R110 direct TCN black-box loss ratio = 0.934590, validation MSE = 1.482733
R111 base hybrid small-train loss ratio = 0.984881, validation MSE = 5.890477
R112 fair comparison: hybrid/best black-box raw MSE ratio = 4.447071, rollout ratio = 1.666144
R113 variant smoke: 15 component/model variants passed forward checks, including optional T3-MoE
R114 fine-tune smoke: FT0-FT6 trainability passed over 2 data bucket cells
R115 ensemble smoke: K=3 ensemble MSE = 6.010266, predictive variance = 0.085524
R111 post-rollout queue smoke: rollout RMSE = 5.861144 over 2 validation episodes x 8 steps, constraint violation rate = 0.0
R046 DS2 extreme smoke: 18 episodes, 6 extreme matrix entries, single/split/transition road coverage
R047 DS2 MoE smoke: T1/T2/T3-MoE CUDA forward checks passed
```

Interpretation:

```text
R107-R111 validate the PyTorch training infrastructure, not final model quality.
R112 confirms the current small direct black-box family is still stronger than the small hybrid on raw one-step MSE and rollout RMSE. The hybrid has lower normalized one-step loss in this small run, but final claims still require full fair training, rollout/cross-split evaluation, and model/loss tuning.
```

PyTorch development report:

```text
output/training/reports/PYTORCH_DEV_REPORT.md
output/training/reports/PYTORCH_DEV_REPORT.json
output/training/reports/PYTORCH_MATRIX_REPORT.md
output/training/reports/PYTORCH_MATRIX_REPORT.json
output/training/reports/B7_extreme_moe.md
```

Generated full training matrix:

```text
configs/experiments/matrix/MANIFEST.json
R200-R216: trainable PyTorch single-factor ablation configs
R300-R334: FT0-FT6 × five fine-tune data bucket configs
```

Execution tooling added:

```text
experiments/experiment_queue.py: local queued execution for config lists or manifest-driven matrix runs
experiments/matrix_report.py: matrix status/result aggregation
experiments/real_data_adapter.py: CSV real-vehicle episode to canonical dataset conversion
experiments/torch_training.py: best checkpoint, early stopping, LR scheduler, nonfinite guard, filtered rollout eval
```

Generated artifacts:

```text
output/training/R000_dataset_generation_minimal/
output/training/R000a_tire_load_validation/
output/training/R000b_road_scenario_generation/
output/training/R000c_sensor_actuator_realism/
output/training/R000d_dataset_export_split/
output/training/R000e_scenario_matrix_v1/
output/training/R000f_vehicle_parameter_randomization/
output/training/R000g_dataset_split_generation/
output/training/R000h_dataset_qa/
output/training/R001_schema_field_role_check/
output/training/R002_teacher_physical_consistency/
output/training/R003_time_dt_alignment/
output/training/R004_derived_physical_quantities/
output/training/R004a_tiny_learnability/
output/training/R004b_physics_rollout_smoke/
output/training/R004c_proxy_perturbation_profiles/
output/training/R004d_proxy_target_windows/
output/training/R004e_proxy_distribution_sanity/
output/training/R005_physics_only_baseline/
output/training/R006_black_box_baseline/
output/training/R007_baseline_fairness_audit/
output/training/R008_baseline_rollout_report/
output/training/R009_base_hybrid_training/
output/training/R010_base_seen_config_eval/
output/training/R011_base_held_out_road_eval/
output/training/R012_base_held_out_vehicle_eval/
output/training/R013_base_residual_constraint_audit/
output/training/R014_base_seed_replication/
output/training/R015_ablation_tire_T0/
output/training/R016_ablation_tire_T1/
output/training/R017_ablation_tire_T1_no_proj/
output/training/R018_ablation_tire_T2/
output/training/R019_ablation_fz_F0/
output/training/R020_ablation_fz_F1/
output/training/R021_ablation_fz_F2/
output/training/R022_ablation_steering_S0/
output/training/R023_ablation_steering_S1/
output/training/R024_ablation_mu_M0_fixed/
output/training/R025_ablation_mu_M1a/
output/training/R026_ablation_mu_M1b/
output/training/R027_ablation_mu_M2_oracle/
output/training/R027a_ablation_encoder_E1/
output/training/R027b_ablation_encoder_E2/
output/training/R027c_ablation_encoder_E3/
output/training/R028_ablation_vehicle_V0/
output/training/R029_ablation_vehicle_V1/
output/training/R030_ablation_vehicle_V1_large/
output/training/R031_ablation_vehicle_V2_small/
output/training/R032_ablation_uncertainty_U0/
output/training/R033_ablation_uncertainty_U1/
output/training/R034_cross_generalization_base/
output/training/R035_cross_generalization_selected_single/
output/training/R036_cross_generalization_selected_ensemble/
output/training/R037_final_single_model_freeze/
output/training/R038_finetune_FT0/
output/training/R039_finetune_FT1_vehicle_param_adapter/
output/training/R040_finetune_FT2_mu_head/
output/training/R041_finetune_FT3_fz_residual/
output/training/R042_finetune_FT4_tire_residual/
output/training/R043_finetune_FT5_steering_residual/
output/training/R044_finetune_FT6_full_model/
output/training/R045_finetune_summary/
output/training/R100_pytorch_data_loader_smoke/
output/training/R101_pytorch_forward_loss_smoke/
output/training/R102_pytorch_tiny_overfit/
output/training/R103_pytorch_rollout_smoke/
output/training/R104_pytorch_checkpoint_smoke/
output/training/R105_pytorch_gpu_forward_smoke/
output/training/R106_pytorch_gpu_tiny_overfit/
output/training/R107_pytorch_one_step_training/
output/training/R108_pytorch_rollout_eval/
output/training/R109_pytorch_resume_eval_smoke/
output/training/R110_pytorch_black_box_tcn_baseline/
output/training/R111_pytorch_base_model_small_training/
output/training/R112_pytorch_fair_small_comparison/
output/training/R113_pytorch_model_variant_smoke/
output/training/R114_pytorch_fine_tune_adapter_smoke/
output/training/R115_pytorch_deep_ensemble_smoke/
output/training/R046_ds2_extreme_dataset_smoke/
output/training/R047_pytorch_ds2_moe_tire_smoke/
configs/experiments/
output/training/reports/PYTORCH_DEV_REPORT.md
output/training/reports/PYTORCH_DEV_REPORT.json
output/training/reports/PYTORCH_MATRIX_REPORT.md
output/training/reports/PYTORCH_MATRIX_REPORT.json
output/training/queue_state_smoke.json
output/training/R111_pytorch_base_model_small_training/artifacts/post_rollout_eval/
output/training/reports/B0_data_generation.md
output/training/reports/B3_baselines.md
output/training/reports/B3_base_hybrid.md
output/training/reports/B4_ablations.md
output/training/reports/B4_ablation_summary.json
output/training/reports/B5_cross_generalization.md
output/training/reports/B5_cross_generalization_summary.json
output/training/reports/B6_fine_tune.md
output/training/reports/B6_fine_tune_summary.json
output/training/reports/B7_extreme_moe.md
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

M5 base hybrid scaffold results:

```text
base_hybrid_training_passed = 1
base_seen_config_eval_passed = 1
base_held_out_road_eval_passed = 1
base_held_out_vehicle_eval_passed = 1
base_residual_constraint_audit_passed = 1
base_seed_replication_passed = 1
base_train_transition_count = 5400
```

R014 3-seed 100-step RMSE ratios:

```text
validation base / physics = 0.665915 ± 0.071089
seen-config combined base / physics = 0.697045 ± 0.076608
held-out road base / physics = 0.793733 ± 0.057844
held-out vehicle base / physics = 0.760209 ± 0.052398
test-window base / physics = 0.852502 ± 0.044349
held-out road base / black-box = 0.856830
test-window base / black-box = 0.932444
```

R014 residual/constraint audit:

```text
base_residual_to_physics_ratio_mean = 0.967944
base_residual_dot_smoothness_mean = 12.162744
base_constraint_violation_rate_mean = 0.0
```

M6 component ablation scaffold results:

```text
R015-R033 all completed with ablation_run_passed = 1
report = output/training/reports/B4_ablations.md
summary_json = output/training/reports/B4_ablation_summary.json
```

Deployable scaffold variants with lowest validation base/physics ratio by family:

```text
Tire: T1, validation ratio = 0.700753
Fz: F1, validation ratio = 0.700753
Steering: S1, validation ratio = 0.700753
Mu: M0-fixed, validation ratio = 0.697565
Encoder: E2, validation ratio = 0.629220
Vehicle residual: V2-small, validation ratio = 0.685365
Uncertainty: U1, validation ratio = 0.665915
```

Interpretation limit:

```text
M6 is still a numpy scaffold. It proves ablation plumbing, split discipline, metric generation, and single-factor run organization. It does not prove final PyTorch component superiority.
F2 and M2-oracle are simulator-only teacher-label/oracle runs and are excluded from deployable family winners.
```

M7 cross-generalization scaffold results:

```text
R034 Base held-out vehicle / physics = 0.742754
R034 Base held-out vehicle / black-box = 1.397572

R035 Final single held-out vehicle / physics = 0.676170
R035 Final single held-out road / physics = 0.721453
R035 Final single seen-config / physics = 0.603213
R035 Final single held-out vehicle / black-box = 1.283654
R035 cross_generalization_passed = 1
R035 cross_generalization_claim_supported = 0

R036 Final + U1 held-out vehicle / physics = 0.674862
R036 Final + U1 held-out road / physics = 0.728261
R036 Final + U1 held-out vehicle / black-box = 1.246397
R036 cross_generalization_passed = 1
R036 cross_generalization_claim_supported = 0
```

M7 interpretation:

```text
The selected scaffold improves over physics-only on held-out vehicle/config and held-out road.
It does not yet support the stronger claim that the selected hybrid beats the black-box held-out-vehicle baseline.
That stronger claim must be re-tested with the real PyTorch implementation and training-scale data.
```

M8 final single scaffold freeze:

```text
checkpoint_descriptor = output/training/R037_final_single_model_freeze/checkpoints/final_single_model_scaffold.json
selected_single_model = E2 + T1 + F1 + S1 + M0-fixed + V2-small + U0
optional_uncertainty_wrapper = U1
final_single_model_freeze_passed = 1
```

M9 fine-tune data efficiency scaffold results:

```text
FT0@FTD0 test RMSE = 0.598498
best small-module FT = FT5@FTD1
best small-module test RMSE = 0.518854
best small-module relative to FT0 = 0.866927
best full-model FT relative to FT0 = 2.458700
completed variants = 7
completed cells = 31
fine_tune_summary_passed = 1
```

M9 interpretation:

```text
The scaffold validates the FT0-FT6 × FTD0-FTD5 matrix, target-window split handling, and report generation.
In this proxy scaffold, FT5 steering residual adapter is the best small-module adapter, while FT6 full-model fine-tune overfits/overcorrects badly.
This is a scaffold signal only; it is not a final adapter ranking.
```

Important implementation note:

```text
R001 found duplicate DS1 episode IDs in the first M1 scaffold. The DS1 balanced sampler was fixed to deduplicate matrix items selected by longitudinal and lateral coverage passes, then R000e-R000h and R001-R004b were regenerated successfully.
```

## Summary

`R000-R045` are complete and were run under the `deep-sim` Miniforge environment. The required scaffold bridge covers teacher/data generation, sanity checks, baselines, base hybrid, component ablations, cross-vehicle/config generalization, final single-model freeze, and target fine-tune data efficiency.

`R100-R115` are complete PyTorch development gates. They validate canonical data loading, model forward/loss, tiny overfit, rollout, checkpoint save/load, CUDA forward/backward, one-step training, resume/eval-only, direct black-box baseline, fair small comparison, component variants, fine-tune adapter trainability, and K=3 ensemble plumbing.

The remaining development scaffolding is now also complete: local queue, matrix report, real-data CSV adapter, training governance, queue-triggered post-rollout evaluation, DS2 extreme data scaffold, and T3-MoE tire residual forward path are implemented and tested. `R200-R216` and `R300-R334` are ready to run from `configs/experiments/matrix/MANIFEST.json`; they are pending full training and should be treated as the next execution step, not as completed model-quality evidence.

`R048+` DS2 MoE full training/eval is ready as a NICE second-phase execution item. It should run only after the main DS1 ablation/fine-tune matrices clarify the stable first-version model.
