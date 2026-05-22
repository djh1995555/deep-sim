# Experiment Code Review

**Date**: 2026-05-21 16:34:31 CST
**Scope**: `R000-R045` teacher/data generation, sanity, proxy, baseline, base hybrid, component ablation, cross-generalization, final freeze, and fine-tune scaffold implementation
**Mode**: local-only fallback

## Review Boundary

Reviewed files:

```text
simulator/
experiments/run.py
experiments/sanity.py
experiments/baselines.py
experiments/hybrid.py
experiments/ablation_report.py
experiments/cross_generalization_report.py
experiments/fine_tune.py
configs/datasets/ds0_minimal.yaml
configs/experiments/b0_data_generation/*.yaml
configs/experiments/b1_sanity/b1_1_schema_field_role_check.yaml
configs/experiments/b1_sanity/b1_2_teacher_physical_consistency.yaml
configs/experiments/b1_sanity/b1_3_time_dt_alignment.yaml
configs/experiments/b1_sanity/b1_4_derived_physical_quantities.yaml
configs/experiments/b1_sanity/b1_5_tiny_learnability.yaml
configs/experiments/b1_sanity/b1_6_physics_rollout_smoke.yaml
configs/experiments/b2_proxy/b2_1_proxy_perturbation_profiles.yaml
configs/experiments/b2_proxy/b2_2_proxy_target_windows.yaml
configs/experiments/b2_proxy/b2_3_proxy_distribution_sanity.yaml
configs/experiments/b3_baselines_base/b3_1_physics_only_baseline.yaml
configs/experiments/b3_baselines_base/b3_2_black_box_baseline.yaml
configs/experiments/b3_baselines_base/b3_3_baseline_fairness_audit.yaml
configs/experiments/b3_baselines_base/b3_4_baseline_rollout_report.yaml
configs/experiments/b3_baselines_base/b3_5_base_hybrid_training.yaml
configs/experiments/b3_baselines_base/b3_6_base_seen_config_eval.yaml
configs/experiments/b3_baselines_base/b3_7_base_held_out_road_eval.yaml
configs/experiments/b3_baselines_base/b3_8_base_held_out_vehicle_eval.yaml
configs/experiments/b3_baselines_base/b3_9_base_residual_constraint_audit.yaml
configs/experiments/b3_baselines_base/b3_10_base_seed_replication.yaml
configs/experiments/b4_ablation_scaffold/b4_1_ablation_tire_t0.yaml
configs/experiments/b4_ablation_scaffold/b4_1_ablation_tire_t1.yaml
configs/experiments/b4_ablation_scaffold/b4_1_ablation_tire_t1_no_proj.yaml
configs/experiments/b4_ablation_scaffold/b4_1_ablation_tire_t2.yaml
configs/experiments/b4_ablation_scaffold/b4_2_ablation_fz_f0.yaml
configs/experiments/b4_ablation_scaffold/b4_2_ablation_fz_f1.yaml
configs/experiments/b4_ablation_scaffold/b4_2_ablation_fz_f2.yaml
configs/experiments/b4_ablation_scaffold/b4_3_ablation_steering_s0.yaml
configs/experiments/b4_ablation_scaffold/b4_3_ablation_steering_s1.yaml
configs/experiments/b4_ablation_scaffold/b4_4_ablation_mu_m0_fixed.yaml
configs/experiments/b4_ablation_scaffold/b4_4_ablation_mu_m1a.yaml
configs/experiments/b4_ablation_scaffold/b4_4_ablation_mu_m1b.yaml
configs/experiments/b4_ablation_scaffold/b4_4_ablation_mu_m2_oracle.yaml
configs/experiments/b4_ablation_scaffold/b4_5_ablation_encoder_e1.yaml
configs/experiments/b4_ablation_scaffold/b4_5_ablation_encoder_e2.yaml
configs/experiments/b4_ablation_scaffold/b4_5_ablation_encoder_e3.yaml
configs/experiments/b4_ablation_scaffold/b4_7_ablation_vehicle_v0.yaml
configs/experiments/b4_ablation_scaffold/b4_7_ablation_vehicle_v1.yaml
configs/experiments/b4_ablation_scaffold/b4_7_ablation_vehicle_v1_large.yaml
configs/experiments/b4_ablation_scaffold/b4_7_ablation_vehicle_v2_small.yaml
configs/experiments/b4_ablation_scaffold/b4_6_ablation_uncertainty_u0.yaml
configs/experiments/b4_ablation_scaffold/b4_6_ablation_uncertainty_u1.yaml
configs/experiments/b5_generalization/b5_1_cross_generalization_base.yaml
configs/experiments/b5_generalization/b5_2_cross_generalization_selected_single.yaml
configs/experiments/b5_generalization/b5_3_cross_generalization_selected_ensemble.yaml
configs/experiments/b5_generalization/b5_4_final_single_model_freeze.yaml
configs/experiments/b6_finetune_scaffold/b6_1_finetune_ft0.yaml
configs/experiments/b6_finetune_scaffold/b6_2_finetune_ft1_vehicle_param_adapter.yaml
configs/experiments/b6_finetune_scaffold/b6_3_finetune_ft2_mu_head.yaml
configs/experiments/b6_finetune_scaffold/b6_4_finetune_ft3_fz_residual.yaml
configs/experiments/b6_finetune_scaffold/b6_5_finetune_ft4_tire_residual.yaml
configs/experiments/b6_finetune_scaffold/b6_6_finetune_ft5_steering_residual.yaml
configs/experiments/b6_finetune_scaffold/b6_7_finetune_ft6_full_model.yaml
configs/experiments/b6_finetune_scaffold/b6_8_finetune_summary.yaml
configs/datasets/ds1_proxy_v1.yaml
configs/datasets/ds1_proxy_ft_v1.yaml
tests/test_simulator.py
output/training/reports/B0_teacher.md
output/training/reports/B3_baselines.md
output/training/reports/B3_base_hybrid.md
output/training/reports/B4_ablations.md
output/training/reports/B5_cross_generalization.md
output/training/reports/B6_fine_tune.md
output/training/R000*/summary.json
output/training/R001*/summary.json
output/training/R002*/summary.json
output/training/R003*/summary.json
output/training/R004*/summary.json
output/training/R009*/summary.json
output/training/R010*/summary.json
output/training/R011*/summary.json
output/training/R012*/summary.json
output/training/R013*/summary.json
output/training/R014*/summary.json
output/training/R015*/summary.json
output/training/R016*/summary.json
output/training/R017*/summary.json
output/training/R018*/summary.json
output/training/R019*/summary.json
output/training/R020*/summary.json
output/training/R021*/summary.json
output/training/R022*/summary.json
output/training/R023*/summary.json
output/training/R024*/summary.json
output/training/R025*/summary.json
output/training/R026*/summary.json
output/training/R027*/summary.json
output/training/R028*/summary.json
output/training/R029*/summary.json
output/training/R030*/summary.json
output/training/R031*/summary.json
output/training/R032*/summary.json
output/training/R033*/summary.json
output/training/R034*/summary.json
output/training/R035*/summary.json
output/training/R036*/summary.json
output/training/R037*/summary.json
output/training/R038*/summary.json
output/training/R039*/summary.json
output/training/R040*/summary.json
output/training/R041*/summary.json
output/training/R042*/summary.json
output/training/R043*/summary.json
output/training/R044*/summary.json
output/training/R045*/summary.json
```

The external review tool was invoked for the M5 and M6 updates, but it timed out before returning actionable findings both times. Per `experiment-bridge` fallback, this file records the local checklist review.

## BLOCKING Issues

None for treating `R000-R000d` as passed.

None for treating `R000e-R000h` as passed as DS1 scaffold / M1 data-generation runs.

None for treating `R001-R004b` as passed as M2 data/physics sanity runs.

None for treating `R004c-R004e` as passed as M3 sim-to-real proxy scaffold runs.

None for treating `R005-R008` as passed as M4 baseline scaffold runs.

None for treating `R009-R014` as passed as M5 base hybrid scaffold runs.

None for treating `R015-R033` as passed as M6 component ablation scaffold runs.

None for treating `R034-R036` as passed as M7 cross-vehicle/config reporting scaffold runs. `cross_generalization_claim_supported = 0` is a result, not a code blocker: the selected scaffold improves over physics-only but does not beat the black-box held-out-vehicle reference.

None for treating `R037` as passed as an M8 final single-model checkpoint descriptor. The checkpoint is explicitly a JSON scaffold descriptor, not neural weights.

None for treating `R038-R045` as passed as M9 fine-tune matrix scaffold runs. Individual adapters may improve or degrade; the aggregate report records the result.

The implemented scope is intentionally minimal and covers DS0/DS1/proxy scaffold generation plus validation, baseline gates, a numpy base-hybrid residual scaffold, scaffold-level single-factor ablation plumbing, final-candidate reporting, and ridge-adapter fine-tune matrix plumbing. Formal PyTorch student model training, DS1 full training-scale dataset generation, and GPU deployment are not included in this pass.

## NON-BLOCKING Issues Before DS1

1. The teacher dynamics are still a v0 reduced implementation even though the design target is high fidelity. DS1 should expand tire relaxation, suspension/unsprung dynamics, actuator delay buffers, sensor delay buffers, and richer road geometry before large dataset generation.
2. `R000-R000d` currently regenerate DS0 independently per run. This is acceptable for sanity but DS1 should use dataset manifests as immutable inputs to downstream runs.
3. The validator checks only representative sign gates, not full physical consistency envelopes. DS1 should add range checks for energy, wheel lock behavior, tire saturation statistics, rough-road contact flags, and actuator/sensor delay diagnostics.
4. DS1 should add a stronger environment lock, for example pinned dependency versions and optional CI smoke commands. The current pass records the Miniforge environment contract in `environment.yml` plus minimal Python dependencies in `requirements.txt`.
5. `R000e-R000h` generate a DS1 scaffold: full 700-scenario matrix plus 120 sampled episodes. This is enough for M1 wiring and QA, but it is not yet the full training-scale DS1 dataset described in `DATA_DESIGN.md`.
6. `R004a` is a tiny data-pipeline overfit proxy implemented without PyTorch. It proves short supervised transition records can be memorized, but it is not evidence that the final hybrid student architecture trains correctly.
7. `R004b` is a finite-rollout smoke check for a simple nominal physics predictor. It checks non-divergence and output plumbing, not final physics-only baseline accuracy.
8. `R004c-R004e` perturb only hidden teacher parameters and metadata. This is correct for leakage prevention, but later B6 fine-tune experiments must still verify that adapters can recover from these shifts.
9. The proxy target windows are scaffold-scale: 3 profiles × 3 target roles × 3 scenario groups × 3 samples. This is sufficient for wiring and QA, not for final fine-tune data-efficiency claims.
10. `R006-R008` use numpy ridge-regression black-box scaffold variants with MLP/GRU-style/TCN-style/N-BEATSx-style feature maps. They validate data loading, split handling, metric computation, and fairness plumbing, but should be replaced by trainable PyTorch GRU/TCN/N-BEATSx implementations before making final model-quality claims.
11. DS1 scaffold episodes are only 4 s long, so baseline report uses 25/50/100 step horizons rather than final 1s/5s/10s tables.
12. `R009-R014` use a numpy ridge residual-dot scaffold for `Base = E1 + T1 + F1 + S1 + M1a + V1 + U0`. It validates the experiment interface and evaluation gates, but it is not the final module implementation described in `MODULE_DESIGN.md`.
13. The current base-hybrid scaffold represents `T1/F1/S1/M1a/V1/U0` through student-visible engineered proxy features and bounded residual-dot correction. It does not yet instantiate separate trainable heads for tire force residual, Fz residual, MuHead, steering residual, or heteroscedastic uncertainty.
14. Strong ridge regularization (`ridge_lambda = 10000.0`) is required for the scaffold to avoid OOD residual overcorrection. This is a useful warning for the later PyTorch implementation: residual capacity and uncertainty/gating must be controlled before making generalization claims.
15. `R014` seed replication uses bootstrap resampling around a deterministic ridge solver. This is enough for plumbing, but final seed replication should use independent neural training seeds.
16. `R015-R033` ablations are proxy feature/bound/target-mode changes inside the numpy scaffold. They validate run organization, single-factor switching, metrics, and reports; they should not be cited as final evidence for selecting actual neural modules.
17. `F2` and `M2-oracle` intentionally use teacher-only information for simulator-only auxiliary/oracle comparison. They are flagged in `output/training/reports/B4_ablations.md` and excluded from deployable family-winner interpretation.
18. `T1` and `T1-no-proj` have identical RMSE in the scaffold because projection is represented as a violation proxy rather than a full force projection layer. The final PyTorch force path must implement true friction ellipse projection before making the projection claim.
19. `U1` currently reports bootstrap/seed uncertainty proxy metrics, not final NLL/coverage/sharpness calibration. Full uncertainty evidence still requires explicit predictive distribution evaluation.
20. `R034-R036` separate scaffold completion from claim support. The selected final single scaffold beats physics-only on held-out vehicle/config, but does not beat the black-box held-out-vehicle baseline. The final PyTorch experiments must treat this as an open risk, not as a supported claim.
21. `R037` writes `final_single_model_scaffold.json` as a checkpoint descriptor only. It contains selected variant metadata and metrics, not trained neural weights.
22. `R038-R045` implement fine-tune as ridge adapter corrections over the frozen scaffold-selected hybrid. This validates FT matrix wiring and target-window evaluation, not final adapter learning dynamics.
23. `FT6` performs poorly in the scaffold, which is a useful overcorrection signal but not a final statement about full-model fine-tuning.
24. The B6 proxy fine-tune dataset uses `configs/datasets/ds1_proxy_ft_v1.yaml` with FTD1-FTD5 coverage. This is separate from the earlier M3 proxy dataset, so M3 historical counts remain unchanged.

## Checklist

| Check | Verdict |
|---|---|
| Required observable fields exported | pass |
| Required `fixed_vehicle_context` exported | pass |
| Required `nominal_physics_prior` exported | pass |
| Required teacher auxiliary labels exported | pass |
| Teacher-only labels excluded from `model_input` | pass |
| Metadata excluded from `model_input` | pass |
| Timestamp monotonicity | pass |
| NaN/Inf validation | pass |
| Wheel order and sign convention checks | pass |
| Braking load-transfer sign | pass |
| Left/right steering yaw sign | pass |
| Split-μ braking yaw sign | pass |
| Run directory contract outputs | pass |
| Parseable `metrics.jsonl` and `summary.json` | pass |
| DS1 full 700-scenario matrix emitted | pass |
| DS1 sampled episodes cover CG-SINGLE / CG-SPLIT / CG-TRANSITION | pass |
| DS1 vehicle/config randomization present | pass |
| DS1 split manifest includes train/validation/test/held-out/fine-tune/test-window | pass |
| DS1 FTD1-FTD5 target buckets present | pass |
| DS1 duplicate episode ID check | pass |
| R001 schema / role metadata sanity | pass |
| R002 teacher physical consistency sanity | pass |
| R003 timestamp / dt alignment sanity | pass |
| R004 derived slip / steering / wheel dynamics sanity | pass |
| R004a tiny learnability overfit proxy | pass |
| R004b physics-only rollout smoke | pass |
| R004c proxy perturbation profiles within 5%-15% | pass |
| R004d proxy target windows have no train/val/test overlap | pass |
| R004e proxy distribution shift is measurable and bounded | pass |
| Proxy perturbation metadata remains outside model input | pass |
| R005 physics-only baseline rollout metrics written against ground truth | pass |
| R006 black-box variants train/evaluate with student-visible fields only | pass |
| R007 fairness audit checks split, horizon, parameter count, and input visibility | pass |
| R008 baseline markdown report written | pass |
| R009 base hybrid residual-dot scaffold trains on observable transitions only | pass |
| R010 seen-config evaluation uses held-out evaluation episodes, not training loss | pass |
| R011 held-out road evaluation reports a separate road-transition group | pass |
| R012 held-out vehicle evaluation reports split_role held-out episodes | pass |
| R013 residual magnitude, smoothness, and constraint audit written | pass |
| R014 3-seed bootstrap replication metrics written | pass |
| Base hybrid report compares against physics-only and black-box reference metrics | pass |
| R015-R033 ablation configs use the same DS1 split and horizon protocol | pass |
| Ablation variants change one family at a time relative to the base scaffold | pass |
| F2 and M2-oracle are marked as teacher-oracle/auxiliary scaffold variants | pass |
| R015-R033 metrics compare rollouts against dataset ground truth | pass |
| B4 aggregate markdown and JSON reports are generated from run artifacts | pass |
| R034-R036 cross-generalization reports held-out vehicle/config separately | pass |
| R034-R036 record strict `cross_generalization_claim_supported` separately from run completion | pass |
| R037 final single-model scaffold checkpoint descriptor is written | pass |
| R038-R045 cover FT0 plus FT1-FT6 over FTD1-FTD5 cells | pass |
| R038-R045 evaluate against target test windows, not target train windows | pass |
| B5 and B6 aggregate markdown and JSON reports are generated from run artifacts | pass |

## P3 PyTorch Smoke Addendum

**Date**: 2026-05-21 19:03 CST
**Scope**: R100-R111 PyTorch smoke/training scaffold
**Mode**: local-only fallback; secondary delegated review not used because the current task did not explicitly request sub-agent delegation.

Reviewed files:

```text
experiments/torch_training.py
experiments/run.py
configs/experiments/p3_smoke/p3_1_pytorch_data_loader_smoke.yaml
configs/experiments/p3_smoke/p3_2_pytorch_forward_loss_smoke.yaml
configs/experiments/p3_smoke/p3_3_pytorch_tiny_overfit.yaml
configs/experiments/p3_smoke/p3_4_pytorch_rollout_smoke.yaml
configs/experiments/p3_smoke/p3_5_pytorch_checkpoint_smoke.yaml
configs/experiments/p4_gpu_smoke/p4_1_pytorch_gpu_forward_smoke.yaml
configs/experiments/p4_gpu_smoke/p4_2_pytorch_gpu_tiny_overfit.yaml
configs/experiments/p5_training_dev/p5_1_pytorch_one_step_training.yaml
configs/experiments/p5_training_dev/p5_2_pytorch_rollout_eval.yaml
configs/experiments/p5_training_dev/p5_3_pytorch_resume_eval_smoke.yaml
configs/experiments/p5_training_dev/p5_4_pytorch_black_box_tcn_baseline.yaml
configs/experiments/p6_model_dev/p6_1_pytorch_base_model_small_training.yaml
tests/test_torch_training.py
refine-logs/PYTORCH_IMPLEMENTATION_STATUS.md
```

Checklist:

| Check | Verdict |
|---|---|
| R100-R104 use canonical `data/ds1_v1` through `dataset_source: existing` | pass |
| R105 uses canonical `data/ds1_v1` and requires CUDA instead of silently falling back to CPU | pass |
| R106 uses canonical `data/ds1_v1` and requires CUDA for backward + optimizer tiny-overfit | pass |
| R107-R111 use canonical `data/ds1_v1` and require CUDA | pass |
| PyTorch import is lazy and runner remains import-safe without torch installed | pass |
| Missing PyTorch produces explicit `blocked` summary instead of fake success | pass |
| Missing CUDA produces explicit `blocked` summary for CUDA-required runs | pass |
| Smoke reports are written to `artifacts/torch_training_report.json` | pass |
| Metrics are appended to `metrics.jsonl` and primary metric gates are wired in `run.py` | pass |
| Data loader / forward / tiny overfit / rollout / checkpoint modes have separate run configs | pass |
| Transition-level dataset samples multiple windows per episode for R107+ | pass |
| R107 writes training history and checkpoint artifacts | pass |
| R108 evaluates rollout predictions against dataset ground truth from checkpoint | pass |
| R109 loads checkpoint, resumes optimizer/model state, and writes a resumed checkpoint | pass |
| R110 implements a direct TCN PyTorch black-box baseline path | pass |
| Direct TCN checkpoint restore uses the saved `model_config` and is covered by a hidden-dim regression test | pass |
| Evaluation compares model output against dataset ground truth, not another model output | pass for R101-R103 smoke design |
| Current local environment can complete passing R100-R111 | pass, including R105-R111 on CUDA |
| `sanitize_state()` avoids in-place mutation on tensors requiring gradients | pass |
| R102 tiny-overfit compares loss before/after on the same fixed tiny batch | pass |
| R106 confirms CUDA backward and AdamW optimizer path with loss ratio below gate | pass |
| `AGENTS.md` documents the local conda environment, GPU status, and smoke commands for `/run-experiment` | pass |

Residual risk:

```text
R100-R111 pass. These validate environment readiness, runner plumbing, checkpoint/resume, rollout evaluation, and a first PyTorch black-box baseline. They are still small-run development gates, not final model-quality evidence.
```

## P4 PyTorch Development Addendum

**Date**: 2026-05-21 19:33 CST
**Scope**: R112-R115 plus generated PyTorch matrix configs
**Mode**: local-only fallback; secondary delegated review not used because the current task did not explicitly request sub-agent delegation.

Reviewed files:

```text
student_model/data.py
student_model/torch_model.py
experiments/torch_training.py
experiments/torch_config_matrix.py
experiments/torch_dev_report.py
experiments/run.py
configs/experiments/p6_model_dev/p6_2_pytorch_fair_small_comparison.yaml
configs/experiments/p6_model_dev/p6_3_pytorch_model_variant_smoke.yaml
configs/experiments/p7_adapter_ensemble/p7_1_pytorch_fine_tune_adapter_smoke.yaml
configs/experiments/p7_adapter_ensemble/p7_2_pytorch_deep_ensemble_smoke.yaml
configs/experiments/matrix/MANIFEST.json
tests/test_student_model.py
tests/test_torch_training.py
output/training/reports/PYTORCH_DEV_REPORT.md
```

Checklist:

| Check | Verdict |
|---|---|
| E1 GRU, E2 TCN, and E3 causal Transformer encoders instantiate and forward on canonical DS1 batches | pass |
| T0/T1/T1-no-proj/T2/T3-MoE tire residual switches, F0/F1, S0/S1, M0/M1a, and V0/V1 variants have CUDA forward coverage | pass via R113 |
| Teacher aux labels are exposed by the transition dataset without entering `observable_history` | pass |
| Normalized loss and per-channel RMSE are emitted in one-step validation metrics | pass |
| Rollout eval emits per-channel RMSE and constraint violation rate against dataset ground truth | pass |
| R112 trains hybrid/direct TCN/direct GRU/direct N-BEATS with matched small budget and evaluates rollout for each | pass |
| R114 verifies FT0 has zero trainable parameters and FT1-FT6 expose the intended trainable modules | pass |
| R115 trains K=3 ensemble members and reports predictive variance | pass |
| Generated `configs/experiments/ablation`, `configs/experiments/finetune`, and `configs/experiments/matrix/MANIFEST.json` contain trainable ablation and fine-tune run configs | pass |
| Unknown generated run ids now use available torch pass metrics instead of falling back to schema-only success | pass |

Residual risk:

```text
R112-R115 are development-scale gates. R112 shows the current hybrid still trails the best direct black-box on raw one-step MSE and rollout RMSE. This is an implementation and diagnosis checkpoint, not evidence that the hybrid method is superior.
```

## P5 Experiment Engineering Addendum

**Date**: 2026-05-21 20:01 CST
**Scope**: training governance, local queue, matrix report, and real-data adapter
**Mode**: local-only fallback; secondary delegated review not used because the current task did not explicitly request sub-agent delegation.

Reviewed files:

```text
experiments/torch_training.py
experiments/experiment_queue.py
experiments/matrix_report.py
experiments/real_data_adapter.py
tests/test_experiment_engineering.py
output/training/reports/PYTORCH_MATRIX_REPORT.md
output/training/queue_state_smoke.json
output/training/R111_pytorch_base_model_small_training/artifacts/post_rollout_eval/summary.json
```

Checklist:

| Check | Verdict |
|---|---|
| One-step training saves a best checkpoint in addition to the final checkpoint | pass |
| Early stopping uses validation loss with configurable patience and min-delta | pass |
| LR scheduler supports `none`, `cosine`, and `step` without changing default behavior | pass |
| Nonfinite loss is detected and prevents a false success | pass |
| Final checkpoint step uses completed steps rather than requested max steps | pass |
| Rollout eval can filter by fine-tune buckets, target window role, scenario groups, and vehicle config IDs | pass |
| Queue supports config lists and matrix manifests | pass |
| Queue supports dry-run, skip-success, retry, stop-on-failure, state JSON, and per-attempt logs | pass |
| Queue skip-success path can run missing rollout evaluation without retraining an already successful run | pass |
| Matrix report reads the generated manifest and reports 17 ablation + 35 fine-tune configs | pass |
| Real-data adapter writes canonical manifest, episode arrays, sidecars, and validates output | pass |
| Added tests cover queue/report/adapter control paths | pass |

Residual risk:

```text
The queue is intentionally local and sequential. It is adequate for the current single-GPU workstation and matrix execution control, but it is not a distributed scheduler. Full R200-R216 and R300-R334 training still needs to be launched before model-quality claims can be made.
```

## P6 DS2 / MoE Development Addendum

**Date**: 2026-05-21 20:43 CST
**Scope**: DS2 extreme data scaffold and optional T3-MoE tire residual forward path
**Mode**: local-only fallback; secondary delegated review not used because the current task did not explicitly request sub-agent delegation.

Reviewed files:

```text
teacher_simulator/scenario.py
teacher_simulator/generate.py
student_model/torch_model.py
configs/datasets/ds2_extreme_v0.yaml
configs/experiments/b7_extreme_moe/b7_1_ds2_extreme_dataset_smoke.yaml
configs/experiments/b7_extreme_moe/b7_2_pytorch_ds2_moe_tire_smoke.yaml
configs/experiments/p6_model_dev/p6_3_pytorch_model_variant_smoke.yaml
tests/test_teacher_simulator.py
tests/test_student_model.py
output/training/reports/B7_extreme_moe.md
```

Checklist:

| Check | Verdict |
|---|---|
| DS2 extreme scenario matrix includes emergency braking, trail braking, power-on exit, fishhook, emergency lane change, and sine-sweep steering | pass |
| DS2 generation covers single, split, and transition road profiles | pass via R046 |
| R046 writes canonical teacher dataset artifacts and passes schema validation | pass |
| T3-MoE is behind an explicit `tire_mode: T3` / B7 config gate | pass |
| T3-MoE exposes `tire_moe_weights` for diagnostics | pass |
| FT4 trainability includes T1/T2 and T3-MoE tire residual parameters | pass |
| R113 CUDA variant smoke covers T3-MoE without changing the base model | pass |
| R047 CUDA forward smoke covers T1/T2/T3-MoE on DS2 | pass |

Residual risk:

```text
R046/R047 are development smokes only. They make DS2 and T3-MoE executable, but they do not establish MoE value. The actual B7 claim still requires full DS2 training/evaluation against T1/T2 and a DS1 normal-regime regression check.
```

## Verification Commands

```bash
conda run -n deep-sim python -m compileall teacher_simulator experiments tests
conda run -n deep-sim python -m unittest tests.test_teacher_simulator
for cfg in configs/experiments/b0_data_generation/b0_1_teacher_simulator_minimal.yaml configs/experiments/b0_data_generation/b0_2_tire_load_validation.yaml configs/experiments/b0_data_generation/b0_3_road_scenario_generation.yaml configs/experiments/b0_data_generation/b0_4_sensor_actuator_realism.yaml configs/experiments/b0_data_generation/b0_5_dataset_export_split.yaml; do conda run -n deep-sim python -m experiments.run --config "$cfg" || exit 1; done
for cfg in configs/experiments/b0_data_generation/b0_6_scenario_matrix_v1.yaml configs/experiments/b0_data_generation/b0_7_vehicle_parameter_randomization.yaml configs/experiments/b0_data_generation/b0_8_dataset_split_generation.yaml configs/experiments/b0_data_generation/b0_9_dataset_qa.yaml; do conda run -n deep-sim python -m experiments.run --config "$cfg" || exit 1; done
for cfg in configs/experiments/b1_sanity/b1_1_schema_field_role_check.yaml configs/experiments/b1_sanity/b1_2_teacher_physical_consistency.yaml configs/experiments/b1_sanity/b1_3_time_dt_alignment.yaml configs/experiments/b1_sanity/b1_4_derived_physical_quantities.yaml configs/experiments/b1_sanity/b1_5_tiny_learnability.yaml configs/experiments/b1_sanity/b1_6_physics_rollout_smoke.yaml; do conda run -n deep-sim python -m experiments.run --config "$cfg" || exit 1; done
for cfg in configs/experiments/b2_proxy/b2_1_proxy_perturbation_profiles.yaml configs/experiments/b2_proxy/b2_2_proxy_target_windows.yaml configs/experiments/b2_proxy/b2_3_proxy_distribution_sanity.yaml; do conda run -n deep-sim python -m experiments.run --config "$cfg" || exit 1; done
for cfg in configs/experiments/b3_baselines_base/b3_1_physics_only_baseline.yaml configs/experiments/b3_baselines_base/b3_2_black_box_baseline.yaml configs/experiments/b3_baselines_base/b3_3_baseline_fairness_audit.yaml configs/experiments/b3_baselines_base/b3_4_baseline_rollout_report.yaml; do conda run -n deep-sim python -m experiments.run --config "$cfg" || exit 1; done
for cfg in configs/experiments/b3_baselines_base/b3_5_base_hybrid_training.yaml configs/experiments/b3_baselines_base/b3_6_base_seen_config_eval.yaml configs/experiments/b3_baselines_base/b3_7_base_held_out_road_eval.yaml configs/experiments/b3_baselines_base/b3_8_base_held_out_vehicle_eval.yaml configs/experiments/b3_baselines_base/b3_9_base_residual_constraint_audit.yaml configs/experiments/b3_baselines_base/b3_10_base_seed_replication.yaml; do conda run -n deep-sim python -m experiments.run --config "$cfg" || exit 1; done
for cfg in configs/experiments/b4_ablation_scaffold/b4_1_ablation_tire_t0.yaml configs/experiments/b4_ablation_scaffold/b4_1_ablation_tire_t1.yaml configs/experiments/b4_ablation_scaffold/b4_1_ablation_tire_t1_no_proj.yaml configs/experiments/b4_ablation_scaffold/b4_1_ablation_tire_t2.yaml configs/experiments/b4_ablation_scaffold/b4_2_ablation_fz_f0.yaml configs/experiments/b4_ablation_scaffold/b4_2_ablation_fz_f1.yaml configs/experiments/b4_ablation_scaffold/b4_2_ablation_fz_f2.yaml configs/experiments/b4_ablation_scaffold/b4_3_ablation_steering_s0.yaml configs/experiments/b4_ablation_scaffold/b4_3_ablation_steering_s1.yaml configs/experiments/b4_ablation_scaffold/b4_4_ablation_mu_m0_fixed.yaml configs/experiments/b4_ablation_scaffold/b4_4_ablation_mu_m1a.yaml configs/experiments/b4_ablation_scaffold/b4_4_ablation_mu_m1b.yaml configs/experiments/b4_ablation_scaffold/b4_4_ablation_mu_m2_oracle.yaml configs/experiments/b4_ablation_scaffold/b4_5_ablation_encoder_e1.yaml configs/experiments/b4_ablation_scaffold/b4_5_ablation_encoder_e2.yaml configs/experiments/b4_ablation_scaffold/b4_5_ablation_encoder_e3.yaml configs/experiments/b4_ablation_scaffold/b4_7_ablation_vehicle_v0.yaml configs/experiments/b4_ablation_scaffold/b4_7_ablation_vehicle_v1.yaml configs/experiments/b4_ablation_scaffold/b4_7_ablation_vehicle_v1_large.yaml configs/experiments/b4_ablation_scaffold/b4_7_ablation_vehicle_v2_small.yaml configs/experiments/b4_ablation_scaffold/b4_6_ablation_uncertainty_u0.yaml configs/experiments/b4_ablation_scaffold/b4_6_ablation_uncertainty_u1.yaml; do conda run -n deep-sim python -m experiments.run --config "$cfg" || exit 1; done
conda run -n deep-sim python -m experiments.ablation_report
for cfg in configs/experiments/b5_generalization/b5_1_cross_generalization_base.yaml configs/experiments/b5_generalization/b5_2_cross_generalization_selected_single.yaml configs/experiments/b5_generalization/b5_3_cross_generalization_selected_ensemble.yaml configs/experiments/b5_generalization/b5_4_final_single_model_freeze.yaml configs/experiments/b6_finetune_scaffold/b6_1_finetune_ft0.yaml configs/experiments/b6_finetune_scaffold/b6_2_finetune_ft1_vehicle_param_adapter.yaml configs/experiments/b6_finetune_scaffold/b6_3_finetune_ft2_mu_head.yaml configs/experiments/b6_finetune_scaffold/b6_4_finetune_ft3_fz_residual.yaml configs/experiments/b6_finetune_scaffold/b6_5_finetune_ft4_tire_residual.yaml configs/experiments/b6_finetune_scaffold/b6_6_finetune_ft5_steering_residual.yaml configs/experiments/b6_finetune_scaffold/b6_7_finetune_ft6_full_model.yaml configs/experiments/b6_finetune_scaffold/b6_8_finetune_summary.yaml; do conda run -n deep-sim python -m experiments.run --config "$cfg" || exit 1; done
conda run -n deep-sim python -m experiments.cross_generalization_report
conda run -n deep-sim python -m compileall experiments student_model tests
conda run -n deep-sim python -m unittest tests.test_experiment_engineering tests.test_torch_training
conda run -n deep-sim python -m experiments.experiment_queue --configs configs/experiments/p6_model_dev/p6_1_pytorch_base_model_small_training.yaml --limit 1 --max-retries 0 --skip-success --rollout-eval --rollout-steps 8 --rollout-max-episodes 2 --state-path output/training/queue_state_smoke.json --log-dir output/training/_queue_logs_smoke
conda run -n deep-sim python -m experiments.matrix_report
conda run -n deep-sim python -m experiments.run --config configs/experiments/b7_extreme_moe/b7_1_ds2_extreme_dataset_smoke.yaml
conda run -n deep-sim python -m experiments.run --config configs/experiments/b7_extreme_moe/b7_2_pytorch_ds2_moe_tire_smoke.yaml
git diff --check
```
