# Experiment Code Review

**Date**: 2026-05-21 15:41:42 CST
**Scope**: `R000-R033` teacher/data generation, sanity, proxy, baseline, base hybrid, and component ablation scaffold implementation
**Mode**: local-only fallback

## Review Boundary

Reviewed files:

```text
teacher_simulator/
experiments/run.py
experiments/sanity.py
experiments/baselines.py
experiments/hybrid.py
experiments/ablation_report.py
configs/teacher/ds0_minimal.yaml
configs/runs/R000*.yaml
configs/runs/R001.yaml
configs/runs/R002.yaml
configs/runs/R003.yaml
configs/runs/R004.yaml
configs/runs/R004a.yaml
configs/runs/R004b.yaml
configs/runs/R004c.yaml
configs/runs/R004d.yaml
configs/runs/R004e.yaml
configs/runs/R005.yaml
configs/runs/R006.yaml
configs/runs/R007.yaml
configs/runs/R008.yaml
configs/runs/R009.yaml
configs/runs/R010.yaml
configs/runs/R011.yaml
configs/runs/R012.yaml
configs/runs/R013.yaml
configs/runs/R014.yaml
configs/runs/R015.yaml
configs/runs/R016.yaml
configs/runs/R017.yaml
configs/runs/R018.yaml
configs/runs/R019.yaml
configs/runs/R020.yaml
configs/runs/R021.yaml
configs/runs/R022.yaml
configs/runs/R023.yaml
configs/runs/R024.yaml
configs/runs/R025.yaml
configs/runs/R026.yaml
configs/runs/R027.yaml
configs/runs/R027a.yaml
configs/runs/R027b.yaml
configs/runs/R027c.yaml
configs/runs/R028.yaml
configs/runs/R029.yaml
configs/runs/R030.yaml
configs/runs/R031.yaml
configs/runs/R032.yaml
configs/runs/R033.yaml
configs/teacher/ds1_proxy_v1.yaml
tests/test_teacher_simulator.py
reports/B0_teacher.md
reports/B3_baselines.md
reports/B3_base_hybrid.md
reports/B4_ablations.md
runs/R000*/summary.json
runs/R001*/summary.json
runs/R002*/summary.json
runs/R003*/summary.json
runs/R004*/summary.json
runs/R009*/summary.json
runs/R010*/summary.json
runs/R011*/summary.json
runs/R012*/summary.json
runs/R013*/summary.json
runs/R014*/summary.json
runs/R015*/summary.json
runs/R016*/summary.json
runs/R017*/summary.json
runs/R018*/summary.json
runs/R019*/summary.json
runs/R020*/summary.json
runs/R021*/summary.json
runs/R022*/summary.json
runs/R023*/summary.json
runs/R024*/summary.json
runs/R025*/summary.json
runs/R026*/summary.json
runs/R027*/summary.json
runs/R028*/summary.json
runs/R029*/summary.json
runs/R030*/summary.json
runs/R031*/summary.json
runs/R032*/summary.json
runs/R033*/summary.json
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

The implemented scope is intentionally minimal and covers DS0/DS1/proxy scaffold generation plus validation, baseline gates, a numpy base-hybrid residual-dot scaffold, and scaffold-level single-factor ablation plumbing. Formal PyTorch student model training, DS1 full training-scale dataset generation, and GPU deployment are not included in this pass.

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
17. `F2` and `M2-oracle` intentionally use teacher-only information for simulator-only auxiliary/oracle comparison. They are flagged in `reports/B4_ablations.md` and excluded from deployable family-winner interpretation.
18. `T1` and `T1-no-proj` have identical RMSE in the scaffold because projection is represented as a violation proxy rather than a full force projection layer. The final PyTorch force path must implement true friction ellipse projection before making the projection claim.
19. `U1` currently reports bootstrap/seed uncertainty proxy metrics, not final NLL/coverage/sharpness calibration. Full uncertainty evidence still requires explicit predictive distribution evaluation.

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

## Verification Commands

```bash
conda run -n deep-sim python -m compileall teacher_simulator experiments tests
conda run -n deep-sim python -m unittest tests.test_teacher_simulator
for cfg in configs/runs/R000.yaml configs/runs/R000a.yaml configs/runs/R000b.yaml configs/runs/R000c.yaml configs/runs/R000d.yaml; do conda run -n deep-sim python -m experiments.run --config "$cfg" || exit 1; done
for cfg in configs/runs/R000e.yaml configs/runs/R000f.yaml configs/runs/R000g.yaml configs/runs/R000h.yaml; do conda run -n deep-sim python -m experiments.run --config "$cfg" || exit 1; done
for cfg in configs/runs/R001.yaml configs/runs/R002.yaml configs/runs/R003.yaml configs/runs/R004.yaml configs/runs/R004a.yaml configs/runs/R004b.yaml; do conda run -n deep-sim python -m experiments.run --config "$cfg" || exit 1; done
for cfg in configs/runs/R004c.yaml configs/runs/R004d.yaml configs/runs/R004e.yaml; do conda run -n deep-sim python -m experiments.run --config "$cfg" || exit 1; done
for cfg in configs/runs/R005.yaml configs/runs/R006.yaml configs/runs/R007.yaml configs/runs/R008.yaml; do conda run -n deep-sim python -m experiments.run --config "$cfg" || exit 1; done
for cfg in configs/runs/R009.yaml configs/runs/R010.yaml configs/runs/R011.yaml configs/runs/R012.yaml configs/runs/R013.yaml configs/runs/R014.yaml; do conda run -n deep-sim python -m experiments.run --config "$cfg" || exit 1; done
for cfg in configs/runs/R015.yaml configs/runs/R016.yaml configs/runs/R017.yaml configs/runs/R018.yaml configs/runs/R019.yaml configs/runs/R020.yaml configs/runs/R021.yaml configs/runs/R022.yaml configs/runs/R023.yaml configs/runs/R024.yaml configs/runs/R025.yaml configs/runs/R026.yaml configs/runs/R027.yaml configs/runs/R027a.yaml configs/runs/R027b.yaml configs/runs/R027c.yaml configs/runs/R028.yaml configs/runs/R029.yaml configs/runs/R030.yaml configs/runs/R031.yaml configs/runs/R032.yaml configs/runs/R033.yaml; do conda run -n deep-sim python -m experiments.run --config "$cfg" || exit 1; done
conda run -n deep-sim python -m experiments.ablation_report
git diff --check
```
