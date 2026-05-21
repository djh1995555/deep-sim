# Experiment Code Review

**Date**: 2026-05-21 14:56:35 CST
**Scope**: `R000-R008` teacher/data generation, sanity, proxy, and baseline scaffold implementation
**Mode**: local-only fallback

## Review Boundary

Reviewed files:

```text
teacher_simulator/
experiments/run.py
experiments/sanity.py
experiments/baselines.py
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
configs/teacher/ds1_proxy_v1.yaml
tests/test_teacher_simulator.py
reports/B0_teacher.md
reports/B3_baselines.md
runs/R000*/summary.json
runs/R001*/summary.json
runs/R002*/summary.json
runs/R003*/summary.json
runs/R004*/summary.json
```

The external review tool was invoked, but it returned tool-call text rather than actionable findings. Per `experiment-bridge` fallback, this file records the local checklist review.

## BLOCKING Issues

None for treating `R000-R000d` as passed.

None for treating `R000e-R000h` as passed as DS1 scaffold / M1 data-generation runs.

None for treating `R001-R004b` as passed as M2 data/physics sanity runs.

None for treating `R004c-R004e` as passed as M3 sim-to-real proxy scaffold runs.

None for treating `R005-R008` as passed as M4 baseline scaffold runs.

The implemented scope is intentionally minimal and covers DS0/DS1/proxy scaffold generation plus validation and baseline gates. Formal hybrid student model training, DS1 full training-scale dataset generation, and GPU deployment are not included in this pass.

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

## Verification Commands

```bash
conda run -n deep-sim python -m compileall teacher_simulator experiments tests
conda run -n deep-sim python -m unittest tests.test_teacher_simulator
for cfg in configs/runs/R000.yaml configs/runs/R000a.yaml configs/runs/R000b.yaml configs/runs/R000c.yaml configs/runs/R000d.yaml; do conda run -n deep-sim python -m experiments.run --config "$cfg" || exit 1; done
for cfg in configs/runs/R000e.yaml configs/runs/R000f.yaml configs/runs/R000g.yaml configs/runs/R000h.yaml; do conda run -n deep-sim python -m experiments.run --config "$cfg" || exit 1; done
for cfg in configs/runs/R001.yaml configs/runs/R002.yaml configs/runs/R003.yaml configs/runs/R004.yaml configs/runs/R004a.yaml configs/runs/R004b.yaml; do conda run -n deep-sim python -m experiments.run --config "$cfg" || exit 1; done
for cfg in configs/runs/R004c.yaml configs/runs/R004d.yaml configs/runs/R004e.yaml; do conda run -n deep-sim python -m experiments.run --config "$cfg" || exit 1; done
for cfg in configs/runs/R005.yaml configs/runs/R006.yaml configs/runs/R007.yaml configs/runs/R008.yaml; do conda run -n deep-sim python -m experiments.run --config "$cfg" || exit 1; done
git diff --check
```
