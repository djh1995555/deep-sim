# Experiment Code Review

**Date**: 2026-05-21 12:41:55 CST
**Scope**: `R000-R000d` sanity-first implementation
**Mode**: local-only fallback

## Review Boundary

Reviewed files:

```text
teacher_simulator/
experiments/run.py
configs/teacher/ds0_minimal.yaml
configs/runs/R000*.yaml
tests/test_teacher_simulator.py
reports/B0_teacher.md
runs/R000*/summary.json
```

The external review tool was invoked, but it returned tool-call text rather than actionable findings. Per `experiment-bridge` fallback, this file records the local checklist review.

## BLOCKING Issues

None for treating `R000-R000d` as passed.

The implemented scope is intentionally minimal and only covers DS0 smoke generation plus validation gates. Student model training, DS1 full scenario generation, and GPU deployment are not included in this pass.

## NON-BLOCKING Issues Before DS1

1. The teacher dynamics are still a v0 reduced implementation even though the design target is high fidelity. DS1 should expand tire relaxation, suspension/unsprung dynamics, actuator delay buffers, sensor delay buffers, and richer road geometry before large dataset generation.
2. `R000-R000d` currently regenerate DS0 independently per run. This is acceptable for sanity but DS1 should use dataset manifests as immutable inputs to downstream runs.
3. The validator checks only representative sign gates, not full physical consistency envelopes. DS1 should add range checks for energy, wheel lock behavior, tire saturation statistics, rough-road contact flags, and actuator/sensor delay diagnostics.
4. DS1 should add a stronger environment lock, for example pinned dependency versions and optional CI smoke commands. The current pass records the Miniforge environment contract in `environment.yml` plus minimal Python dependencies in `requirements.txt`.

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

## Verification Commands

```bash
conda run -n deep-sim python -m compileall teacher_simulator experiments tests
conda run -n deep-sim python -m unittest tests.test_teacher_simulator
for cfg in configs/runs/R000.yaml configs/runs/R000a.yaml configs/runs/R000b.yaml configs/runs/R000c.yaml configs/runs/R000d.yaml; do conda run -n deep-sim python -m experiments.run --config "$cfg" || exit 1; done
git diff --check
```
