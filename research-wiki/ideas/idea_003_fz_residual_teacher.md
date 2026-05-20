---
type: idea
node_id: idea:003
stage: recommended
outcome: proposed
title: "Bounded FzResidualNN with Simulator Teacher Supervision"
based_on: ["paper:gillespie_fundamentals_vehicle_dynamics", "paper:tire_normal_force_estimation_ann", "paper:wheel_ground_contact_normal_forces"]
target_gaps: ["gap:G5", "gap:G7"]
added: 2026-05-18T11:54:01Z
---

# Bounded FzResidualNN with Simulator Teacher Supervision

## Hypothesis

显式载荷转移物理 + 小幅 `ΔFz_i` residual 能补悬架和高自由度仿真器差异，改善低附着和 Split-μ 下轮胎力上限估计。

## Proposed Method

`Fz_i = LoadTransferPhysics + bounded ΔFz_i`，然后做 positivity 与 total-load projection；仿真器 `Fz_true_i` 仅用于 teacher loss。

## Expected Outcome

降低 roll/pitch/yaw response 误差，并提高 tire force saturation 判断准确性。

## Pilot

待执行：有/无 `FzResidualNN`，有/无 `Fz_true` teacher loss。

## Status

RECOMMENDED MODULE / ABLATION REQUIRED。

