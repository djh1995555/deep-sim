---
type: idea
node_id: idea:002
stage: recommended_ablation
outcome: proposed
title: "Parameter-Residual Tire Model"
based_on: ["paper:pacejka_tire_vehicle_dynamics", "paper:neural_tire_force_drifting"]
target_gaps: ["gap:G4", "gap:G6"]
added: 2026-05-18T11:54:01Z
---

# Parameter-Residual Tire Model

## Hypothesis

输出 `ΔC_alpha/ΔC_kappa/Δμ_scale` 比直接输出 force residual 更稳、更可解释，但极限区表达能力可能不足。

## Proposed Method

让 NN 修正 tire parameters，再由物理 tire model 计算 `Fx/Fy`。

## Expected Outcome

常规和中低 slip 区更稳；低附着极限区可能弱于 force residual。

## Pilot

待执行：与 idea:001 相同 backbone 对比。

## Status

GROUP 3 / RECOMMENDED ABLATION。
