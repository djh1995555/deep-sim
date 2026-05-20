---
type: idea
node_id: idea:001
stage: recommended
outcome: proposed
title: "Three-Group Hybrid Dynamics Experiment: Global Residual vs Force Tire Residual vs Parameter Tire Residual"
based_on: ["paper:pacejka_tire_vehicle_dynamics", "paper:deep_dynamics_pinn_vehicle_dynamics", "paper:trfc_review_research_perspectives"]
target_gaps: ["gap:G1", "gap:G2", "gap:G4", "gap:G6", "gap:G7"]
added: 2026-05-18T11:54:01Z
---

# Three-Group Hybrid Dynamics Experiment

## Hypothesis

先用物理轮胎模型 + 单一 global residual 建立最小闭环，再分别对比 force-level tire residual 和 parameter-level tire residual，可以明确轮胎误差是否需要从整车 residual 中拆出来。

## Proposed Method

三组主实验：

```text
Shared: MuHead outputs wheel-level μ_mean/logvar
Group 1: TirePhysics only + GlobalResidualNN
Group 2: TirePhysics + ΔFx/ΔFy TireResidualNN + friction ellipse projection + VehicleResidualNN
Group 3: TirePhysics with ΔC_alpha/ΔC_kappa/Δμ_scale parameter residual + VehicleResidualNN
```

默认主结构更新为：

```text
Steering: S2 = first-order steering lag
Steering ablation: S3 = S2 + bounded NN steering residual
Residual placement: V1 = bounded Δx_dot before integration
```

其中 `VehicleResidualNN` 对照：

```text
V0: no residual
V1: x_next = Integrator(x_t, x_dot_phys + Δx_dot)
V2: x_next = Integrator(x_t, x_dot_phys) + Δx
V3: both Δx_dot + Δx
```

`SteeringHead` 对照：

```text
S0: fixed steering ratio
S1: learnable ratio + bias
S2: first-order steering lag (Rank 1 default)
S3: S2 + bounded NN steering residual (ablation)
```

## Expected Outcome

Group 1 建立最小可用闭环；Group 2 若在 wet/snow/Split-μ 中显著降低 `vy/r/omega_i` rollout error，则说明需要显式 tire residual；Group 3 若接近 Group 2 且更平滑，则优先考虑 parameter residual。

## Pilot

待执行：P0 physics-only/black-box；P1 Group 1 V0/V1/V2；P2 Group 1 vs Group 2；P3 Group 2 vs Group 3；P4 FzResidualNN；P5 S2 vs S3；P6 ensemble。

## Status

RECOMMENDED EXPERIMENT STRUCTURE。
