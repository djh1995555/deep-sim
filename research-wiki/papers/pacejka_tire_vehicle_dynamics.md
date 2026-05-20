---
type: paper
node_id: paper:pacejka_tire_vehicle_dynamics
title: "Tire and Vehicle Dynamics"
authors: ["Hans B. Pacejka"]
year: 2012
venue: "Book"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["vehicle-dynamics", "tire-model", "pacejka", "friction-ellipse"]
added: 2026-05-18T11:28:12Z
---

# Tire and Vehicle Dynamics

## One-line thesis

车辆动力学模型必须通过 `κ, α, Fz, μ` 显式约束轮胎力，不能让轮胎力完全由黑盒网络自由输出。

## Problem / Gap

本项目需要在低附着、Split-μ 和轮胎非线性区保持 rollout 稳定，核心瓶颈是轮胎力上限和载荷敏感性。

## Method

采用显式 tire physics：slip ratio、slip angle、normal load、friction coefficient 和 tire parameters 共同决定 `Fx, Fy`。

## Key Results

对本项目的工程结论是：轮胎力必须满足摩擦椭圆/饱和约束，且 `Fz_i` 是 tire force 的核心输入。

## Assumptions

轮胎可由低维 slip/load/friction 参数化；完整 Pacejka 不一定作为可训练 backbone，可作为高保真仿真器或 teacher。

## Limitations / Failure Modes

完整 Magic Formula 参数多、辨识成本高。第一版可采用 Fiala/Dugoff/simplified brush 作为 backbone，再由 residual 补偿。

## Reusable Ingredients

- friction ellipse
- load sensitivity
- combined slip
- tire parameter residual

## Open Questions

`TireResidualNN` 应输出 `ΔFx/ΔFy`，还是输出 `ΔC_alpha/ΔC_kappa/Δμ_scale`。

## Claims

待建立。

## Connections

待由 graph 自动生成。

## Relevance to This Project

决定 tire physics 和 tire residual 的边界。

