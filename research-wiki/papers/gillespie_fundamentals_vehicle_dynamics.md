---
type: paper
node_id: paper:gillespie_fundamentals_vehicle_dynamics
title: "Fundamentals of Vehicle Dynamics"
authors: ["Thomas D. Gillespie"]
year: 1992
venue: "Book"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["vehicle-dynamics", "load-transfer", "roll", "pitch", "suspension"]
added: 2026-05-18T11:28:12Z
---

# Fundamentals of Vehicle Dynamics

## One-line thesis

纵向/横向载荷转移和 roll/pitch 动态应显式决定四轮 `Fz_i`，再由 `Fz_i` 限制轮胎力。

## Problem / Gap

低附着和 Split-μ 的关键不是只估计 μ，还要估计每个轮胎实际可用的 `μ_i * Fz_i`。

## Method

使用静态载荷、纵向载荷转移、横向载荷转移和姿态动态修正构造四轮法向载荷。

## Key Results

`Fz_i = Fz_static + ΔFz_long + ΔFz_lat + ΔFz_roll_pitch + ΔFz_residual` 是本项目推荐结构。

## Assumptions

车辆质量、轴距、质心高度、轮距和等效 roll/pitch 刚度阻尼可获得或可辨识。

## Limitations / Failure Modes

真实悬架非线性、防倾杆、轮胎垂向刚度和路面扰动会导致解析模型偏差，需要小 residual。

## Reusable Ingredients

- longitudinal load transfer
- lateral load transfer
- roll/pitch spring-damper dynamics
- mass consistency constraint for normal loads

## Open Questions

是否需要引入 `az`；当前输入没有 `az`，第一版使用 `ΣFz_i ≈ m*g`。

## Claims

待建立。

## Connections

待由 graph 自动生成。

## Relevance to This Project

直接决定 `Fz_i` 计算模块和 `FzResidualNN` 约束。

