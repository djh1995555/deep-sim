---
type: paper
node_id: paper:milliken_race_car_vehicle_dynamics
title: "Race Car Vehicle Dynamics"
authors: ["William F. Milliken", "Douglas L. Milliken"]
year: 1995
venue: "Book"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["vehicle-dynamics", "load-transfer", "roll-stiffness", "handling"]
added: 2026-05-18T11:28:12Z
---

# Race Car Vehicle Dynamics

## One-line thesis

左右载荷转移不应只按几何粗分，front/rear roll stiffness distribution 会显著影响每个轮胎的可用力。

## Problem / Gap

Split-μ 和极限工况下，左右轮载荷差会影响 yaw moment 和饱和顺序。

## Method

通过 roll stiffness distribution、track width、CG/roll center 几何分配 lateral load transfer。

## Key Results

本项目应引入 `λ_f = k_roll_front / (k_roll_front + k_roll_rear)`，用它分配前后轴横向载荷转移。

## Assumptions

前后 roll stiffness 可作为等效参数学习，不需要第一版完整建悬架。

## Limitations / Failure Modes

赛车动力学强调极限操控，工程仿真第一版可以简化，但不能完全忽略 roll stiffness distribution。

## Reusable Ingredients

- lateral load transfer distribution
- front/rear roll stiffness split
- tire load sensitivity in handling

## Open Questions

`k_roll_front/k_roll_rear` 是否能从高自由度仿真数据稳定辨识。

## Claims

待建立。

## Connections

待由 graph 自动生成。

## Relevance to This Project

支持第二版载荷转移方案中加入可学习 roll stiffness distribution。

