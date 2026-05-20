---
type: paper
node_id: paper:rajamani_vehicle_dynamics_control
title: "Vehicle Dynamics and Control"
authors: ["Rajesh Rajamani"]
year: 2012
venue: "Book"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["vehicle-dynamics", "state-estimation", "control", "friction-estimation"]
added: 2026-05-18T11:28:12Z
---

# Vehicle Dynamics and Control

## One-line thesis

控制导向车辆动力学应以低自由度显式状态方程为 backbone，再用估计器或 residual 处理未知参数和环境。

## Problem / Gap

本项目需要从真实可观测状态构造可部署 dynamics，而不是依赖高自由度仿真器内部变量。

## Method

使用车体坐标动力学、横摆动力学、侧偏/轮胎模型和状态估计框架。

## Key Results

`vx, vy, r` 主方程应作为 vehicle physics 的核心，NN 不应直接替代整车状态转移。

## Assumptions

低自由度模型可以覆盖主动力学，未建模部分由参数估计或 residual 补偿。

## Limitations / Failure Modes

单纯 bicycle model 不足以表达四轮 Split-μ 和左右载荷转移。

## Reusable Ingredients

- body-frame longitudinal/lateral/yaw dynamics
- state observer framing
- tire-road friction estimation framing

## Open Questions

正式模型采用 double-track；bicycle 仅保留为 baseline。

## Claims

待建立。

## Connections

待由 graph 自动生成。

## Relevance to This Project

确定 vehicle physics 主方程和 history encoder 的可观测输入边界。

