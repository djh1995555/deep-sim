---
type: paper
node_id: paper:auxiliary_tire_lateral_force_estimation
title: "Auxiliary-Enhanced Neural Network for Tire Lateral Force Estimation"
authors: []
year: 2025
venue: "Vehicle Dynamics"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["tire-force", "lateral-force", "neural-network", "auxiliary-features"]
added: 2026-05-18T11:41:23Z
---

# Auxiliary-Enhanced Neural Network for Tire Lateral Force Estimation

## One-line thesis

Tire lateral force estimation benefits from auxiliary physical variables, not just slip angle alone.

## Problem / Gap

Tire force depends on load, road friction, speed, and combined slip, so single-input force networks are weak.

## Method

Neural tire lateral force estimation with auxiliary inputs.

## Key Results

TireResidualNN should use `κ, α, Fz, μ, vx, h_t, wheel_id`.

## Assumptions

Auxiliary physical variables are available or estimated.

## Limitations / Failure Modes

If auxiliary variables are inaccurate, force estimates may inherit their bias.

## Reusable Ingredients

Auxiliary-feature tire residual design.

## Open Questions

Whether tire residual needs wheel-specific parameters or shared network with wheel embedding.

## Claims

待建立。

## Connections

待由 graph 自动生成。

## Relevance to This Project

Directly guides TireResidualNN input design.

