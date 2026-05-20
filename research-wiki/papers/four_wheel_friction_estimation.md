---
type: paper
node_id: paper:four_wheel_friction_estimation
title: "Road Friction Estimation for Four-Wheel Independent Drive Electric Vehicles"
authors: []
year: 2020
venue: "Vehicle Dynamics / Control"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["friction-estimation", "four-wheel", "4wid", "split-mu"]
added: 2026-05-18T11:33:46Z
---

# Road Friction Estimation for Four-Wheel Independent Drive Electric Vehicles

## One-line thesis

Wheel-level friction estimation is structurally better aligned with Split-μ than a single global road friction state.

## Problem / Gap

Global μ cannot describe four wheels on different surfaces.

## Method

Use wheel torque, wheel speed, slip, and vehicle response for wheel-level friction inference.

## Key Results

This project should model `μ_fl, μ_fr, μ_rl, μ_rr` separately.

## Assumptions

Wheel torque and wheel speed are available, as in the project input list.

## Limitations / Failure Modes

Low excitation still causes weak identifiability.

## Reusable Ingredients

Wheel-level latent friction heads.

## Open Questions

How strongly to regularize left/right or front/rear μ similarity on uniform roads.

## Claims

待建立。

## Connections

待由 graph 自动生成。

## Relevance to This Project

Direct support for Split-μ architecture.

