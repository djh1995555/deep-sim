---
type: paper
node_id: paper:neural_tire_force_drifting
title: "Neural Network Tire Force Modeling for Automated Drifting"
authors: []
year: 2024
venue: "Autonomous Driving / Robotics"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["tire-force", "neural-network", "drifting", "nonlinear-tire"]
added: 2026-05-18T11:33:46Z
---

# Neural Network Tire Force Modeling for Automated Drifting

## One-line thesis

Large slip and drifting regimes require nonlinear tire force modeling, but learned tire forces need physical saturation constraints.

## Problem / Gap

Linear tire models fail at high slip angles and saturation.

## Method

Neural tire force modeling in extreme handling.

## Key Results

Use `TireResidualNN` for nonlinear regimes, then project force to friction ellipse.

## Assumptions

Training data must cover high-slip regimes.

## Limitations / Failure Modes

Pure NN force outputs can violate force limits and extrapolate poorly.

## Reusable Ingredients

Slip-region gating; tire force residual.

## Open Questions

Whether to use force residual or parameter residual for first implementation.

## Claims

待建立。

## Connections

待由 graph 自动生成。

## Relevance to This Project

Supports tire residual as main learned correction.

