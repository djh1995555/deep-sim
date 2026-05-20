---
type: paper
node_id: paper:deep_dynamics_pinn_vehicle_dynamics
title: "Deep Dynamics: Vehicle Dynamics Modeling with a Physics-Informed Neural Network for Autonomous Racing"
authors: ["Michael A. Hessel", "Vikram Krishnan", "John K. Subosits", "J. Christian Gerdes"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2402.14660"
  doi: null
  s2: null
tags: ["vehicle-dynamics", "physics-informed", "autonomous-racing", "neural-network"]
added: 2026-05-18T11:41:23Z
---

# Deep Dynamics: Vehicle Dynamics Modeling with a Physics-Informed Neural Network for Autonomous Racing

## One-line thesis

Physics-informed neural dynamics can improve autonomous racing prediction by embedding vehicle dynamics structure instead of relying on a fully black-box model.

## Problem / Gap

High-performance vehicle dynamics are nonlinear and hard to model with fixed analytical parameters.

## Method

Physics-informed neural network for vehicle dynamics modeling.

## Key Results

Supports using a physics backbone with learned correction for high-dynamics regimes.

## Assumptions

Autonomous racing provides high-excitation data; road friction variation is not the only focus.

## Limitations / Failure Modes

Does not directly solve Split-μ, road transition, or low-excitation friction identifiability.

## Reusable Ingredients

Physics-informed loss; structured dynamics learning.

## Open Questions

Whether residual should act at force, acceleration, or state increment level.

## Claims

待建立。

## Connections

待由 graph 自动生成。

## Relevance to This Project

Core support for hybrid physics + neural residual design.

