---
type: paper
node_id: paper:kabzan_learning_based_vehicle_dynamics
title: "Learning-Based Vehicle Dynamics Modeling for Autonomous Racing"
authors: ["Juraj Kabzan", "Lukas Hewing", "Alexander Liniger", "Melanie N. Zeilinger"]
year: 2019
venue: "IEEE Robotics and Automation Letters"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["vehicle-dynamics", "gaussian-process", "residual-learning", "uncertainty"]
added: 2026-05-18T11:33:46Z
---

# Learning-Based Vehicle Dynamics Modeling for Autonomous Racing

## One-line thesis

Nominal vehicle dynamics can be improved by learning residual modeling errors, with uncertainty useful for downstream control.

## Problem / Gap

Pure physical models miss tire and actuator effects; pure black-box models need more data and lose structure.

## Method

Learning-based residual dynamics for autonomous racing.

## Key Results

For this project, use physical backbone as mean structure and NN/ensemble as scalable residual and uncertainty proxy.

## Assumptions

Racing setting has strong excitation; normal driving has weaker identifiability.

## Limitations / Failure Modes

Does not by itself solve wet/snow/Split-μ or road transition.

## Reusable Ingredients

Residual dynamics; uncertainty-aware model correction.

## Open Questions

Whether GP-style uncertainty is worth the engineering cost versus deep ensembles.

## Claims

待建立。

## Connections

待由 graph 自动生成。

## Relevance to This Project

Supports `VehicleResidualNN` as small residual, not full dynamics replacement.

