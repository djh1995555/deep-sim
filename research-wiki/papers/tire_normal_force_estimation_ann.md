---
type: paper
node_id: paper:tire_normal_force_estimation_ann
title: "Tire Normal Force Estimation Using Artificial Neural Networks and Fuzzy Logic"
authors: []
year: 2023
venue: "Vehicle Dynamics"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["normal-load", "fz", "neural-network", "load-transfer"]
added: 2026-05-18T11:33:46Z
---

# Tire Normal Force Estimation Using Artificial Neural Networks and Fuzzy Logic

## One-line thesis

Data-driven normal-load estimation is useful as a residual, but direct unconstrained `Fz_i` outputs are risky.

## Problem / Gap

Analytical load transfer misses suspension nonlinearities and transient effects.

## Method

ANN/fuzzy methods for tire normal force estimation.

## Key Results

Use `FzResidualNN`, then enforce positivity and total-load consistency.

## Assumptions

Roll, pitch, acceleration, and suspension-related signals are informative.

## Limitations / Failure Modes

Direct learned normal forces may violate conservation without projection.

## Reusable Ingredients

Bounded `ΔFz_i`; teacher supervision from high-fidelity simulator.

## Open Questions

Whether true simulator `Fz_i` is reliable enough as teacher.

## Claims

待建立。

## Connections

待由 graph 自动生成。

## Relevance to This Project

Supports explicit `Fz_i` plus residual correction.

