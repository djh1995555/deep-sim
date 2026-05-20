---
type: paper
node_id: paper:lakshminarayanan2017_deep_ensembles
title: "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
authors: ["Balaji Lakshminarayanan", "Alexander Pritzel", "Charles Blundell"]
year: 2017
venue: "NeurIPS"
external_ids:
  arxiv: "1612.01474"
  doi: null
  s2: null
tags: ["uncertainty", "deep-ensembles", "calibration"]
added: 2026-05-18T11:33:46Z
---

# Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles

## One-line thesis

Independent ensembles with probabilistic outputs provide a strong, simple uncertainty baseline.

## Problem / Gap

Vehicle dynamics model needs confidence estimates for OOD, low-friction, and low-identifiability cases.

## Method

Train multiple models with predictive variance heads.

## Key Results

Use ensemble variance for epistemic uncertainty and predicted variance for aleatoric uncertainty.

## Assumptions

Multiple model training is acceptable.

## Limitations / Failure Modes

Computational cost scales with ensemble size.

## Reusable Ingredients

K-model ensemble; NLL loss; calibration metrics.

## Open Questions

Choose K=3 or K=5 for first engineering version.

## Claims

待建立。

## Connections

待由 graph 自动生成。

## Relevance to This Project

Recommended first-version uncertainty approach.

