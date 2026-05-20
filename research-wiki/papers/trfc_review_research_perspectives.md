---
type: paper
node_id: paper:trfc_review_research_perspectives
title: "Tire-Road Friction Coefficient Estimation: Review and Research Perspectives"
authors: []
year: 2022
venue: "Review"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["friction-estimation", "trfc", "low-friction", "identifiability"]
added: 2026-05-18T11:33:46Z
---

# Tire-Road Friction Coefficient Estimation: Review and Research Perspectives

## One-line thesis

TRFC estimation is fundamentally limited under low excitation, so μ should be represented with uncertainty rather than forced into a single deterministic value.

## Problem / Gap

Vehicle-only signals may not identify true road friction during normal steady driving.

## Method

Survey of dynamics-based, tire-based, vision-based, and fusion-based TRFC estimation.

## Key Results

This project should output wheel-level `μ_mean, μ_logvar` and increase uncertainty when excitation is weak.

## Assumptions

Friction becomes more observable when slip, braking, steering, or lateral excitation is sufficient.

## Limitations / Failure Modes

Without vision, early recognition of upcoming road surface is limited.

## Reusable Ingredients

Friction latent, identifiability analysis, sensor fusion roadmap.

## Open Questions

How to detect low-excitation windows robustly from available signals.

## Claims

待建立。

## Connections

待由 graph 自动生成。

## Relevance to This Project

Directly supports uncertainty-aware road latent design.

