---
type: idea
node_id: idea:005
stage: backup
outcome: proposed
title: "Slip-Regime Mixture-of-Experts Tire Residual"
based_on: ["paper:neural_tire_force_drifting", "paper:auxiliary_tire_lateral_force_estimation"]
target_gaps: ["gap:G4"]
added: 2026-05-18T11:54:01Z
---

# Slip-Regime Mixture-of-Experts Tire Residual

## Hypothesis

按 slip regime 软门控 tire residual experts，可同时处理小侧偏线性区和大侧偏非线性区。

## Proposed Method

基于 `κ, α, Fz, μ` 的 gating network，选择/混合多个 tire residual heads。

## Expected Outcome

极限操控和大 slip 子集更好。

## Pilot

推迟到极限工况数据加入后。

## Status

BACKUP / SECOND PHASE。

