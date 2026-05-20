---
type: paper
node_id: paper:phys_constrained_rnn_dynamics
title: "Physics-Constrained Recurrent Neural Networks for Dynamical Systems"
authors: []
year: 2020
venue: "Dynamical Systems / Machine Learning"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["physics-constrained", "rnn", "dynamical-systems", "rollout"]
added: 2026-05-18T11:41:23Z
---

# Physics-Constrained Recurrent Neural Networks for Dynamical Systems

## One-line thesis

Sequence models become more reliable when physical constraints are imposed through architecture, loss, or projection.

## Problem / Gap

Unconstrained recurrent dynamics can drift during long rollout.

## Method

Recurrent neural dynamics with physical constraints.

## Key Results

For this project, use TCN/GRU as history encoder while physical projection enforces tire/load constraints.

## Assumptions

Constraints can be written in differentiable form.

## Limitations / Failure Modes

Generic physics-constrained RNN does not provide vehicle-specific tire/load equations.

## Reusable Ingredients

Rollout training; constraint penalties; sequence encoders.

## Open Questions

Whether TCN or GRU is more stable for current sensor history.

## Claims

待建立。

## Connections

待由 graph 自动生成。

## Relevance to This Project

Supports using history encoder plus physical constraints rather than monolithic RNN dynamics.

