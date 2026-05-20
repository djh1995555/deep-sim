---
type: paper
node_id: paper:chen2018_neural_ode
title: "Neural Ordinary Differential Equations"
authors: ["Ricky T. Q. Chen", "Yulia Rubanova", "Jesse Bettencourt", "David Duvenaud"]
year: 2018
venue: "NeurIPS"
external_ids:
  arxiv: "1806.07366"
  doi: null
  s2: null
tags: ["neural-ode", "dynamical-systems", "continuous-time"]
added: 2026-05-18T11:41:23Z
---

# Neural Ordinary Differential Equations

## One-line thesis

Neural ODEs learn continuous-time dynamics with differentiable ODE solvers.

## Problem / Gap

Dynamical systems may require continuous-time modeling or irregular sampling.

## Method

Parameterize the derivative function with a neural network and solve using ODE integration.

## Key Results

Useful implementation reference, but current project can start with fixed-step differentiable integration around a physical ODE.

## Assumptions

Continuous-time formulation and solver cost are acceptable.

## Limitations / Failure Modes

Can add implementation complexity without solving friction identifiability.

## Reusable Ingredients

Differentiable integration; continuous-time state dynamics.

## Open Questions

Whether variable simulator timestep requires ODE solver later.

## Claims

待建立。

## Connections

待由 graph 自动生成。

## Relevance to This Project

Supports possible future continuous-time implementation; not first-version core.

