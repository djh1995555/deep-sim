---
type: paper
node_id: paper:multistep_gp_vehicle_dynamics
title: "Multi-Step Prediction of Vehicle Dynamics Using Gaussian Process Regression"
authors: []
year: 2020
venue: "Vehicle Dynamics / Machine Learning"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["vehicle-dynamics", "multi-step", "gaussian-process", "rollout"]
added: 2026-05-18T11:41:23Z
---

# Multi-Step Prediction of Vehicle Dynamics Using Gaussian Process Regression

## One-line thesis

Vehicle dynamics models must be evaluated in multi-step rollout, not only one-step prediction.

## Problem / Gap

One-step errors compound through slip, tire force, and state update chains.

## Method

Multi-step vehicle dynamics prediction using Gaussian process regression.

## Key Results

Rollout loss and rollout metrics should be primary training and evaluation targets.

## Assumptions

The prediction horizon is long enough for error accumulation to matter.

## Limitations / Failure Modes

GP methods may not scale to all planned simulator data.

## Reusable Ingredients

Multi-step rollout evaluation.

## Open Questions

Training horizon schedule: 1s to 5s progressive, evaluation 5s to 20s.

## Claims

待建立。

## Connections

待由 graph 自动生成。

## Relevance to This Project

Directly supports multi-step rollout loss as primary objective.

