---
type: idea
node_id: idea:004
stage: recommended
outcome: proposed
title: "Deep Ensemble Uncertainty Wrapper"
based_on: ["paper:lakshminarayanan2017_deep_ensembles", "paper:torres2023_dkmgp_vehicle_dynamics", "paper:trfc_review_research_perspectives"]
target_gaps: ["gap:G1", "gap:G6"]
added: 2026-05-18T11:54:01Z
---

# Deep Ensemble Uncertainty Wrapper

## Hypothesis

K=3/5 ensemble + heteroscedastic head 能提供最实用的置信度输出，尤其用于 OOD、低附着和低激励不可辨识场景。

## Proposed Method

多个同结构 hybrid model 独立训练；aleatoric 用 predicted variance，epistemic 用 model mean variance。

## Expected Outcome

提高 NLL、coverage 和 error-uncertainty correlation。

## Pilot

待执行：single model vs K=3 ensemble。

## Status

RECOMMENDED AFTER SINGLE MODEL STABLE。

