---
type: idea
node_id: idea:006
stage: eliminated
outcome: rejected
title: "End-to-End Black-Box Dynamics as Production Model"
based_on: []
target_gaps: ["gap:G2", "gap:G6", "gap:G7"]
added: 2026-05-18T11:54:01Z
---

# End-to-End Black-Box Dynamics as Production Model

## Hypothesis

TCN/GRU 直接预测状态可能在训练分布内有低 one-step error。

## Rejection Reason

不满足物理约束、Split-μ、显式 `Fz_i`、可解释性和部署稳定性要求。只能作为 baseline，不能作为主模型。

## Status

ELIMINATED AS MAINLINE / KEEP AS BASELINE。

