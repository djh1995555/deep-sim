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

TCN/GRU/MLP 直接预测状态可能在训练分布内有低 one-step error。N-BEATSx 作为 direct multi-horizon forecasting baseline，可能在固定预测窗口内给出强 black-box trajectory reference。

## Rejection Reason

不满足物理约束、Split-μ、显式 `Fz_i`、可解释性和部署稳定性要求。只能作为 baseline，不能作为主模型。

## Baseline Variants

```text
BB-GRU:
  observable / control history -> state increment or autoregressive rollout

BB-TCN:
  causal observable / control history -> state increment or autoregressive rollout

BB-MLP:
  flattened short history -> state increment

BB-NBEATSx:
  observable / control history + exogenous context
  -> direct multi-horizon state or state-increment trajectory
```

`BB-NBEATSx` 不使用显式车辆/轮胎物理结构，不输出 `Fz_i / μ_i / tire force` 等物理中间量。

## Status

ELIMINATED AS MAINLINE / KEEP AS BASELINE。
