# 精读后冻结的最终设计决策

完整内容见：`idea-stage/FINAL_DESIGN_DECISIONS_20260518_193346.md`

## 最终主线

```text
double-track 5-DOF body + 4-wheel rotational dynamics
+ roll/pitch dynamics
+ explicit Fz_i
+ wheel-level μ distribution
+ tire physics + tire residual
+ small vehicle residual
+ deep ensemble uncertainty
+ multi-step rollout training
```

## 第一版不做

- 纯端到端黑盒动力学；
- 单纯 bicycle 正式模型；
- 视觉输入；
- 仿真器内部变量作为部署输入；
- 无约束直接输出 `Fz_i` 或 tire force。

