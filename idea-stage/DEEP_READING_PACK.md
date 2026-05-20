# 当前方案精读包

完整内容见：`idea-stage/DEEP_READING_PACK_20260518_192434.md`

## 本轮精读目标

围绕当前方案精读 20 篇左右材料：

```text
double-track 5-DOF body + 4-wheel rotational dynamics
+ roll/pitch dynamics
+ explicit Fz_fl/fr/rl/rr
+ wheel-level latent μ
+ tire physics + tire residual
+ small vehicle residual
+ uncertainty-aware rollout
```

## 优先顺序

1. P1-P4：冻结物理 backbone 和 `Fz_i` 公式；
2. P9-P12：冻结 `μ_i`、Split-μ、可辨识性假设；
3. P13-P17：冻结 `TireResidualNN` 和 `FzResidualNN`；
4. P5-P8、P18-P20：冻结学习框架、rollout loss 和 uncertainty。

## 需要冻结的工程决策

- `Fz_i` 是否采用 roll stiffness distribution；
- tire model 用 Pacejka / Dugoff / Fiala / simplified brush；
- tire residual 输出 `ΔFx/ΔFy` 还是 tire parameter residual；
- `μ_i` 是标量、分布，还是类别 + 连续附着；
- residual 作用在 force、acceleration 还是 state；
- uncertainty 用 ensemble、heteroscedastic Gaussian、BNN 或 GP；
- rollout loss horizon 和低附着/Split-μ 场景权重。

