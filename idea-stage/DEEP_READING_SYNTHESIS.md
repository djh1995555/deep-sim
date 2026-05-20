# P5-P20 精读综合

完整内容见：`idea-stage/DEEP_READING_SYNTHESIS_20260518_193346.md`

## 已冻结最终结构

```text
history encoder
  -> wheel-level μ distribution
  -> delta_eff
  -> bounded ΔFz_i
  -> tire residual
  -> uncertainty

physical backbone
  -> double-track body dynamics
  -> roll/pitch dynamics
  -> explicit Fz_i
  -> tire physics + friction ellipse
  -> small vehicle residual
```

## 关键决策

- `μ_i` 输出均值+方差，不输出确定单点；
- Split-μ 使用四轮级 `μ_fl/fr/rl/rr`；
- 低激励时 uncertainty 主动增大；
- `TireResidualNN` 输出 `ΔFx/ΔFy`，再投影到 friction ellipse；
- `FzResidualNN` 做小幅、质量守恒的 `ΔFz_i`；
- `VehicleResidualNN` 只做小幅状态/加速度修正；
- 训练以 multi-step rollout loss 为主；
- 第一版 uncertainty 用 deep ensemble + heteroscedastic head。

