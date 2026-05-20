# 侧倾/俯仰动力学与四轮载荷转移定向调研

完整内容见：`idea-stage/LOAD_TRANSFER_SURVEY_20260518_190429.md`

## 检查点摘要

建议采用第二版结构：

```text
roll/pitch/load-transfer physics
  -> Fz_fl, Fz_fr, Fz_rl, Fz_rr
  -> tire physics + tire residual
  -> vehicle dynamics + small vehicle residual
```

关键设计：

- 使用 double-track vehicle model 支持 Split-μ；
- 显式建模 `roll, pitch, p, q`；
- `Fz_i` 作为物理中间变量进入轮胎模型；
- NN 不直接替代全部 `Fz_i`，只输出受约束的 `ΔFz_i`；
- 约束 `Fz_i >= 0`、`sum_i Fz_i ≈ m*g`、摩擦椭圆；
- tire residual 是主 residual，vehicle residual 只做小幅末端修正。

## 推荐进入 Phase 2 的固定前提

1. 正式 backbone 用 double-track，不用单纯 bicycle；
2. roll/pitch dynamics 放进车辆物理层；
3. `Fz_i` 显式计算；
4. `FzResidualNN` 做小幅补偿；
5. 内部仿真变量只作为辅助监督，不作为部署输入。

