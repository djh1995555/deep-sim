# 工程方案候选报告

完整内容见：`idea-stage/IDEA_REPORT_20260518_211203.md`

## 三组主实验方案

三组都保留 `MuHead`，输出四轮 `μ_mean/logvar`，并作为轮胎物理模型的路面附着输入。

### Group 1

只有物理轮胎模型，不单独设置 tire residual；只保留一个 global residual，负责覆盖轮胎和整车残差。

### Group 2

加入 tire residual，输出 `ΔFx/ΔFy`，并投影到 friction ellipse。

### Group 3

加入 tire residual，但输出 `ΔC_alpha/ΔC_kappa/Δμ_scale` 等参数修正。

## 默认顺序

```text
physics-only baseline
-> Group 1
-> Group 2
-> Group 3
-> FzResidualNN / S3 / ensemble 等共享 ablation
```

## 当前默认

```text
Steering: S2
Vehicle/global residual placement: V1, Δx_dot before integration
```
