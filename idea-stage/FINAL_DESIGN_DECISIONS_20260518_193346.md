# 精读后冻结的最终设计决策

**日期**：2026-05-18  
**输入**：P1-P20 精读综合、`requirement.md`、`research-wiki/query_pack.md`  
**目标**：作为后续 Phase 2 idea generation / experiment plan 的固定工程约束。

## 1. 模型主结构

```text
Observable history:
  x_{t-K:t}, u_{t-K:t}

History encoder:
  h_t = TCN/GRU(history)

Latent heads:
  μ_i_mean, μ_i_logvar = MuHead(h_t, wheel features)
  delta_eff = SteeringHead(h_t, sw_angle, steer_cmd)
  ΔFz_i = FzResidualHead(h_t, ax, ay, roll, pitch, p, q)

Physics:
  Fz_i = LoadTransferPhysics(x_t, params) + projected(ΔFz_i)
  Fx_phys_i, Fy_phys_i = TirePhysics(κ_i, α_i, Fz_i, μ_i)

Tire residual:
  ΔFx_i, ΔFy_i = TireResidualHead(κ_i, α_i, Fz_i, μ_i, h_t)
  Fx_i, Fy_i = friction_ellipse_projection(F_phys_i + ΔF_i)

Vehicle dynamics:
  x_dot = DoubleTrackRollPitchDynamics(x_t, Fx_i, Fy_i, delta_eff)
  x_next = Integrator(x_t, x_dot) + bounded VehicleResidualHead(...)

Uncertainty:
  heteroscedastic logvar head + deep ensemble
```

## 2. 自由度与状态

正式模型：

```text
5-DOF body + 4-wheel rotational dynamics
```

状态：

```text
x = [
  vx, vy,
  yaw, r,
  roll, p,
  pitch, q,
  omega_fl, omega_fr, omega_rl, omega_rr
]
```

输出接口保持 requirement 中顺序：

```text
[vx, vy, roll, pitch, yaw, p, q, r, omega_fl, omega_fr, omega_rl, omega_rr]
```

## 3. 必须显式建模的物理中间量

```text
κ_i      slip ratio
α_i      slip angle
Fz_i     four-wheel normal load
μ_i      wheel-level road friction latent
Fx_i,Fy_i tire forces
delta_eff effective steering angle
```

禁止把这些全部藏进 monolithic NN。

## 4. `Fz_i` 决策

```text
Fz_i =
  Fz_static_i
  + ΔFz_long_i
  + ΔFz_lat_i
  + ΔFz_roll_pitch_i
  + ΔFz_residual_i
```

约束：

```text
Fz_i >= Fz_min
ΣFz_i ≈ m*g
|ΔFz_residual_i| <= 0.05~0.15 * Fz_static_i
```

采用 front/rear roll stiffness distribution：

```text
λ_f = k_roll_front / (k_roll_front + k_roll_rear)
λ_r = 1 - λ_f
```

## 5. `μ_i` 决策

输出四轮级分布：

```text
μ_fl/fr/rl/rr_mean
μ_fl/fr/rl/rr_logvar
```

低激励时不强行确定：

```text
low excitation -> σ_mu ↑ -> confidence ↓
```

如果仿真器有 true μ：

```text
true μ 可用于 teacher loss
true μ 不可作为部署输入
```

## 6. Tire residual 决策

第一版优先：

```text
Fx_i, Fy_i = TirePhysics(...) + ΔFx_i, ΔFy_i
```

然后：

```text
project_to_friction_ellipse(Fx_i, Fy_i, μ_i, Fz_i)
```

备选对照：

```text
NN 输出 ΔC_alpha, ΔC_kappa, Δμ_scale
```

## 7. Vehicle residual 决策

`VehicleResidualNN` 只做小幅末端修正：

```text
Δa_x, Δa_y, Δr_dot, Δp_dot, Δq_dot
```

或小幅：

```text
Δx_next
```

必须正则和限幅，避免退化成黑盒状态预测。

## 8. 不确定性决策

第一版：

```text
K = 3 or 5 deep ensemble
each model outputs mean + logvar

epistemic = Var(model means)
aleatoric = Mean(predicted variances)
total_uncertainty = epistemic + aleatoric
```

校准指标：

```text
NLL
coverage
error-vs-uncertainty correlation
scenario-wise calibration
```

## 9. 训练目标

```text
L_total =
  L_rollout
  + λ_1 * L_one_step
  + λ_2 * L_friction_ellipse
  + λ_3 * L_Fz_positive
  + λ_4 * L_Fz_sum
  + λ_5 * L_residual_magnitude
  + λ_6 * L_mu_smooth
  + λ_7 * L_uncertainty_NLL
  + λ_8 * L_teacher_optional
```

`L_rollout` 是主目标，`L_one_step` 是辅助。

## 10. 数据与评估场景

必须分组评估：

```text
dry steady
wet steady
snow/ice low friction
braking on low μ
turning on low μ
Split-μ left/right
Split-μ front/rear
diagonal μ mismatch
road transition dry->wet/snow
```

核心指标：

```text
rollout error of vx, vy, r
roll/pitch/p/q error
omega_i error
trajectory/yaw drift
constraint violation rate
uncertainty calibration
```

## 11. 第一批架构对照实验

建议 Phase 2 生成并比较：

1. Force residual tire model vs parameter residual tire model；
2. 有/无 `FzResidualNN`；
3. 有/无 `VehicleResidualNN`；
4. deterministic μ vs μ distribution；
5. global μ vs wheel-level μ；
6. one-step training vs rollout training；
7. single model vs deep ensemble。

## 12. 明确不做

第一版不做：

- 纯端到端 `x_next = NN(history)` 主模型；
- 单纯 bicycle 作为正式模型；
- 视觉输入；
- 把仿真器内部变量作为部署输入；
- 无约束直接输出全部 `Fz_i` 或全部 tire force。

