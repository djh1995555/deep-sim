# P5-P20 精读综合：低附着、轮胎/Fz residual、Hybrid Learning 与不确定性

**日期**：2026-05-18  
**范围**：完成 `DEEP_READING_PACK` 中 P5-P20 的精读综合，并将未验证或题名不稳定的条目降级为“方向证据”，不作为强引用。  
**用途**：冻结 Phase 2 方案生成前的工程设计约束。

## 1. 总结结论

结合 P1-P4 已冻结的物理 backbone，剩余文献支持以下工程路线：

```text
history encoder
  -> wheel-level μ distribution / road latent
  -> steering effective dynamics
  -> bounded ΔFz residual
  -> tire residual / tire parameter residual
  -> uncertainty

physical backbone
  -> double-track body dynamics
  -> roll/pitch dynamics
  -> explicit Fz_i
  -> tire physics + friction ellipse
  -> small vehicle residual
```

最重要的设计结论：

1. `μ_i` 不应作为确定单点输出，而应输出分布或均值+方差；
2. Split-μ 需要 wheel-level latent，至少四轮 `μ_fl/fr/rl/rr`；
3. 低激励时 μ 不可辨识，uncertainty 应主动增大；
4. `TireResidualNN` 优先输出受 friction ellipse projection 约束的 `ΔFx/ΔFy`；
5. `FzResidualNN` 只补悬架/仿真器差异，不能破坏 `ΣFz≈mg`；
6. `VehicleResidualNN` 应作用在 acceleration/state increment 上，但强限幅；
7. 训练必须以 multi-step rollout 为核心，one-step loss 只作辅助；
8. 第一版不确定性用 deep ensemble + heteroscedastic Gaussian head，比复杂 BNN/GP 更工程可控。

## 2. P5-P8：Hybrid / learned vehicle dynamics

### P5. Deep Dynamics / Physics-informed vehicle dynamics for autonomous racing

**读到的工程点**：

- physics-informed 学习模型通常不是纯端到端状态映射，而是在 loss、结构或状态更新里嵌入车辆动力学；
- 自动赛车文献更关注极限动态和模型误差，但常假设路面相对固定；
- 对本项目有价值的是“让 NN 学 residual 或未知参数”，而不是让 NN 替代整车动力学。

**对方案的冻结**：

```text
x_next = integrate(f_vehicle_physics(...)) + bounded_vehicle_residual
```

`VehicleResidualNN` 建议输出：

```text
Δa_x, Δa_y, Δr_dot, Δp_dot, Δq_dot
```

而不是直接输出完整 `x_next`。这样更容易做限幅和物理解释。

### P6. Kabzan et al., Learning-Based Vehicle Dynamics Modeling for Autonomous Racing

**读到的工程点**：

- GP / learned residual 适合补偿 nominal dynamics 与真实车辆之间的偏差；
- GP 自带 uncertainty，但在大数据和高维输入时工程成本高；
- 自动赛车模型通常会把学习模块放在动力学误差而非完整动力学上。

**对方案的冻结**：

第一版不采用 GP 作为主模型，但保留思想：

```text
physics backbone = mean structure
NN residual = scalable residual
ensemble variance = GP-like epistemic proxy
```

### P7. Deep Kernel Learning / GP variants for vehicle dynamics identification

**读到的工程点**：

- DKL/GP 类模型的主要价值是 uncertainty 和小数据建模；
- 缺点是实现复杂、训练/推理成本高、和长 rollout 结合不如普通 NN 直接；
- 可作为后续 uncertainty baseline，不作为第一版主干。

**对方案的冻结**：

第一版 uncertainty：

```text
K-model deep ensemble + heteroscedastic σ head
```

DKL/GP 放到对照实验或后续版本。

### P8. Long-horizon / multi-step dynamics learning

**读到的工程点**：

- one-step error 不能代表 rollout 质量；
- multi-step training、scheduled sampling、closed-loop rollout loss 对稳定性关键；
- vehicle dynamics 的误差会通过 `vx, vy, r, omega_i` 积累，并进一步污染 slip 和 tire force。

**对方案的冻结**：

训练 loss：

```text
L = L_1step
  + λ_rollout * L_multi_step
  + λ_phys * L_physics_violation
  + λ_res * L_residual_regularization
  + λ_unc * L_nll_or_calibration
```

rollout horizon 建议：

```text
warmup history: 1-3 s
training rollout: 1-5 s progressive
evaluation rollout: 5-20 s, 按场景分组
```

## 3. P9-P12：低附着、TRFC、Split-μ、路面衔接

### P9. Tire-Road Friction Coefficient Estimation: Review and Research Perspectives

**读到的工程点**：

- TRFC 估计方法主要分为基于车辆动力学、基于轮胎/轮速、基于视觉/路面感知、融合方法；
- 低激励下 μ 的可辨识性弱；
- 动力学方法通常需要足够的纵向/横向激励才能可靠估计 μ；
- 视觉能提前感知路面，但第一阶段可先不接入。

**对方案的冻结**：

`NN_mu` 不输出“真实 μ 单点”，而输出：

```text
μ_mean_i, μ_logvar_i, road_latent_i
```

低激励时：

```text
σ_mu_i ↑
confidence ↓
```

### P10. Four-wheel / 4WID road friction estimation

**读到的工程点**：

- 四轮独立驱动/制动车辆天然适合 wheel-level friction estimation；
- 轮速、驱动/制动扭矩、纵向滑移是低附着估计的关键；
- wheel-level 估计比 single global μ 更适合 Split-μ。

**对方案的冻结**：

```text
μ_i = NN_mu_i(history, omega_i, tau_drv_i, tau_brk_i, slip_i, local Fz_i)
```

禁止只用：

```text
μ_global
```

作为正式模型。

### P11. Split-μ identification and vehicle stability

**读到的工程点**：

- 左右附着差会通过制动力/驱动力差和侧向力差产生明显 yaw moment；
- Split-μ 识别必须结合左右轮速度、制动/驱动输入和 yaw response；
- 控制文献中 Split-μ 常用于稳定性控制，但对本项目则用于动力学仿真准确性。

**对方案的冻结**：

Split-μ 评估必须包含：

```text
left dry / right ice
front dry / rear snow
diagonal μ mismatch
transition from dry to wet/snow
```

核心指标：

```text
yaw rate r rollout error
vy rollout error
wheel speed omega_i error
μ_i uncertainty calibration
```

### P12. Spatio-temporal / time-delay friction estimation

**读到的工程点**：

- 路面变化不能只用单帧分类，需要时间窗口；
- TDNN/RNN/history encoder 对 wheel-level μ 估计有价值；
- 路面衔接处 latent 不应无约束跳变。

**对方案的冻结**：

`μ_i(t)` 的 latent dynamics：

```text
h_t = Encoder(x_{t-K:t}, u_{t-K:t})
μ_i(t), σ_i(t) = Head_mu(h_t)
L_mu_smooth = ||μ_i(t) - μ_i(t-1)||, transition 场景放宽
```

如果仿真器有 true road μ，可作为 teacher loss；真实车部署时不需要 true μ 输入。

## 4. P13-P17：轮胎力、法向载荷、Fz residual

### P13. Neural-network tire force modeling for drifting / extreme handling

**读到的工程点**：

- 大侧偏、大滑移区域的 tire force 很难由线性模型覆盖；
- 学习模型在极限区有价值，但必须覆盖足够数据；
- 纯 NN tire force 容易违反力上限，工程上应加 physics projection。

**对方案的冻结**：

`TireResidualNN` 输入：

```text
κ_i, α_i, Fz_i, μ_i_mean, μ_i_sigma, vx, wheel_id, road_latent_i
```

输出优先：

```text
ΔFx_i, ΔFy_i
```

然后做：

```text
[Fx_i, Fy_i] = project_to_friction_ellipse(F_phys + ΔF)
```

### P14. Auxiliary-enhanced tire lateral force estimation

**读到的工程点**：

- tire force estimation 不应只看一个 slip angle；
- auxiliary variables，如 load、speed、yaw response、slip ratio、road condition，能显著改善估计；
- 对本项目，`Fz_i`、`μ_i` 和历史编码器输出都应作为 tire residual 输入。

**对方案的冻结**：

不要设计：

```text
ΔFy_i = MLP(alpha_i)
```

应设计：

```text
ΔF_i = MLP(kappa_i, alpha_i, Fz_i, μ_i, vx, h_t, wheel_id)
```

### P15. Bayesian / uncertainty-aware intelligent tire force estimation

**读到的工程点**：

- tire force 本身应带 uncertainty，尤其在高 slip、低附着、未覆盖工况；
- BNN 理论上可用，但工程复杂度高；
- ensemble 和 heteroscedastic loss 更适合第一版。

**对方案的冻结**：

输出：

```text
Fx_i_mean, Fy_i_mean
Fx_i_logvar, Fy_i_logvar
```

或者总状态层输出：

```text
x_next_mean, x_next_logvar
```

第一版优先状态层 uncertainty，第二版再细化到 tire force uncertainty。

### P16-P17. Tire normal force estimation / wheel-ground contact normal force

**读到的工程点**：

- 直接学习 `Fz_i` 可行，但需要强约束，否则会违反总载荷守恒；
- suspension dynamics、roll/pitch、加速度是估计 `Fz_i` 的关键输入；
- 实验验证通常依赖真实车难以部署的传感器，因此本项目应利用高保真仿真器内部 `Fz_i` 作为辅助监督。

**对方案的冻结**：

`FzResidualNN` 训练：

```text
Fz_phys_i = load_transfer_physics(...)
ΔFz_i = bounded_NN(...)
Fz_i = mass_consistent_projection(Fz_phys_i + ΔFz_i)
```

如果仿真器输出 true `Fz_i`：

```text
L_Fz_teacher = ||Fz_i - Fz_true_i||
```

但部署时：

```text
Fz_true_i 不作为输入
```

## 5. P18-P20：物理约束序列模型、Neural ODE、Deep Ensembles

### P18. Physics-constrained RNN / sequence dynamics

**读到的工程点**：

- 物理约束可以放在结构、loss 或 projection 中；
- 对本项目最实用的是 projection + violation penalty；
- RNN/TCN/Transformer 只负责 history encoder，不直接接管物理状态更新。

**对方案的冻结**：

History encoder 推荐：

```text
TCN or GRU first
Transformer later if data量足够
```

因为第一版工程落地更看重稳定、可调试和低延迟训练。

### P19. Neural ODE

**读到的工程点**：

- Neural ODE 适合连续时间和不规则采样；
- 当前项目如果仿真步长固定，显式离散积分更简单；
- 物理 backbone 本身已经是 ODE，可微分积分可作为实现细节。

**对方案的冻结**：

第一版：

```text
fixed-step differentiable integrator
RK4 or semi-implicit Euler
```

不把 Neural ODE 作为主卖点。

### P20. Deep Ensembles

**读到的工程点**：

- Deep ensembles 是工程上强且简单的不确定性基线；
- 可以同时估计 aleatoric 与 epistemic；
- calibration 可用 NLL、ECE、coverage、error-vs-uncertainty correlation。

**对方案的冻结**：

第一版 uncertainty：

```text
K = 3 or 5 models
each outputs mean + logvar
epistemic = variance of means
aleatoric = mean predicted variance
total uncertainty = epistemic + aleatoric
```

## 6. 最终冻结的模型结构

```text
Inputs:
  x_{t-K:t}, u_{t-K:t}

History Encoder:
  h_t = Encoder(history)

Latent Heads:
  μ_i_mean, μ_i_logvar = MuHead(h_t, wheel features)
  delta_eff = SteeringHead(h_t, sw_angle, steer_cmd)
  ΔFz_i = FzResidualHead(h_t, ax, ay, roll, pitch, p, q)

Physics:
  Fz_i = LoadTransferPhysics(x_t, params) + projected(ΔFz_i)
  Fx_phys_i, Fy_phys_i = TirePhysics(κ_i, α_i, Fz_i, μ_i)

Tire Residual:
  ΔFx_i, ΔFy_i = TireResidualHead(κ_i, α_i, Fz_i, μ_i, h_t)
  Fx_i, Fy_i = friction_ellipse_projection(F_phys_i + ΔF_i)

Vehicle Dynamics:
  x_dot_phys = DoubleTrackRollPitchDynamics(x_t, Fx_i, Fy_i, delta_eff)
  Δx_vehicle = bounded VehicleResidualHead(h_t, x_t, u_t)
  x_next = Integrator(x_t, x_dot_phys) + Δx_vehicle

Uncertainty:
  x_next_logvar = UncertaintyHead(...)
  ensemble for epistemic
```

## 7. 最终冻结的训练目标

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

教师监督只允许用于 loss：

```text
Fz_true, Fx_true, Fy_true, μ_true
```

不能作为部署输入。

## 8. 下一步进入 Phase 2 的建议

可以进入 idea generation，但这里的“idea”应理解为工程方案变体，而不是论文创新。建议生成 4-6 个架构候选：

1. 参数 residual 型 tire model；
2. force residual 型 tire model；
3. mixture-of-experts tire residual；
4. latent μ distribution + conservative tire force；
5. no vehicle residual / small vehicle residual 对照；
6. Fz teacher supervision vs no Fz teacher supervision。

