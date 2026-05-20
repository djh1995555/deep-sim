# MODULE_DESIGN

**日期**：2026-05-20  
**状态**：current v2  
**作用**：定义 student model 和各组件的详细设计方案。

本文档是 student 组件设计的权威来源，覆盖模块职责、输入输出、插入位置、约束、diagnostics、默认网络结构、bounded 参数化、超参初值、训练 schedule 和 ablation 落地方式。`EXPERIMENT_PLAN.md` 只引用 `T/F/S/M/V/U/FT` 等实验配置编号，不重复定义具体组件结构。

## 1. 总体原则

第一版模型以可微分物理 backbone 为主，神经网络模块只做受控补偿。

默认约束：

```text
所有 residual head 使用 bounded output
越靠后的 residual 容量越小
所有 teacher-only 字段只用于 loss / diagnostics
所有 ablation 配置从头训练
B6 fine-tune 从同一个 final single model checkpoint 初始化
```

默认时间窗口：

```text
history_len: 0.5-2.0 s
rollout_train_horizon: 1-5 s warm-up, then extend to 10 s
dt: follow dataset sampling rate, resample only in data pipeline
```

默认编码器候选：

```text
Option A: small GRU encoder
Option B: causal TCN encoder
```

第一版优先使用 small GRU encoder，原因是实现简单、适合不同长度 target windows、便于 fine-tune adapter。

## 2. Shared Encoder

实验引用：`Base = T1 + F1 + S1 + M1a + V1 + U0` 以及 `FT1-FT6`。数据字段以 `DATA_DESIGN.md` 为准。

输入：

```text
observable history
fixed_vehicle_context embedding
nominal_physics_prior embedding
```

默认结构：

```text
observable_norm -> GRU(hidden=128, layers=1 or 2)
context_mlp: MLP([context_dim, 64, 64])
prior_mlp: MLP([prior_dim, 64, 64])
fusion: concat([h_obs, h_context, h_prior]) -> MLP([256, 128])
activation: SiLU
normalization: LayerNorm on fused latent
```

输出：

```text
z_shared: 128-dim
```

约束：

```text
encoder 不接收 teacher-only 字段
encoder 在 FT1-FT5 默认冻结
FT6 才允许整体 fine-tune
```

## 3. VehicleParamAdapter

实验引用：`FT1` 和 `FT6`。这里定义目标车/目标时间段慢变量 adapter 的接口、默认结构和参数化方式。

默认结构：

```text
input: z_shared + nominal_physics_prior
adapter_mlp: MLP([128 + prior_dim, 128, 64])
heads:
  Δmass_scale: 1
  ΔIx_scale / ΔIy_scale / ΔIz_scale: 3
  Δcg_x / Δcg_z: 2
  optional z_vehicle: 16
activation: SiLU
```

bounded 参数化：

```text
mass_eff = mass_nominal * exp(clip(Δmass_scale, -b_m, b_m))
I_eff = I_nominal * exp(clip(ΔI_scale, -b_I, b_I))
cg_eff = cg_nominal + b_cg * tanh(raw_cg)
```

初始 bound 建议：

```text
b_m: log(1.3)
b_I: log(1.5)
b_cg_x: 0.2-0.5 m
b_cg_z: 0.1-0.3 m
```

训练：

```text
base training: trainable
FT1: only VehicleParamAdapter trainable
regularization: parameter delta magnitude + temporal smoothness
```

与 `VehicleResidualNN` 的边界：

```text
VehicleParamAdapter:
  episode/window-level slow variables
  modifies mass_eff / inertia_eff / cg_eff / optional z_vehicle
  explains persistent target-vehicle/time-window shifts

VehicleResidualNN:
  timestep-level small fast residual
  modifies selected x_dot channels after physics aggregation
  explains remaining high-frequency or unmodeled dynamics
```

实现判据：如果 FT1 已经使 `VehicleResidualNN` magnitude 明显下降，说明慢变量 adapter 捕获了主要目标车差异；如果 FT1 无效但 V1/FT6 有效，说明当前 adapter 输出空间不足。

## 4. Steering Module

实验引用：`S0/S1`、`FT5`。这里定义 steering module 的接口、默认结构和约束。

### 4.1 S0: First-Order Steering Lag

形式：

```text
delta_cmd = sw_angle / steering_ratio_nominal
dot(delta_eff) = (delta_cmd - delta_eff) / tau_steer
```

默认：

```text
tau_steer: learnable scalar or small table by vehicle type
delta_eff clamp: physical steering limit
```

### 4.2 S1: S0 + steerResidualNN

默认结构：

```text
input: z_shared + delta_cmd + delta_eff_S0 + vx + r
mlp: MLP([input_dim, 64, 32])
head: Δdelta_eff_FL, Δdelta_eff_FR
activation: SiLU
output: bound_delta * tanh(raw)
```

初始 bound：

```text
bound_delta: 1-3 deg in radians
```

训练：

```text
base: trainable
FT5: only steerResidualNN / SteeringHead trainable
regularization: Δdelta magnitude + temporal smoothness
```

## 5. FzResidualNN

实验引用：`F0/F1/F2`、`FT3`。这里定义 `FzResidualNN` 的接口、默认结构、投影和 teacher auxiliary loss。

默认结构：

```text
input:
  z_shared
  Fz_physics_i
  vx, vy, r, p, q
  steering/brake/drive summary
mlp: MLP([input_dim, 128, 64])
head: ΔFz_FL, ΔFz_FR, ΔFz_RL, ΔFz_RR
activation: SiLU
output: bound_Fz * tanh(raw)
```

初始 bound：

```text
bound_Fz_i: 0.05-0.10 * mass_eff * g per wheel
```

投影/约束：

```text
Fz_raw_i = Fz_physics_i + ΔFz_i
Fz_i = softplus(Fz_raw_i) or clamp_min(Fz_raw_i, eps)
L_sumFz = |ΣFz_i - m_eff * g| or bounded dynamic range penalty
```

teacher auxiliary loss：

```text
F2 only:
  L_Fz_teacher = SmoothL1(Fz_i, Fz_true_i)
simulation only
```

训练：

```text
base F1: trainable via rollout + constraint loss
FT3: only FzResidualNN adapter trainable
```

## 6. MuHead

实验引用：`M0-fixed/M1a/M1b/M2-oracle`、`FT2`。这里定义 `MuHead` 的输出设计和参数化方式。

### 6.1 M1a: μ_mean / μ_logvar

默认结构：

```text
input:
  z_shared
  slip_ratio_est_i / slip_angle_est_i
  Fz_i
  wheel_acc_est_i
mlp: MLP([input_dim, 128, 64])
heads:
  μ_mean_i: 4
  μ_logvar_i: 4
activation: SiLU
```

参数化：

```text
μ_mean_i = μ_min + (μ_max - μ_min) * sigmoid(raw_mu)
μ_logvar_i = clamp(raw_logvar, logvar_min, logvar_max)
```

默认范围：

```text
μ_min: 0.05
μ_max: 1.3-1.5
```

### 6.2 M1b: μ_scale / confidence

默认结构：

```text
heads:
  μ_scale_i: 4
  confidence_i: 4
```

参数化：

```text
μ_i = μ_nominal * exp(clip(Δμ_scale_i))
confidence_i = sigmoid(raw_confidence)
```

实现用途：

```text
confidence 用于 loss weighting / calibration diagnostic
small-slip 区 confidence 应降低
```

### 6.3 M2-oracle

只在仿真实验中使用：

```text
μ_i = μ_true_i
```

禁止作为可部署模型或 student input。

## 7. TireResidualNN

实验引用：`T0/T1/T1-no-proj/T2`、`FT4`。这里定义 tire residual 的 force-level 与 parameter-level 两条实现路线。

### 7.1 T1: Force-Level Residual

默认结构：

```text
input:
  z_shared
  slip_ratio_est_i / slip_angle_est_i
  Fz_i
  μ_i or μ_mean_i
  Fx_phy_i / Fy_phy_i
mlp: shared MLP([input_dim, 128, 64])
head: ΔFx_i, ΔFy_i for 4 wheels
activation: SiLU
output: bound_force * tanh(raw)
```

初始 bound：

```text
bound_force_i: 0.05-0.10 * μ_i * Fz_i
```

projection：

```text
[Fx_i, Fy_i] = project_to_friction_ellipse(
  Fx_phy_i + ΔFx_i,
  Fy_phy_i + ΔFy_i,
  μ_i,
  Fz_i
)
```

T1-no-proj：

```text
same residual
skip projection
keep violation metrics
```

### 7.2 T2: Parameter-Level Residual

默认结构：

```text
input: same as T1
head:
  ΔC_alpha_i
  ΔC_kappa_i
  Δμ_scale_i
```

参数化：

```text
C_alpha_eff = C_alpha_nominal * exp(clip(ΔC_alpha))
C_kappa_eff = C_kappa_nominal * exp(clip(ΔC_kappa))
μ_eff = μ_i * exp(clip(Δμ_scale))
```

实现取舍：

```text
更可解释
更强先验
可能在 large slip / low-μ 中弱于 T1
```

## 8. VehicleResidualNN

实验引用：`V0/V1/V1-large/V2-small`、`FT6`。这里定义 final vehicle residual 的位置、容量和输出通道。

默认结构：

```text
input:
  z_shared
  x_dot_physics
  aggregated force/moment diagnostics
  Fz_i, Fx_i, Fy_i
  friction_usage_i
mlp: MLP([input_dim, 64, 32])
head: Δx_dot selected channels
activation: SiLU
output: bound_xdot * tanh(raw)
```

建议第一版输出通道：

```text
Δvx_dot
Δvy_dot
Δr_dot
optional Δp_dot / Δq_dot
```

不建议第一版直接输出：

```text
absolute yaw
absolute roll
absolute pitch
wheel speed state overwrite
```

bound 初值：

```text
Δvx_dot / Δvy_dot: 0.05-0.10 * typical acceleration
Δr_dot: 0.05-0.10 * typical yaw acceleration
```

V1-large：

```text
hidden size x2 or layers +1
diagnostic only, not default deployable model
```

V2-small：

```text
same capacity as V1
output bounded Δx after integration
ablation only
```

实现边界：

```text
VehicleResidualNN 不应长期承担 mass/cg/inertia 等慢变量误差
VehicleResidualNN 的 residual magnitude 应随 FT1 adapter 生效而下降
FT6 同时开放 VehicleParamAdapter 和 VehicleResidualNN，用作上界但也监控两者抢解释权
```

## 9. Module Dependency Graph

第一版假设模块依赖是单向的：

```text
VehicleParamAdapter
  -> mass_eff / inertia_eff / cg_eff
  -> Fz physics
  -> FzResidualNN
  -> MuHead
  -> TirePhysics + TireResidualNN
  -> vehicle physics aggregation
  -> VehicleResidualNN
```

关键假设：

```text
FzResidualNN 不直接依赖 MuHead 输出
MuHead 可以依赖 Fz_i、student-side slip estimates、wheel dynamics 和 observable history
TireResidualNN 可以依赖 MuHead 输出的 μ_i / uncertainty
VehicleResidualNN 只看聚合后的 physics diagnostics，不向前反馈
```

如果实现中引入 Fz/Mu/Tire 的双向依赖，需要增加交互消融，不能只依赖 B4.7 的单向执行顺序。

## 10. Training Schedule

推荐顺序：

```text
Stage 0:
  physics-only sanity
  no neural residual

Stage 1:
  VehicleParamAdapter + FzResidualNN warm-up
  short rollout horizon

Stage 2:
  enable MuHead + TireResidualNN
  include low-μ / Split-μ / transition data

Stage 3:
  enable small VehicleResidualNN
  add residual magnitude penalty

Stage 4:
  low-LR end-to-end training
  increase rollout horizon

Stage 5:
  B6 target fine-tune
  FT1-FT6 × FTD0-FTD5
```

默认 loss 权重先从以下量级搜索：

```text
λ_one_step: 0.1-1.0
λ_constraint: 0.1-10.0
λ_residual: 1e-4-1e-2
λ_smooth: 1e-4-1e-2
λ_teacher: 0.1-1.0 for simulation-only auxiliary losses
```

## 11. Ablation Implementation Notes

每个 ablation 配置必须明确记录：

```text
config name
enabled modules
trainable modules
random seed
train/val/test split id
rollout horizon
optimizer / learning rate
best checkpoint selection metric
```

B4 ablation：

```text
all configs train from scratch
same train/val/test split
same training budget
same early stopping rule
```

B6 fine-tune：

```text
same final single model checkpoint
same target train pools
same target test windows
FT1-FT5 only open specified module
FT6 opens full model
FTD1-FTD5 data subsets nested when possible
```

## 12. Open Design Decisions

仍需在实现前确认：

```text
GRU vs TCN encoder
which x_dot channels VehicleResidualNN may modify
exact friction ellipse projection differentiable form
whether M1a or M1b is default after first MuHead ablation
bound_scale fixed or weakly learnable
teacher simulator dt and student integration method
```
