# MODULE_DESIGN

**日期**：2026-05-20  
**状态**：current v3
**作用**：定义 student model 和各组件的详细设计方案。

本文档是 student 组件设计的权威来源，覆盖模块职责、输入输出、插入位置、约束、diagnostics、默认网络结构、bounded 参数化、超参初值、训练 schedule 和 ablation 落地方式。`EXPERIMENT_PLAN.md` 只引用 `E/T/F/S/M/V/U/FT` 等实验配置编号，不重复定义具体组件结构。

## 1. 总体原则

第一版模型以可微分物理 backbone 为主，神经网络模块只做受控补偿。

默认约束：

```text
所有 residual head 使用 bounded output
越靠后的 residual 容量越小
所有 teacher-only 字段只用于 loss / diagnostics
所有 model input 必须来自 observable、fixed_vehicle_context、nominal_physics_prior 或 student-side computed diagnostics
所有 ablation 配置从头训练
B6 fine-tune 从同一个 final single model checkpoint 初始化
```

默认时间窗口：

```text
history_len: 0.5-2.0 s
rollout_train_horizon: 1-5 s warm-up, then extend to 10 s
dt: follow dataset sampling rate, resample only in data pipeline
```

## 2. Shared Encoder

编码器实验配置：

```text
E1: small GRU encoder
E2: causal TCN encoder
E3: causal Transformer encoder
```

第一版 base 默认使用 E1。E2/E3 作为 encoder ablation 必做项；三者必须使用相同 observable 字段、相同 `z_shared` 维度和相同 downstream modules。

实验引用：`E1/E2/E3`、`Base = E1 + T1 + F1 + S1 + M1a + V1 + U0` 以及 `FT1-FT6`。数据字段以 `DATA_DESIGN.md` 为准。

### 2.1 Common Interface

```text
observable history
fixed_vehicle_context embedding
nominal_physics_prior embedding
```

输出：

```text
z_shared: 128-dim
```

通用 context / prior 分支：

```text
context_mlp:
  fixed_vehicle_context -> MLP([context_dim, 64, 64])

prior_mlp:
  nominal_physics_prior -> MLP([prior_dim, 64, 64])

fusion:
  concat([h_obs, h_context, h_prior])
  -> MLP([256, 128])
  -> LayerNorm

activation: SiLU
dropout: 0.0-0.1
```

约束：

```text
encoder 不接收 teacher-only 字段
encoder 必须 causal，不允许使用未来帧
encoder 在 FT1-FT5 默认冻结
FT6 才允许整体 fine-tune
```

### 2.2 E1: Small GRU Encoder

默认用途：第一版 base encoder。优先用它建立稳定闭环，因为结构小、支持变长 history、在线递推简单，也便于后续 target window fine-tune。

结构：

```text
input:
  observable_history: [B, L, d_obs]
  valid_mask: [B, L] if variable-length history

observable branch:
  observable_norm
  -> Linear(d_obs, 128)
  -> SiLU
  -> GRU(input=128, hidden=128, layers=1 or 2, batch_first=True)
  -> h_obs = last_valid_hidden

fusion:
  h_obs + context_mlp + prior_mlp
  -> common fusion
  -> z_shared
```

默认超参：

```text
hidden_dim: 128
layers: 1 for first implementation, 2 only if underfit
dropout: 0.0 for 1 layer, 0.05-0.1 for 2 layers
history_len: 0.5-2.0 s
```

诊断重点：

```text
long rollout stability
target window fine-tune stability
hidden state drift across long episodes
low-excitation μ uncertainty calibration
```

### 2.3 E2: Causal TCN Encoder

默认用途：与 E1 平级的必做 encoder ablation。TCN 的优势是训练并行、固定感受野明确、梯度稳定；缺点是对 history length / sampling rate 更敏感。

结构：

```text
input:
  observable_history: [B, L, d_obs]

observable branch:
  observable_norm
  -> Linear(d_obs, 128)
  -> transpose to [B, 128, L]
  -> causal residual TCN blocks
  -> h_obs = last_valid_timestep feature

TCN block, repeated over dilation d in [1, 2, 4, 8]:
  causal Conv1d(128, 128, kernel=3, dilation=d)
  -> SiLU
  -> LayerNorm or GroupNorm
  -> Dropout(0.05)
  -> causal Conv1d(128, 128, kernel=3, dilation=d)
  -> residual add

fusion:
  h_obs + context_mlp + prior_mlp
  -> common fusion
  -> z_shared
```

默认超参：

```text
channels: 128
kernel_size: 3
dilations: [1, 2, 4, 8]
num_blocks: 4
dropout: 0.05
left_padding_only: true
```

实现要求：

```text
必须使用 causal padding 或显式裁剪，禁止看到未来帧
感受野必须覆盖主要 history_len；如果 dt 较高，可增加 dilation 或 block 数
variable-length history 用 mask 取 last valid feature
```

诊断重点：

```text
μ transition response delay
Split-μ yaw / wheel-speed prediction
training throughput
fixed-window sensitivity
```

### 2.4 E3: Causal Transformer Encoder

默认用途：高容量 encoder ablation。它用于验证 attention 是否能更好地从历史片段中找到关键激励，例如短时制动、转向阶跃、路面 transition。第一版不默认部署，除非收益明显且成本可接受。

结构：

```text
input:
  observable_history: [B, L, d_obs]
  valid_mask: [B, L]

observable branch:
  observable_norm
  -> Linear(d_obs, d_model=128)
  -> add causal time encoding
  -> TransformerEncoder layers with causal attention mask
  -> h_obs = last_valid_token feature

Transformer layer, repeated 2-4 times:
  Pre-LayerNorm
  -> MultiHeadSelfAttention(d_model=128, heads=4, causal_mask=True, padding_mask=True)
  -> residual add
  -> Pre-LayerNorm
  -> FFN(128 -> 256 -> 128, activation=SiLU)
  -> residual add

fusion:
  h_obs + context_mlp + prior_mlp
  -> common fusion
  -> z_shared
```

默认超参：

```text
d_model: 128
num_layers: 2 initially, 4 only if underfit
num_heads: 4
ffn_dim: 256
dropout: 0.05-0.1
position_encoding: relative time encoding preferred; sinusoidal acceptable first
causal_mask: true
```

实现要求：

```text
必须同时使用 causal mask 和 padding mask
不得使用 bidirectional encoder 默认实现
history_len 不宜一开始过长，先用 1-2 s 控制成本
参数量和训练预算必须与 E1/E2 报告清楚
```

诊断重点：

```text
transition / intermittent excitation 场景收益
held-out vehicle/config 泛化
过拟合风险
推理延迟和显存
attention 是否集中在有物理意义的激励片段
```

## 3. Steering Module

实验引用：`S0/S1`、`FT5`。这里定义 steering module 的接口、默认结构和约束。

### 3.1 S0: First-Order Steering Lag

形式：

```text
delta_cmd = sw_angle / steering_ratio_nominal
dot(delta_eff) = (delta_cmd - delta_eff) / tau_steer
```

默认：

```text
tau_steer = tau_steer_nominal from nominal_physics_prior
delta_eff clamp: physical steering limit
```

`tau_steer_nominal` 是 student-visible nominal prior，不是真实执行器时间常数。真实 steering delay、柔度、间隙、滞回和饱和误差由 `S1` 的 bounded steerResidualNN、FT5 adapter 或 FT6 full fine-tune 适配。

### 3.2 S1: S0 + steerResidualNN

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
FT5: only steerResidualNN fine-tune adapter trainable
regularization: Δdelta magnitude + temporal smoothness
```

FT5 fine-tune adapter：

```text
freeze S0 tau_steer_nominal input, shared encoder, and main steerResidualNN body
train small residual adapter or head affine/bias on Δdelta_eff output
keep bound_delta unchanged unless explicitly running FT6
```

## 4. FzResidualNN

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
L_sumFz_static = |ΣFz_i - m_eff * g| only for flat quasi-static sanity cases
L_Fz_budget = |ΣFz_i - Fz_budget_physics| for dynamic training
```

`Fz_budget_physics` 必须来自 student-side physics，可包含 grade / bank projection、aero vertical force、body vertical acceleration proxy、rough-road contact event mask 等可观测或模型内部可计算项。不得用 teacher-only `Fz_true_i` 构造 base `F1` 的输入或约束目标。

teacher auxiliary loss：

```text
F2 only:
  L_Fz_teacher = SmoothL1(Fz_i, Fz_true_i)
simulation only
```

训练：

```text
base F1: trainable via rollout + constraint loss
FT3: only FzResidualNN fine-tune adapter trainable
```

FT3 fine-tune adapter：

```text
freeze shared encoder and main FzResidualNN body
train output-layer affine/bias or small bottleneck adapter
keep Fz positivity projection and Fz budget constraint active
```

## 5. MuHead

实验引用：`M0-fixed/M1a/M1b/M2-oracle`、`FT2`。这里定义 `MuHead` 的输出设计和参数化方式。

### 5.1 M1a: μ_mean / μ_logvar

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

### 5.2 M1b: μ_scale / confidence

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

训练信号：

```text
base M1a/M1b:
  train via rollout loss, tire force consistency, friction usage consistency, and optional simulation-only μ auxiliary loss

simulation-only auxiliary:
  L_mu_teacher = masked NLL or SmoothL1 against μ_true_i
  only for M1a/M1b diagnostics or training on simulated data
  never used as student input

real-data / deployment setting:
  no μ_true_i
  use rollout NLL, friction ellipse consistency, transition response, and calibration diagnostics

small-slip handling:
  downweight absolute μ supervision when slip excitation is insufficient
  confidence/logvar must reflect weak observability instead of forcing overconfident μ
```

FT2 fine-tune adapter：

```text
freeze shared encoder and main MuHead body
train μ calibration parameters only:
  M1a: affine/bias on raw_mu and temperature/offset on raw_logvar
  M1b: affine/bias on Δμ_scale and confidence calibration temperature
do not train TireResidualNN or VehicleResidualNN in FT2
```

### 5.3 M2-oracle

只在仿真实验中使用：

```text
μ_i = μ_true_i
```

禁止作为可部署模型或 student input。

## 6. TireResidualNN

实验引用：`T0/T1/T1-no-proj/T2`、`FT4`。这里定义 tire residual 的 force-level 与 parameter-level 两条实现路线。

### 6.1 T1: Force-Level Residual

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

### 6.2 T2: Parameter-Level Residual

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

FT4 fine-tune adapter：

```text
freeze shared encoder, tire physics, MuHead, and main TireResidualNN body
train output-layer affine/bias or small bottleneck adapter inside TireResidualNN
keep friction ellipse projection active for T1 unless the config is explicitly T1-no-proj
```

## 7. VehicleResidualNN

实验引用：`V0/V1/V1-large/V2-small`、`FT6`。这里定义 final vehicle residual 的位置、容量和输出通道。

默认结构：

```text
input:
  z_shared
  x_dot_physics
  aggregated force/moment diagnostics
  Fz_i, Fx_i, Fy_i
  friction_usage_est_i
mlp: MLP([input_dim, 64, 32])
head: Δx_dot selected channels
activation: SiLU
output: bound_xdot * tanh(raw)
```

`friction_usage_est_i` 与 `DATA_DESIGN.md` 的 student-side derived feature 命名保持一致，必须由 student 当前计算图中的 `Fx_i/Fy_i/Fz_i/μ_i` 计算：

```text
friction_usage_est_i =
  sqrt(Fx_i^2 + Fy_i^2) / max(μ_i * Fz_i, eps)
```

teacher-only 的 `friction_usage_i` 只能用于 diagnostics、auxiliary loss 或 plotting，禁止进入 `VehicleResidualNN` 输入。

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

## 8. Uncertainty Wrapper

实验引用：`U0/U1`。这里定义 base 中的单模型不确定性和 ensemble 对照。

### 8.1 U0: Single-Model Heteroscedastic Uncertainty

U0 是 base 默认 uncertainty wrapper，不改变 physics rollout 的均值路径，只为预测目标输出 aleatoric uncertainty。

默认结构：

```text
input:
  z_shared
  selected model diagnostics
  residual magnitudes
  friction_usage_est_i
head:
  logvar for supervised state channels or state increments
activation: SiLU
logvar = clamp(raw_logvar, logvar_min, logvar_max)
```

默认输出：

```text
prediction mean: from hybrid dynamics rollout
prediction logvar: from U0 head
```

训练 loss：

```text
L_nll = 0.5 * exp(-logvar) * (target - mean)^2 + 0.5 * logvar
L_unc_calib = optional calibration / coverage penalty
```

约束：

```text
U0 不接收 teacher-only 字段
U0 不改变 physics force / state mean
rollout mean 使用 hybrid dynamics 输出
uncertainty 只用于 NLL、coverage、calibration、OOD diagnostics 和风险评估
```

### 8.2 U1: K=3 Deep Ensemble

U1 是 B4.6 的 uncertainty 对照，不是第一版默认部署结构。

训练协议：

```text
train K=3 independent copies of the selected single-model architecture
use different random seeds and optional bootstrap train splits
each member keeps the same E/T/F/S/M/V/U0 structure
record per-member checkpoint, seed, split, and validation metric
```

推理合成：

```text
member_mean_k = prediction mean from ensemble member k
mean_ensemble = mean_k(member_mean_k)
aleatoric_var = mean_k(exp(logvar_k))
epistemic_var = var_k(member_mean_k)
total_var = aleatoric_var + epistemic_var
```

成功判据与 `EXPERIMENT_PLAN.md` B4.6 保持一致：U1 必须改善 calibration / coverage / OOD detection 等 uncertainty 指标；如果只改善 RMSE，不作为 uncertainty 贡献。

## 9. VehicleParamAdapter

实验引用：`FT1` 和 `FT6`。这里定义目标车/目标时间段慢变量 adapter 的接口、默认结构和参数化方式。

`VehicleParamAdapter` 不属于 Stage B base model，也不参与 B4 组件级 ablation。Stage B 多车多工况训练时，student physics 直接使用 `nominal_physics_prior` 中的 `mass_nominal / inertia_nominal / cg_nominal / tau_steer_nominal`，不启用 adapter。这样可以让 base 表示真正的通用动力学能力，避免 adapter 在多车训练中按 episode 吸收车辆差异，削弱 E/T/F/S/M/V/U ablation 的可解释性。

`VehicleParamAdapter` 只在 Stage C fine-tune 阶段启用：

```text
FT1:
  freeze base model
  enable VehicleParamAdapter
  train only VehicleParamAdapter on target vehicle/time-window data

FT6:
  enable VehicleParamAdapter
  full model fine-tune as upper bound
```

实现上，`VehicleParamAdapter` 从 final single model checkpoint 旁路接入，初始化为 identity/no-op：

```text
Δmass_scale = 0
ΔI_scale = 0
Δcg_x = 0
Δcg_z = 0
optional z_vehicle = 0
```

默认结构：

```text
input: pooled z_shared over target window/history + nominal_physics_prior
adapter_mlp: MLP([128 + prior_dim, 128, 64])
heads:
  Δmass_scale: 1
  ΔIx_scale / ΔIy_scale / ΔIz_scale: 3
  Δcg_x / Δcg_z: 2
  optional z_vehicle: 16
activation: SiLU
```

慢变量约束：

```text
pooled z_shared = mean/attention pooling over history window, not per-step instantaneous latent
mass_eff / inertia_eff / cg_eff 在一个 rollout window 内保持常值
跨 window 更新时使用 low-pass smoothing 或 temporal smoothness penalty
禁止让 VehicleParamAdapter 按 timestep 输出快速变化量
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
base training: disabled / identity / no-op
B4 ablation: disabled in all configs
FT1: only VehicleParamAdapter trainable from final single model checkpoint
FT6: VehicleParamAdapter trainable together with full model
regularization: parameter delta magnitude + temporal smoothness / low-pass update
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

## 10. Fine-Tune Adapter Granularity

FT1-FT5 都从同一个 final single model checkpoint 初始化。除指定模块的 fine-tune adapter 外，shared encoder、physics backbone 和其他 residual modules 默认冻结。

```text
FT1:
  enable and train VehicleParamAdapter
  mass_eff / inertia_eff / cg_eff 保持 window-level slow variables

FT2:
  train MuHead calibration adapter only
  do not train TireResidualNN / VehicleResidualNN

FT3:
  train FzResidualNN adapter only
  keep Fz projection and budget constraints active

FT4:
  train TireResidualNN adapter only
  keep tire physics and MuHead fixed

FT5:
  train steerResidualNN / SteeringHead adapter only
  keep S0 steering physics fixed

FT6:
  full model fine-tune
  used as upper bound and overfitting-risk reference
```

所有 FT1-FT5 必须记录 adapter 参数漂移、residual magnitude drift、held-out target window 性能和物理约束变化。

## 11. Module Dependency Graph

第一版假设模块依赖是单向的：

```text
Stage B base graph:
  observable history + fixed_vehicle_context + nominal_physics_prior
  -> Shared Encoder E1/E2/E3
  -> z_shared

  nominal_physics_prior
  -> mass_eff / inertia_eff / cg_eff
  -> Fz physics

  z_shared + Fz physics
  -> FzResidualNN

  z_shared + Fz_i + student-side slip / wheel dynamics
  -> MuHead

  z_shared + Fz_i + μ_i + slip states
  -> TirePhysics + TireResidualNN

  -> vehicle physics aggregation

  z_shared + vehicle physics diagnostics
  -> VehicleResidualNN
  -> UncertaintyWrapper

Stage C FT1/FT6 add-on:
  VehicleParamAdapter
  -> mass_eff / inertia_eff / cg_eff
  -> same downstream physics graph
```

关键假设：

```text
FzResidualNN 不直接依赖 MuHead 输出
MuHead 可以依赖 Fz_i、student-side slip estimates、wheel dynamics 和 observable history
TireResidualNN 可以依赖 MuHead 输出的 μ_i / uncertainty
VehicleResidualNN 只看聚合后的 student-side physics diagnostics，不向前反馈
UncertaintyWrapper 只包裹最终预测分布，不改变 dynamics mean
```

如果实现中引入 Fz/Mu/Tire 的双向依赖，需要增加交互消融，不能只依赖 B4.8 的单向执行顺序。

## 12. Training Schedule

推荐顺序：

```text
Stage 0:
  physics-only sanity
  no neural residual

Stage 1:
  FzResidualNN warm-up
  short rollout horizon

Stage 2:
  enable MuHead + TireResidualNN
  include low-μ / Split-μ / transition data

Stage 3:
  enable small VehicleResidualNN
  add residual magnitude penalty

Stage 4:
  enable U0 heteroscedastic uncertainty after deterministic rollout is stable
  low-LR end-to-end training
  increase rollout horizon

Stage 5:
  B6 target fine-tune
  enable VehicleParamAdapter only for FT1 and FT6
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

## 13. Ablation Implementation Notes

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
VehicleParamAdapter disabled in all B4 configs
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

## 14. Open Design Decisions

仍需在实现前确认：

```text
final encoder choice after E1/E2/E3 ablation
which x_dot channels VehicleResidualNN may modify
exact friction ellipse projection differentiable form
whether M1a or M1b is default after first MuHead ablation
bound_scale fixed or weakly learnable
teacher simulator dt and student integration method
```
