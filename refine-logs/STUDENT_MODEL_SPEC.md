# STUDENT_MODEL_SPEC

**日期**：2026-05-21  
**状态**：implementation spec v0  
**作用**：定义 student hybrid dynamics model 的代码接口、模块输入输出、tensor shape、配置编号、forward contract 和 loss contract。

## 0. 文档边界

`MODULE_DESIGN.md` 是组件设计说明；本文档是实现入口。训练协议、实验块和 run id 以 `EXPERIMENT_PLAN.md` / `EXPERIMENT_RUN_SPEC.md` 为准。数据字段以 `DATA_SCHEMA_SPEC.md` 为准。

Base configuration:

```text
Base = E1 + T1 + F1 + S1 + M1a + V1 + U0
```

## 1. Package Boundary

建议实现模块：

```text
student_model/
  config.py
  batch.py
  model.py
  losses.py
  rollout.py
  physics/
    vehicle_backbone.py
    steering.py
    load_transfer.py
    tire.py
    integrator.py
  modules/
    encoder.py
    fz_residual.py
    mu_head.py
    tire_residual.py
    vehicle_residual.py
    uncertainty.py
    vehicle_param_adapter.py
```

Public API:

```python
model = StudentHybridModel(config: StudentModelConfig)
out = model.forward(batch: StudentBatch, mode: ForwardMode) -> StudentOutput
loss = compute_loss(out: StudentOutput, batch: StudentBatch, loss_config: LossConfig) -> LossOutput
```

## 2. Input Batch Contract

`StudentBatch` is produced by dataloader from `DATA_SCHEMA_SPEC.md`.

```yaml
StudentBatch:
  observable_history: float32[B, L, d_obs]
  observable_history_mask: bool[B, L]
  current_observable: float32[B, d_obs]
  fixed_vehicle_context: encoded or structured
  nominal_physics_prior: float32[B, d_prior]
  rollout_controls: float32[B, H, d_ctrl]
  targets:
    state: float32[B, H, d_state]
    mask: bool[B, H]
  teacher_aux_labels:
    optional map[str, tensor]
  metadata:
    episode_id, scenario_id, split_role, target_window_id, etc.
```

Required visible inputs:

```text
observable_history
fixed_vehicle_context
nominal_physics_prior
student-side derived features
```

Forbidden inputs:

```text
teacher_aux_labels as model input
episode metadata as model input
teacher hidden params / hashes / true road labels as model input
```

## 3. State And Rollout Contract

v0 supervised state:

```text
state x:
  vx
  vy
  roll
  pitch
  yaw
  p
  q
  r
  omega_fl
  omega_fr
  omega_rl
  omega_rr
```

State tensor:

```text
x_t:      float32[B, d_state]
x_roll:   float32[B, H, d_state]
x_dot:    float32[B, d_xdot]
```

Rollout API:

```python
rollout_out = model.rollout(
    batch: StudentBatch,
    horizon: int,
    teacher_forcing: bool = False,
) -> StudentOutput
```

Rollout requirements:

```text
all dynamics steps are causal
no future target can enter model input
all force/load/residual diagnostics are stored per rollout step
integration method recorded in output diagnostics
```

## 4. Model Configuration

```yaml
StudentModelConfig:
  encoder: E1 | E2 | E3
  steering: S0 | S1
  fz_residual: F0 | F1 | F2
  mu_head: M0-fixed | M1a | M1b | M2-oracle
  tire_residual: T0 | T1 | T1-no-proj | T2
  vehicle_residual: V0 | V1 | V1-large | V2-small
  uncertainty: U0 | U1
  vehicle_param_adapter: disabled | FT1 | FT6
  base_config_name: optional string
```

Default:

```yaml
encoder: E1
steering: S1
fz_residual: F1
mu_head: M1a
tire_residual: T1
vehicle_residual: V1
uncertainty: U0
vehicle_param_adapter: disabled
```

## 5. Shared Encoder

Input:

```text
observable_history: [B, L, d_obs]
observable_history_mask: [B, L]
fixed_vehicle_context: structured or encoded
nominal_physics_prior: [B, d_prior]
```

Output:

```text
z_shared: float32[B, 128]
h_obs: optional float32[B, d_h]
h_context: optional float32[B, 64]
h_prior: optional float32[B, 64]
```

Variants:

```text
E1:
  small GRU encoder, base default

E2:
  causal TCN encoder

E3:
  causal Transformer encoder
```

Requirements:

```text
use valid mask for variable-length history
must be causal
E1/E2/E3 output same z_shared dimension
FT1-FT5 freeze encoder
FT6 may fine-tune encoder
```

## 6. Steering Module

Input:

```text
sw_angle or steer_cmd from current observable / rollout control
steering_ratio_nominal from fixed_vehicle_context
tau_steer_nominal from nominal_physics_prior
z_shared for S1
optional current_state_skip = vx, r for S1-skip
```

Output:

```text
delta_eff_fl: float32[B]
delta_eff_fr: float32[B]
delta_eff_front: float32[B, 2]
steering_diagnostics
```

Variants:

```text
S0:
  first-order steering lag only

S1:
  S0 + bounded steerResidualNN
  default implementation variant: S1-skip
  optional implementation check: S1-min
```

S1 residual:

```text
S1-min input:
  z_shared + delta_cmd + delta_eff_S0

S1-skip input:
  z_shared + delta_cmd + delta_eff_S0 + vx + r

output:
  Δdelta_eff = bound_delta * tanh(raw)
```

## 7. Load Transfer And FzResidualNN

Physics input:

```text
state x
nominal or effective mass/inertia/cg
fixed geometry
steering / drive / brake summaries
```

FzResidualNN input:

```text
z_shared
Fz_physics_i: [B, 4]
vx, vy, r, p, q
steering / brake / drive summary
```

Output:

```text
Fz_i: float32[B, 4]
delta_Fz_i: float32[B, 4]
Fz_physics_i: float32[B, 4]
Fz_budget_physics: float32[B]
```

Variants:

```text
F0:
  Fz_i = Fz_physics_i

F1:
  Fz_i = positivity_projection(Fz_physics_i + bounded ΔFz_i)

F2:
  same as F1, plus optional Fz_true_i auxiliary loss
```

Requirements:

```text
Fz_i nonnegative after projection
Fz budget constraint available
teacher Fz_true_i never used as model input
```

## 8. MuHead

Input:

```text
z_shared
slip_ratio_est_i: [B, 4]
slip_angle_est_i: [B, 4]
Fz_i: [B, 4]
wheel_acc_est_i: [B, 4]
```

Output:

```text
mu_i: float32[B, 4]
mu_uncertainty: optional float32[B, 4]
mu_logvar_i: optional float32[B, 4]
confidence_i: optional float32[B, 4]
```

Variants:

```text
M0-fixed:
  fixed nominal μ

M1a:
  μ_mean_i, μ_logvar_i

M1b:
  μ_scale_i, confidence_i

M2-oracle:
  μ_i = μ_true_i, simulation upper bound only
```

Requirements:

```text
M2-oracle cannot be used for deployable model
small-slip auxiliary supervision must be downweighted
confidence/logvar must remain available for calibration diagnostics
```

## 9. Tire Physics And TireResidualNN

Input:

```text
z_shared
slip_ratio_est_i: [B, 4]
slip_angle_est_i: [B, 4]
Fz_i: [B, 4]
mu_i or mu_mean_i: [B, 4]
Fx_phy_i, Fy_phy_i from tire physics: [B, 4]
```

Output:

```text
Fx_i: float32[B, 4]
Fy_i: float32[B, 4]
Fx_phy_i: float32[B, 4]
Fy_phy_i: float32[B, 4]
delta_Fx_i: optional float32[B, 4]
delta_Fy_i: optional float32[B, 4]
friction_ellipse_violation: float32[B, 4]
```

Variants:

```text
T0:
  tire physics only

T1:
  force-level residual ΔFx_i / ΔFy_i with friction ellipse projection

T1-no-proj:
  same force residual, projection disabled, violation metrics recorded

T2:
  parameter-level residual ΔC_alpha_i / ΔC_kappa_i / Δμ_scale_i
```

Requirements:

```text
T1 default must project to friction ellipse
T1-no-proj must keep violation metrics
TireResidualNN must not bypass MuHead silently; report sensitivity to μ_i
```

## 10. Vehicle Physics And VehicleResidualNN

Input:

```text
state x
Fx_i, Fy_i, Fz_i
fixed geometry
nominal/effective mass/inertia/cg
z_shared
x_dot_physics
aggregated force/moment diagnostics
friction_usage_est_i
```

Output:

```text
x_dot_physics: float32[B, d_xdot]
x_dot: float32[B, d_xdot]
x_next: float32[B, d_state]
delta_x_dot: optional float32[B, selected_channels]
delta_x: optional float32[B, selected_channels]
```

Variants:

```text
V0:
  no final vehicle residual

V1:
  small bounded Δx_dot before integration, base default

V1-large:
  larger Δx_dot head, diagnostic only

V2-small:
  small bounded Δx after integration
```

Requirements:

```text
VehicleResidualNN capacity must remain small
do not overwrite absolute yaw/roll/pitch/wheel speed state directly in v0
teacher-only friction_usage_i cannot be input
report residual magnitude / physics ratio
```

## 11. Uncertainty Wrapper

Input:

```text
z_shared
selected model diagnostics
residual magnitudes
friction_usage_est_i
x_next mean
```

Output:

```text
prediction_mean: float32[B, H, d_target]
prediction_logvar: float32[B, H, d_target]
```

Variants:

```text
U0:
  single-model heteroscedastic uncertainty

U1:
  K=3 deep ensemble with U0 member heads
```

Requirements:

```text
uncertainty wrapper does not change dynamics mean
logvar clamped to configured range
U1 reports aleatoric, epistemic and total variance
```

## 12. VehicleParamAdapter

Stage:

```text
disabled in Stage B base and Stage B ablation
enabled only for FT1 and FT6
```

Input:

```text
pooled z_shared over target window/history
nominal_physics_prior
```

Output:

```text
mass_eff
Ix_eff, Iy_eff, Iz_eff
cg_x_eff, cg_z_eff
optional z_vehicle
adapter diagnostics
```

Requirements:

```text
identity/no-op initialization
window-level slow variables, not timestep-level fast output
low-pass or temporal smoothness across target windows
FT1 trains only VehicleParamAdapter
FT6 trains full model plus VehicleParamAdapter
```

## 13. StudentOutput Contract

Forward output must include:

```yaml
StudentOutput:
  prediction:
    mean: float32[B, H, d_target]
    logvar: optional float32[B, H, d_target]
  rollout:
    x: float32[B, H, d_state]
    x_dot_physics: float32[B, H, d_xdot]
    x_dot_final: float32[B, H, d_xdot]
  intermediates:
    z_shared: float32[B, 128]
    delta_eff_front: float32[B, H, 2]
    Fz_physics_i: float32[B, H, 4]
    Fz_i: float32[B, H, 4]
    mu_i: float32[B, H, 4]
    Fx_phy_i: float32[B, H, 4]
    Fy_phy_i: float32[B, H, 4]
    Fx_i: float32[B, H, 4]
    Fy_i: float32[B, H, 4]
    delta_residuals: map
  diagnostics:
    constraint_metrics: map
    residual_magnitudes: map
    uncertainty_metrics_inputs: map
```

## 14. Loss Contract

Loss components:

```text
L_total =
  L_rollout
+ λ_one_step L_one_step
+ λ_constraint L_constraint
+ λ_residual L_residual_magnitude
+ λ_smooth L_smoothness
+ λ_teacher L_teacher_aux
+ optional L_nll / L_unc_calib
```

Required loss outputs:

```yaml
LossOutput:
  total: scalar
  components:
    rollout:
    one_step:
    constraint:
    residual_magnitude:
    smoothness:
    teacher_aux:
    nll:
    uncertainty_calibration:
  metrics:
    rmse_by_channel:
    constraint_violation:
    residual_norms:
```

Teacher auxiliary usage:

```text
F2 may use Fz_true_i auxiliary loss
M2-oracle may use mu_true_i only as oracle upper bound
MuHead auxiliary may use mu_true_i only when configured for simulation training
teacher labels never enter model input
```

## 15. Fine-Tune Trainability Matrix

| Config | Trainable | Frozen |
|---|---|---|
| `FT0` | none | all |
| `FT1` | `VehicleParamAdapter` | base model |
| `FT2` | MuHead calibration adapter | encoder, main MuHead, TireResidualNN, VehicleResidualNN |
| `FT3` | FzResidualNN adapter | encoder, main FzResidualNN |
| `FT4` | TireResidualNN adapter | encoder, tire physics, MuHead, main TireResidualNN |
| `FT5` | steering adapter | S0, encoder, main steering body |
| `FT6` | full model + VehicleParamAdapter | only non-trainable constants |

Implementation must expose:

```python
model.set_trainability(fine_tune_mode: Literal["FT0", ..., "FT6"]) -> None
```

## 16. Required Diagnostics

Every training/evaluation step must be able to log:

```text
rollout RMSE by horizon and state channel
Fz positivity violation
Fz budget error
friction ellipse violation rate
mu calibration / confidence diagnostics
residual magnitude by module
residual / physics ratio
steering residual magnitude and smoothness
uncertainty NLL / coverage / calibration
adapter parameter drift for FT1-FT6
```

## 17. Model Construction Errors

Implementation must reject:

```text
VehicleParamAdapter enabled in Stage B base / B4 ablation
M2-oracle in deployable or real-data config
F2 teacher auxiliary when Fz_true_i is unavailable
U1 ensemble member count < 3 for U1 run
non-causal encoder configuration
teacher_aux_labels or metadata keys in model input
```

## 18. Minimal Smoke Tests

Required before full training:

```text
forward pass on one DS0 batch
rollout H=1 and H=10 without NaN/Inf
teacher label leakage test
physics-only V0/T0/F0 sanity rollout
tiny overfit on 5-10 short sequences
all FT0-FT6 trainability masks verified
T1 projection produces no friction ellipse violation above tolerance
T1-no-proj records nonzero violation when raw forces exceed limits
```
