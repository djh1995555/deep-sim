# DATA_SCHEMA_SPEC

**日期**：2026-05-21  
**状态**：implementation spec v0  
**作用**：定义外部 simulator 数据生成器、dataset builder、dataloader、student model 共同使用的数据 schema。

## 0. 文档边界

本文档是实现层 schema 的权威来源。`DATA_DESIGN.md` 解释数据设计理由、工况矩阵和采样策略；本文档定义代码必须实现的字段、role、shape、dtype、单位、坐标系、可见性和校验规则。

字段 role 只允许使用以下值：

```text
time_series_observable:
  随时间变化，允许进入 student input history。

fixed_vehicle_context:
  换车时给定的固定结构/layout 参数，允许进入 student input。

nominal_physics_prior:
  可变物理量的名义先验，允许进入 student input，但不等于真实值。

student_derived:
  dataloader 或 student graph 从可见字段在线计算得到，允许作为模块输入。

teacher_aux_label:
  外部 simulator 真值或内部诊断，只允许用于 auxiliary loss / diagnostics / plotting / oracle。

episode_metadata:
  用于 split、audit、tracking、reproducibility，不进入 student input。
```

## 1. Top-Level Episode Record

每条 episode 必须导出一个结构化 record：

```yaml
episode:
  metadata: EpisodeMetadata
  fixed_vehicle_context: FixedVehicleContext
  nominal_physics_prior: NominalPhysicsPrior
  time_series_observable: TimeSeriesObservable
  teacher_aux_labels: TeacherAuxLabels
```

实现约束：

```text
time dimension name: T
wheel dimension name: W
wheel order: [FL, FR, RL, RR]
all floating tensors: float32 unless explicitly stated
all masks: bool
all ids/enums: string
all angular values: rad or rad/s, never degree
```

## 2. Time Series Observable

这些字段允许进入 student input。shape 均以单条 episode 表示；batch 后由 dataloader 变成 `[B, L, ...]` 或 padded sequence。

| Field | Shape | Dtype | Unit | Frame / Convention | Required | Source | Validation |
|---|---:|---|---|---|---|---|---|
| `timestamp` | `[T]` | float64 | s | monotonic episode time | yes | dataset clock | strictly increasing after sensor timestamp processing |
| `vx` | `[T]` | float32 | m/s | body x forward | yes | sensor model | finite; low-speed handling documented |
| `vy` | `[T]` | float32 | m/s | body y left | yes | sensor model | finite |
| `roll` | `[T]` | float32 | rad | right-hand body convention | yes | sensor model | finite; unwrap policy fixed |
| `pitch` | `[T]` | float32 | rad | right-hand body convention | yes | sensor model | finite; unwrap policy fixed |
| `yaw` | `[T]` | float32 | rad | world heading | yes | sensor model | finite; unwrap or sin/cos transform policy fixed |
| `p` | `[T]` | float32 | rad/s | roll rate | yes | sensor model | finite |
| `q` | `[T]` | float32 | rad/s | pitch rate | yes | sensor model | finite |
| `r` | `[T]` | float32 | rad/s | yaw rate | yes | sensor model | finite |
| `omega_fl` | `[T]` | float32 | rad/s | wheel FL | yes | sensor model | finite |
| `omega_fr` | `[T]` | float32 | rad/s | wheel FR | yes | sensor model | finite |
| `omega_rl` | `[T]` | float32 | rad/s | wheel RL | yes | sensor model | finite |
| `omega_rr` | `[T]` | float32 | rad/s | wheel RR | yes | sensor model | finite |
| `sw_angle` | `[T]` | float32 | rad | steering wheel or equivalent command angle | yes | sensor model | finite; sign matches left-positive convention |
| `tau_drv_obs_fl` | `[T]` | float32 | N*m | wheel FL drive torque observable | yes | sensor/ECU/estimator | source recorded by `torque_observability_mode` |
| `tau_drv_obs_fr` | `[T]` | float32 | N*m | wheel FR drive torque observable | yes | sensor/ECU/estimator | same |
| `tau_drv_obs_rl` | `[T]` | float32 | N*m | wheel RL drive torque observable | yes | sensor/ECU/estimator | same |
| `tau_drv_obs_rr` | `[T]` | float32 | N*m | wheel RR drive torque observable | yes | sensor/ECU/estimator | same |
| `tau_brk_obs_fl` | `[T]` | float32 | N*m | wheel FL brake torque observable | yes | sensor/ECU/estimator | source recorded by `torque_observability_mode` |
| `tau_brk_obs_fr` | `[T]` | float32 | N*m | wheel FR brake torque observable | yes | sensor/ECU/estimator | same |
| `tau_brk_obs_rl` | `[T]` | float32 | N*m | wheel RL brake torque observable | yes | sensor/ECU/estimator | same |
| `tau_brk_obs_rr` | `[T]` | float32 | N*m | wheel RR brake torque observable | yes | sensor/ECU/estimator | same |
| `steer_cmd` | `[T]` | float32 | normalized or rad | actuator command | yes | control script / logged command | command unit recorded in metadata |
| `throttle_cmd` | `[T]` | float32 | normalized | drive request | yes | control script / logged command | expected range documented |
| `brake_cmd` | `[T]` | float32 | normalized | brake request | yes | control script / logged command | expected range documented |

Observable fields are sensor-model outputs, not noiseless teacher truth. If true noiseless state is exported, it must go to `teacher_aux_labels`.

## 3. Fixed Vehicle Context

这些字段允许进入 student input。它们描述换车时固定的几何/layout，不包含 mass、inertia、cg、轮胎刚度、悬架刚度等可能随配置或时间变化的真实量。

| Field | Shape | Dtype | Unit | Required | Validation |
|---|---:|---|---|---|---|
| `wheelbase` | `[]` | float32 | m | yes | `> 0` |
| `track_front` | `[]` | float32 | m | yes | `> 0` |
| `track_rear` | `[]` | float32 | m | yes | `> 0` |
| `wheel_radius` | `[]` | float32 | m | yes | `> 0`; nominal effective rolling radius |
| `wheel_order` | `[4]` | string enum | none | yes | exactly `[FL, FR, RL, RR]` for v0 |
| `steering_layout.type` | `[]` | string enum | none | yes | v0 default `front_wheel_steer` |
| `steering_layout.steered_wheels` | `[2]` | string enum | none | yes | v0 `[FL, FR]` |
| `steering_layout.input_signal` | `[]` | string enum | none | yes | v0 `sw_angle` |
| `steering_layout.steering_ratio_nominal` | `[]` | float32 | dimensionless | yes | `> 0`; direction convention documented |
| `drive_layout.type` | `[]` | string enum | none | yes | `FWD`, `RWD`, or `AWD` |
| `drive_layout.driven_wheels` | `[1..4]` | string enum | none | yes | consistent with drive torque observables |
| `brake_layout.type` | `[]` | string enum | none | yes | `four_wheel_observed` or `hydraulic_split` |

## 4. Nominal Physics Prior

这些字段允许进入 student input，但只是不精确先验。运行时真实变化由 observable history、adapter 和 residual modules 适配。

| Field | Shape | Dtype | Unit | Required | Validation |
|---|---:|---|---|---|---|
| `mass_nominal` | `[]` | float32 | kg | yes | `> 0` |
| `Ix_nominal` | `[]` | float32 | kg*m^2 | yes | `> 0` |
| `Iy_nominal` | `[]` | float32 | kg*m^2 | yes | `> 0` |
| `Iz_nominal` | `[]` | float32 | kg*m^2 | yes | `> 0` |
| `cg_x_nominal` | `[]` | float32 | m | yes | coordinate origin documented |
| `cg_z_nominal` | `[]` | float32 | m | yes | `> 0` |
| `tau_steer_nominal` | `[]` | float32 | s | yes | `> 0` |

Forbidden in `fixed_vehicle_context` and `nominal_physics_prior`:

```text
true runtime mass / inertia / cg
true tire stiffness / wear / pressure / temperature
true suspension stiffness / damping
true actuator delay / bias / hysteresis states
true sensor noise / bias / delay states
road μ map or road labels as model input
```

## 5. Student-Derived Features

这些字段不由 dataset 作为真值输入提供；它们只能由 dataloader 或 student graph 基于可见字段计算。

| Field | Shape | Dtype | Unit | Source | Validation |
|---|---:|---|---|---|---|
| `wheel_acc_est_i` | `[T,4]` | float32 | rad/s^2 | finite difference / filtered wheel speed | filter config recorded |
| `slip_ratio_est_i` | `[T,4]` | float32 | dimensionless | observable + geometry + current model state | low-speed clamp documented |
| `slip_angle_est_i` | `[T,4]` | float32 | rad | observable + geometry + current model state | sign convention tested |
| `friction_usage_est_i` | `[T,4]` | float32 | dimensionless | student `Fx/Fy/Fz/μ` graph | never read teacher `friction_usage_i` |

## 6. Teacher Auxiliary Labels

这些字段禁止进入 student input。它们只用于 auxiliary loss、diagnostics、oracle upper bound 和 plotting。

| Field | Shape | Dtype | Unit | Required | Use |
|---|---:|---|---|---|---|
| `Fz_true_i` | `[T,4]` | float32 | N | yes | Fz diagnostics; optional F2 auxiliary |
| `Fx_true_i` | `[T,4]` | float32 | N | yes | tire diagnostics |
| `Fy_true_i` | `[T,4]` | float32 | N | yes | tire diagnostics |
| `mu_true_i` | `[T,4]` | float32 | dimensionless | yes | M2 oracle; optional MuHead auxiliary |
| `slip_ratio_true_i` | `[T,4]` | float32 | dimensionless | yes | slip diagnostics |
| `slip_angle_true_i` | `[T,4]` | float32 | rad | yes | slip diagnostics |
| `Mz_true_i` | `[T,4]` | float32 | N*m | optional | aligning moment diagnostics |
| `friction_usage_i` | `[T,4]` | float32 | dimensionless | yes | diagnostics only |
| `delta_eff_i` | `[T,2]` | float32 | rad | yes | steering diagnostics / FT5 analysis |
| `suspension_states` | nested `[T,...]` | float32 | mixed | yes | diagnostics |
| `unsprung_states` | nested `[T,...]` | float32 | mixed | yes | diagnostics |
| `camber_true_i` | `[T,4]` | float32 | rad | yes | tire diagnostics |
| `toe_true_i` | `[T,4]` | float32 | rad | yes | tire diagnostics |
| `actuator_delay_states` | nested `[T,...]` | float32 | mixed | yes | actuator diagnostics |
| `tau_drv_true_i` | `[T,4]` | float32 | N*m | yes | torque observable audit only |
| `tau_brk_true_i` | `[T,4]` | float32 | N*m | yes | torque observable audit only |
| `brake_pressure_i` | `[T,4]` | float32 | Pa or bar | optional | brake diagnostics |
| `abs_tcs_esc_modulation_states` | nested `[T,...]` | float32/bool | mixed | optional | control intervention diagnostics |
| `brake_temperature_i` | `[T,4]` | float32 | degC or K | optional | thermal diagnostics |
| `road_surface_labels` | `[T,4]` or segment list | string enum | none | yes | analysis only |
| `road_height_true_i` | `[T,4]` | float32 | m | optional | rough-road diagnostics |
| `road_normal_true_i` | `[T,4,3]` | float32 | unit vector | optional | rough-road diagnostics |
| `grade_true` | `[T]` | float32 | rad | optional | road diagnostics |
| `bank_true` | `[T]` | float32 | rad | optional | road diagnostics |
| `sensor_true_states` | nested `[T,...]` | float32 | mixed | optional | sensor audit |
| `sensor_error_states` | nested `[T,...]` | float32 | mixed | optional | sensor audit |
| `aero_force_moment_diagnostics` | `[T,6]` | float32 | N, N*m | optional | aero diagnostics |
| `teacher_hidden_params` | nested scalar map | float32/string | mixed | yes | audit/hash only; not model input |

## 7. Episode Metadata

Metadata is not model input and is not teacher auxiliary label. It must be preserved for split, audit and reproducibility.

| Field | Dtype | Required | Validation |
|---|---|---|---|
| `episode_id` | string | yes | unique |
| `vehicle_id` | string | yes | stable within generated dataset |
| `vehicle_family` | string | yes | used for held-out vehicle/config split |
| `vehicle_config_id` | string | yes | stable vehicle/config identifier |
| `scenario_id` | string | yes | `{road_factor_id}-{longitudinal_factor_id}-{lateral_factor_id}` |
| `road_type` | string enum | yes | one of single/split/transition/extreme |
| `mu_pattern` | string enum | yes | dry/wet/snow/ice/split/transition/patchy |
| `split_mu_type` | string or null | yes | required for split μ scenarios |
| `transition_type` | string or null | yes | required for transition scenarios |
| `control_script` | structured map | yes | includes waveform type, amplitude, duration, seed |
| `seed` | int64 | yes | reproducibility |
| `duration_s` | float32 | yes | `> 0` |
| `dt` | float32 | yes | `> 0`; sampling interval |
| `teacher_model_version` | string | yes | semantic version or git hash |
| `sensor_noise_profile` | structured map | yes | records noise/delay/filter/dropout config |
| `actuator_profile` | structured map | yes | records actuator delay/limit/saturation config |
| `torque_observability_mode` | string enum | yes | `true_per_wheel_sensor`, `actuator_estimate`, or `command_only_projection` |
| `teacher_feature_flags` | structured map | yes | enabled/disabled features, downgrade reason, generator version |
| `vehicle_internal_params_hash_algo` | string | yes | v0 `sha256` |
| `vehicle_internal_params_hash` | string | yes | SHA-256 over canonical hidden parameter JSON |
| `observable_fields` | list[string] | yes | actual exported observable fields |
| `teacher_only_fields` | list[string] | yes | actual exported teacher aux labels |
| `split_role` | string enum | yes | train/val/test/held-out/fine-tune/test-window |
| `target_window_id` | string or null | yes | required for Stage C target windows |
| `fine_tune_data_bucket` | string or null | yes | `FTD0`-`FTD5` when applicable |

## 8. Dataloader Output Contract

Student dataloader must output:

```yaml
batch:
  observable_history: float32[B, L, d_obs]
  observable_history_mask: bool[B, L]
  fixed_vehicle_context: nested tensors or flattened float/categorical encodings
  nominal_physics_prior: float32[B, d_prior]
  targets:
    state_next_or_rollout: float32[B, H, d_state]
    target_mask: bool[B, H]
  teacher_aux_labels:
    optional tensors, excluded from model input namespace
  metadata:
    ids and split/audit fields, excluded from model input namespace
```

Required leakage tests:

```text
No teacher_aux_labels key may appear in model_input.
No episode_metadata key may appear in model_input except fixed_vehicle_context and nominal_physics_prior fields explicitly listed above.
M2-oracle and F2 auxiliary configs must request teacher labels explicitly.
All tensors in model_input must be derivable from time_series_observable, fixed_vehicle_context, nominal_physics_prior or student_derived features.
```

## 9. Coordinate And Sign Conventions

v0 conventions:

```text
body x: forward
body y: left
body z: up
left steer: positive
yaw rate r: positive left turn
wheel order: FL, FR, RL, RR
torque sign: positive drive torque increases wheel angular velocity in forward direction
brake torque fields are nonnegative magnitudes unless explicitly configured otherwise
```

Any deviation must be recorded in metadata and fail the default schema validator unless the experiment explicitly opts in.

## 10. Validator Requirements

The schema validator must check:

```text
required fields exist
dtype and shape match this spec
timestamp monotonicity
no NaN/Inf in numeric fields
wheel_order consistency
unit/sign sanity on braking, turning, load transfer and yaw response smoke cases
teacher_aux_labels not present in model input
episode_metadata not present in model input
torque_observability_mode consistent with torque observable fields
held-out splits have no episode/window overlap
vehicle_internal_params_hash is present and not exposed to student input
```
