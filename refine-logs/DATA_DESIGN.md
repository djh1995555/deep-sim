# DATA_DESIGN

**日期**：2026-05-19  
**状态**：current v1  
**作用**：定义 teacher simulator 数据集的车辆/配置策略、工况矩阵、字段 schema、metadata、划分规则、fine-tune 数据量对照和质量检查。`EXPERIMENT_PLAN.md` 只保留数据设计简介；本文档是当前有效的数据方案。

## 1. 设计原则

当前研究目标要求第一版正式数据集就支持多车、多配置、多工况。单车型数据只作为 debug dataset，不作为 full research dataset。

```text
DS0 Debug Dataset:
  目标是检查 teacher simulator、schema、单位、符号和物理 sanity

DS1 V1 Research Dataset:
  目标是训练通用 base model
  验证 held-out vehicle/config
  验证 FT0-FT6 × FTD0-FTD5 fine-tune 数据效率

DS2 Expanded Dataset:
  目标是覆盖更多车型、极限操控和 MoE tire residual
```

字段必须分成两类：

```text
observable:
  真实车可观测输入和状态
  允许作为 student input

teacher_only:
  高自由度 teacher simulator 内部量
  只能用于 loss / diagnostics
  禁止进入 student input
```

## 2. 数据集阶段

### 2.1 DS0 Debug Dataset

用途：

```text
teacher simulator smoke test
schema / metadata 检查
坐标系、单位、符号检查
physics-only rollout sanity
tiny model overfit
```

建议规模：

```text
vehicle count: 1
scenarios: 8-15
episodes per scenario: 5-20
duration per episode: 10-30 s
frequency: 50-100 Hz
```

DS0 可以使用单一 `vehicle_A_debug`，但必须覆盖基本 longitudinal、lateral、braking、turning、Split-μ 和 transition smoke cases。

### 2.2 DS1 V1 Research Dataset

用途：

```text
训练通用 base model
执行 B3 base hybrid / B4 component ablation / B5 cross-configuration generalization
执行 B6 target vehicle fine-tune
验证 fixed_vehicle_context + nominal_physics_prior + adapter 是否足够
```

车辆策略：

```text
第一版正式数据集即采用多车 / 多配置
至少包含 3-5 个 vehicle/config family
每个 family 内包含若干 hidden parameter variants
至少留出 1 个 vehicle/config family 作为 held-out test
```

建议构造方式：

```text
vehicle_A_base
vehicle_A_mass_variant
vehicle_A_cg_variant
vehicle_A_tire_variant
vehicle_A_suspension_variant
vehicle_B_geometry_variant
vehicle_C_drive_layout_variant
```

这里的“多车”可以先由 teacher simulator 参数化生成，不要求第一版就来自多台真实车。

建议规模：

```text
vehicle/config count: 3-5 train families + 1-2 held-out families
scenarios: 40-60
episodes per scenario per vehicle/config: 50-100+
duration per episode: 10-60 s
frequency: 50-100 Hz
```

### 2.3 DS2 Expanded Dataset

用途：

```text
更多真实车型 / 更多几何 layout
不同载荷和轮胎状态
极限操控 large-slip / fishhook / emergency maneuver
MoE tire residual
更强 OOD 泛化验证
```

DS2 不作为第一版主线；等 DS1 单模型结构稳定后再扩展。

## 3. 车辆与参数策略

### 3.1 Student 可见车辆参数与 nominal prior

Student 输入分三类：

```text
time_series_observable:
  运行过程中随时间变化的可观测信号，见第 4 章

fixed_vehicle_context:
  换车时由车辆设计/配置给出的固定结构参数

nominal_physics_prior:
  车企或标定资料可给出的名义质量、惯量、质心和转向执行器先验
```

本节只定义 `fixed_vehicle_context` 和 `nominal_physics_prior`。它们允许作为 student input，但不代表运行时真实值；载荷、轮胎状态、悬架状态、执行器状态等时间变化因素由 observable history、adapter 和 residual modules 适配。

换车时允许给 student 的固定结构参数：

```yaml
fixed_vehicle_context:
  wheelbase: ...
  track_front: ...
  track_rear: ...
  wheel_radius: ...
  wheel_order: [FL, FR, RL, RR]

  steering_layout:
    type: front_wheel_steer
    steered_wheels: [FL, FR]
    input_signal: sw_angle
    steering_ratio_nominal: ...

  drive_layout:
    type: FWD | RWD | AWD
    driven_wheels: [...]

  brake_layout:
    type: four_wheel_observed | hydraulic_split
```

字段解释：

| 字段                                       | 含义             | 约束                                                              |
| ---------------------------------------- | -------------- | --------------------------------------------------------------- |
| `wheelbase`                              | 前后轴中心距离        | 固定几何量，换车时给定                                                     |
| `track_front`                            | 前轴左右轮中心距       | 固定几何量，影响横向载荷转移和 yaw moment arm                                  |
| `track_rear`                             | 后轴左右轮中心距       | 固定几何量，影响横向载荷转移和 yaw moment arm                                  |
| `wheel_radius`                           | 名义有效滚动半径       | 第一版作为固定 nominal 值；胎压/磨损导致的变化交给模型适配                              |
| `wheel_order`                            | 四轮张量字段顺序       | 必须全数据集一致，默认 `[FL, FR, RL, RR]`                                  |
| `steering_layout.type`                   | 转向机构类型         | 第一版默认 `front_wheel_steer`                                       |
| `steering_layout.steered_wheels`         | 被转向执行器影响的车轮    | 前轮转向时为 `[FL, FR]`                                               |
| `steering_layout.input_signal`           | 用于转向输入的观测字段    | 第一版使用 `sw_angle`                                                |
| `steering_layout.steering_ratio_nominal` | 方向盘角到前轮转角的名义比例 | 只作为 nominal ratio；转向迟滞/偏差由 steering module 适配                   |
| `drive_layout.type`                      | 驱动形式           | `FWD`、`RWD` 或 `AWD`                                             |
| `drive_layout.driven_wheels`             | 可输出驱动力矩的车轮集合   | 必须与 `tau_drv_obs_*` 字段和动力分配逻辑一致                                 |
| `brake_layout.type`                      | 制动系统观测/分配形式    | `four_wheel_observed` 表示四轮制动力矩可观测；`hydraulic_split` 表示后续需要分配或估计 |

允许给 student 的 nominal prior：

```yaml
nominal_physics_prior:
  mass_nominal: ...
  Ix_nominal: ...
  Iy_nominal: ...
  Iz_nominal: ...
  cg_x_nominal: ...
  cg_z_nominal: ...
  tau_steer_nominal: ...
```

字段解释：

| 字段 | 含义 | 约束 |
|---|---|---|
| `mass_nominal` | 名义整车质量 | 可来自整备质量、试验装载名义值或车企给定估计；不等于运行时真实质量 |
| `Ix_nominal` | 绕车体 x 轴的名义转动惯量 | 作为 physics backbone 初始先验 |
| `Iy_nominal` | 绕车体 y 轴的名义转动惯量 | 作为 physics backbone 初始先验 |
| `Iz_nominal` | 绕车体 z 轴的名义转动惯量 | 作为 physics backbone 初始先验 |
| `cg_x_nominal` | 名义质心纵向位置 | 坐标原点和正方向必须在 metadata 中固定 |
| `cg_z_nominal` | 名义质心高度 | 载荷/乘员/货物造成的真实变化不直接暴露给 student |
| `tau_steer_nominal` | 名义转向执行器一阶时间常数 | 作为 `S0` steering lag 的 nominal 输入；真实执行器 delay、柔度、间隙、温度/老化漂移不直接暴露给 student |

不放入 `fixed_vehicle_context` 的量：

```text
真实 mass / inertia / cg
载荷变化
轮胎刚度和磨损状态
悬架刚度 / 阻尼真实值
执行器真实 delay / bias
真实 steering time constant / backlash / hysteresis
传感器真实 noise / bias
road μ map
```

这些量在 teacher simulator 内部随机化，或在真实车阶段表现为隐藏状态；student 只能通过可观测历史、nominal prior、adapter 和 residual modules 适配。

### 3.2 Teacher 内部随机化参数

teacher simulator 内部必须设置但不直接暴露给 student 的参数：

```text
真实 mass
真实 Ixx / Iyy / Izz
真实 cg_x / cg_z
载荷变化
roll stiffness / damping
pitch stiffness / damping
tire stiffness
Pacejka / MF coefficients
actuator delay
sensor noise / bias / filtering
road μ map
```

这些参数用于生成不同车辆、不同配置、不同目标时间段状态。student 只能通过 observable history、nominal prior、adapter 和 residual modules 适配。

## 4. Student 时序观测字段

本章只定义随时间变化的真实车可观测信号。完整 student input 是：

```text
student_input =
  time_series_observable history
+ fixed_vehicle_context
+ nominal_physics_prior
```

第一版 `time_series_observable` 字段：

```text
vx, vy
roll, pitch, yaw
p, q, r
omega_fl, omega_fr, omega_rl, omega_rr
sw_angle
tau_drv_obs_fl, tau_drv_obs_fr, tau_drv_obs_rl, tau_drv_obs_rr
tau_brk_obs_fl, tau_brk_obs_fr, tau_brk_obs_rl, tau_brk_obs_rr
steer_cmd, throttle_cmd, brake_cmd
```

字段解释：

| 字段 | 含义 | 备注 |
|---|---|---|
| `vx`, `vy` | 车体坐标系下纵向/横向速度 | 坐标系符号必须在 metadata 中固定 |
| `roll`, `pitch`, `yaw` | 车身姿态角 | 用于长时 rollout 状态监督和姿态相关动力学 |
| `p`, `q`, `r` | 车体角速度 | 分别对应 roll / pitch / yaw rate |
| `omega_fl/fr/rl/rr` | 四轮轮速 | 顺序必须与 `wheel_order` 一致 |
| `sw_angle` | 方向盘角或等价转向输入观测 | 经 `steering_ratio_nominal` 和 steering module 转为有效前轮转角 |
| `tau_drv_obs_fl/fr/rl/rr` | 四轮驱动力矩观测或估计 | sensor-corrupted observable；来源由 `torque_observability_mode` 标注 |
| `tau_brk_obs_fl/fr/rl/rr` | 四轮制动力矩观测或估计 | sensor-corrupted observable；来源由 `torque_observability_mode` 标注 |
| `steer_cmd` | 转向执行器命令 | 用于建模执行器 delay / saturation |
| `throttle_cmd` | 加速踏板或驱动请求 | 与 `tau_drv_obs_*` 一起用于驱动输入一致性检查 |
| `brake_cmd` | 制动踏板或制动请求 | 与 `tau_brk_obs_*` 一起用于制动输入一致性检查 |

`tau_drv_obs_* / tau_brk_obs_*` 不是 teacher 真值。它们必须来自 sensor model 输出，可以是真实四轮力矩传感、ECU 估计、或 actuator inference 估计。若真实车无法提供四轮力矩，仍不得把 teacher-only 的 `tau_drv_true_* / tau_brk_true_*` 暴露给 student。

允许从 `time_series_observable` 和 `fixed_vehicle_context` 在线推导的 student 特征：

```text
wheel_acc_est_i
slip_ratio_est_i
slip_angle_est_i
friction_usage_est_i
```

这些字段是 student-side derived features，不是 teacher 真值。实现中必须保证它们只由可观测信号、固定几何参数和当前模型状态计算得到，不能读取 teacher-only 的 `*_true_i` 字段。

## 5. Teacher-Only Fields

teacher-only 字段：

```text
Fz_true_i
Fx_true_i
Fy_true_i
μ_true_i
slip_ratio_true_i
slip_angle_true_i
Mz_true_i
friction_usage_i
suspension states
unsprung states
camber_true_i / toe_true_i
actuator delay states
tau_drv_true_i / tau_brk_true_i
brake_pressure_i
ABS/TCS/ESC modulation states
brake_temperature_i
road surface labels
road_height_true_i / road_normal_true_i
grade_true / bank_true
sensor bias / delay / quantization / dropout / timestamp jitter states
aero force / moment diagnostics
真实 mass/inertia/cg/suspension/tire 参数
```

使用规则：

```text
允许用于 auxiliary loss
允许用于 diagnostics
允许用于 plotting / analysis
禁止进入 student input
必须做 dataloader 级 leakage test
```

Episode metadata 不属于 teacher-only label，也不属于 student input。metadata 必须保留给 split、audit、diagnostics 和结果追踪，但 dataloader 不得把它拼入模型输入：

```text
scenario_id
vehicle_internal_params_hash
vehicle_internal_params_hash_algo
teacher_feature_flags
torque_observability_mode
split_role
target_window_id
fine_tune_data_bucket
```

## 6. 工况矩阵

工况采用因子化组合，而不是手写孤立 scenario。一个基础 scenario 由以下三类因子组成：

```text
scenario = longitudinal_factor × lateral_factor × road_factor
```

第一版 DS1 完整基础组合规模：

```text
longitudinal factors: 5
lateral factors: 5
road factors: 4 single + 12 split + 12 transition = 28
total base scenarios: 5 × 5 × 28 = 700
```

### 6.1 纵向因子

| ID | 名称 | 建议控制含义 | 目的 |
|---|---|---|---|
| L0 | const v | 目标纵向加速度接近 0，速度保持 | baseline、低激励、μ 不可辨识性检查 |
| L1 | small acceleration | 小油门或小正加速度，约 `+0.3` 到 `+1.0 m/s^2` | 轻微 longitudinal slip |
| L2 | large acceleration | 大油门或较大正加速度，受路面 μ 和动力上限限制 | 驱动饱和、低附着加速 |
| L3 | small braking | 小制动，约 `-0.3` 到 `-1.5 m/s^2` | 轻微制动载荷转移 |
| L4 | large braking | 大制动，受 ABS / 轮胎附着 / actuator saturation 限制 | 强制动、纵向饱和、Split-μ yaw moment |

### 6.2 横向因子

约定左转为正方向，右转为负方向。具体 steering waveform 在第 7 章定义，可使用 step、sine、sweep 或 lane-change 形式。

| ID | 名称 | 建议控制含义 | 目的 |
|---|---|---|---|
| Y0 | no steer | 转向输入接近 0 | 直线 longitudinal dynamics |
| Y1 | small left steer | 小幅左转，目标横向加速度约 `0.05-0.25 g` | 线性区侧向动力学 |
| Y2 | small right steer | 小幅右转，幅值与 Y1 对称 | 左右对称性和符号检查 |
| Y3 | large left steer | 大幅左转，目标进入非线性轮胎区 | 大侧偏、侧向饱和、roll/load transfer |
| Y4 | large right steer | 大幅右转，幅值与 Y3 对称 | 左右对称性和极限侧向工况 |

### 6.3 单一路况因子

| ID | 路况 | 目的 |
|---|---|---|
| R00 | dry | 高附着 baseline |
| R01 | wet | 中等低附着 |
| R02 | snow | 低附着 |
| R03 | ice | 极低附着 |

### 6.4 Split-μ 路况因子

Split-μ 保留左右有序方向。虽然 `left dry / right ice` 与 `left ice / right dry` 是镜像关系，但都需要保留，用于检查 wheel order、yaw moment 符号、左右转向和载荷转移的一致性。

| ID | 路况 | 目的 |
|---|---|---|
| SP00 | left dry / right wet | 轻度左右附着差 |
| SP01 | left dry / right snow | 中强左右附着差 |
| SP02 | left dry / right ice | 强左右附着差 |
| SP03 | left wet / right snow | 中等左右附着差 |
| SP04 | left wet / right ice | 强左右附着差 |
| SP05 | left snow / right ice | 低附着区内差异 |
| SP06 | left wet / right dry | SP00 镜像 |
| SP07 | left snow / right dry | SP01 镜像 |
| SP08 | left ice / right dry | SP02 镜像 |
| SP09 | left snow / right wet | SP03 镜像 |
| SP10 | left ice / right wet | SP04 镜像 |
| SP11 | left ice / right snow | SP05 镜像 |

### 6.5 路面衔接 / transition 因子

Transition 是有序的，`A -> B` 与 `B -> A` 都保留。默认在 episode 中段触发路面变化；若与大制动/大转向组合，应让 transition 落在主要 excitation 期间，以测试 `MuHead` 和 history encoder 的响应延迟。

| ID | 路况 | 目的 |
|---|---|---|
| TR00 | dry -> wet | 高附着到中等附着 |
| TR01 | dry -> snow | 高附着到低附着 |
| TR02 | dry -> ice | 高附着到极低附着 |
| TR03 | wet -> dry | 中等附着恢复到高附着 |
| TR04 | wet -> snow | 中等附着到低附着 |
| TR05 | wet -> ice | 中等附着到极低附着 |
| TR06 | snow -> dry | 低附着恢复到高附着 |
| TR07 | snow -> wet | 低附着恢复到中等附着 |
| TR08 | snow -> ice | 低附着到极低附着 |
| TR09 | ice -> dry | 极低附着恢复到高附着 |
| TR10 | ice -> wet | 极低附着恢复到中等附着 |
| TR11 | ice -> snow | 极低附着恢复到低附着 |

### 6.6 工况详情组

| Group ID | Road factors | Control factors | Scenario count | 用途 |
|---|---|---|---|---|
| CG-SINGLE | R00-R03 | L0-L4 × Y0-Y4 | 100 | 基础单一路面、低附着、combined slip |
| CG-SPLIT | SP00-SP11 | L0-L4 × Y0-Y4 | 300 | Split-μ yaw moment、左右对称性、wheel-order 检查 |
| CG-TRANSITION | TR00-TR11 | L0-L4 × Y0-Y4 | 300 | μ transition、history encoder、MuHead 响应延迟 |

生成规则：

```text
scenario_id = {road_factor_id}-{longitudinal_factor_id}-{lateral_factor_id}

examples:
  R00-L0-Y0      dry + const v + no steer
  R03-L4-Y0      ice + large braking + no steer
  SP02-L4-Y1     left dry / right ice + large braking + small left steer
  SP08-L4-Y2     left ice / right dry + large braking + small right steer
  TR02-L4-Y3     dry -> ice + large braking + large left steer
  TR09-L1-Y0     ice -> dry + small acceleration + no steer
```

### 6.7 可行性与采样规则

所有组合都可以保留，但不能假设每个组合都能达到同一目标加速度或转向响应。低 μ 路面上的 `large acceleration`、`large braking`、`large steer` 应允许进入 actuator saturation、wheel slip 或 closed-loop instability 区域，并在 metadata 中记录是否达到目标指令。

采样建议：

```text
DS0:
  只抽取每组的代表性 smoke cases，用于 schema、符号、物理 sanity

DS1:
  覆盖 CG-SINGLE + CG-SPLIT + CG-TRANSITION 全组合
  每个 scenario 再采样不同速度、控制幅值、持续时间、seed 和 vehicle/config

DS2:
  在 DS1 基础上加入极限操控和更复杂空间路面
```

### 6.8 数据比例与采样权重

不要按工况组合数量直接采样。第 6.6 节中 `CG-SINGLE:CG-SPLIT:CG-TRANSITION = 100:300:300` 只是 scenario space 大小，不应直接变成训练数据比例。否则 split 和 transition 会占训练集的 `6/7`，基础单一路况不足。

DS1 training 推荐 group-level episode 比例：

| Group | 训练比例 | 原因 |
|---|---:|---|
| `CG-SINGLE` | 45% | 学习基础车辆/轮胎动力学、常规低附着和 combined slip 工作点 |
| `CG-SPLIT` | 25% | 覆盖左右附着差、yaw moment、wheel-order 和左右符号检查 |
| `CG-TRANSITION` | 30% | 覆盖 μ transition、history encoder 和 `MuHead` 响应延迟 |

`CG-SINGLE` 组内 road factor 推荐比例：

| Road factor | 比例 |
|---|---:|
| `R00 dry` | 30% |
| `R01 wet` | 30% |
| `R02 snow` | 20% |
| `R03 ice` | 20% |

`CG-SPLIT` 组内按镜像 pair 采样，pair 内左右方向各占 50%：

| Split pair | 包含因子 | 比例 |
|---|---|---:|
| dry / wet | `SP00`, `SP06` | 15% |
| dry / snow | `SP01`, `SP07` | 20% |
| dry / ice | `SP02`, `SP08` | 25% |
| wet / snow | `SP03`, `SP09` | 15% |
| wet / ice | `SP04`, `SP10` | 15% |
| snow / ice | `SP05`, `SP11` | 10% |

`CG-TRANSITION` 组内按 transition pair 采样，方向不强制 50/50；降附着方向略高，用于学习突发低 μ 风险，升附着方向仍需保留用于恢复过程：

| Transition family | 包含因子 | 比例 |
|---|---|---:|
| dry <-> wet | `TR00`, `TR03` | 15% |
| dry <-> snow | `TR01`, `TR06` | 20% |
| dry <-> ice | `TR02`, `TR09` | 25% |
| wet <-> snow | `TR04`, `TR07` | 15% |
| wet <-> ice | `TR05`, `TR10` | 15% |
| snow <-> ice | `TR08`, `TR11` | 10% |

纵向因子推荐训练比例：

| Longitudinal factor | 比例 |
|---|---:|
| `L0 const v` | 20% |
| `L1 small acceleration` | 20% |
| `L2 large acceleration` | 15% |
| `L3 small braking` | 25% |
| `L4 large braking` | 20% |

横向因子推荐训练比例：

| Lateral factor | 比例 |
|---|---:|
| `Y0 no steer` | 25% |
| `Y1 small left steer` | 20% |
| `Y2 small right steer` | 20% |
| `Y3 large left steer` | 17.5% |
| `Y4 large right steer` | 17.5% |

关键 stress 组合可以在上述基础权重上做 `1.5x-2x` oversampling，但不能超过 `2x`，避免训练分布被极端工况主导：

```text
large braking + split μ
large braking + transition
large steer + low μ
brake + steer + snow/ice
dry -> ice
ice -> dry
left dry/right ice 与 left ice/right dry 镜像对
```

评估集必须同时保留两套统计：

```text
balanced test:
  对 group / road / longitudinal / lateral 尽量等权
  用于模型能力比较和 ablation 归因

deployment-weighted test:
  按预期真实使用频率或业务关注频率加权
  用于估计部署场景表现
```

B6 fine-tune 数据不按 DS1 training 分布采样，而应模拟真实新车可采集数据：

| Fine-tune 数据类型 | 推荐比例 | 示例 |
|---|---:|---|
| 常规单一路况 | 70% | dry/wet + `L0/L1/L3` + `Y0/Y1/Y2` |
| 中等激励 | 20% | wet/snow + brake/steer |
| 安全边界内 stress | 10% | low μ、transition、轻度 split |

### 6.9 后续极限操控

DS2 阶段加入：

```text
large slip angle
emergency lane change
fishhook
drift-like high sideslip
combined high braking + steering
```

## 7. 控制输入脚本

控制输入脚本用于实例化第 6 章的 `L* × Y*` 控制因子。每个 scenario 至少绑定一个 script；DS1 中可对同一 `scenario_id` 采样多个 amplitude / duration / seed，用于构造 held-out control amplitude。

```text
constant throttle
step throttle
ramp throttle
step brake
ramp brake
sine steering
sweep steering
step steering
double lane-change steering
combined brake + steering
```

低激励场景必须保留，因为它测试 `MuHead` 的不可辨识性：

```text
low excitation -> μ uncertainty should rise
```

## 8. 数据划分

所有划分必须按完整 episode / scenario / target time window 完成，禁止把同一条 rollout 的相邻片段同时放入 train/test。

### 8.0 Episode 定义

一个 episode 是同一车辆/配置、同一 scenario、同一 control script、同一 road/actuator/sensor profile 下连续生成的一段 rollout。

第一版建议：

```text
duration: 10-60 s
frequency: 50-100 Hz
single scenario/control script per episode
episode 内不跨 train/val/test split
episode_id 全局唯一
```

对于 B6 fine-tune 数据量，`1 episode` 优先选用 1-2 min target data。如果单条原始 rollout 短于该时长，可把同一 target time window 内的多个短 episode 合并成一个 fine-tune bucket，但必须在 metadata 中记录组成关系。

### 8.1 常规划分

```text
train episodes
validation episodes
test episodes
held-out seeds
held-out control amplitudes
held-out actuator/noise settings
```

### 8.2 Held-Out Road μ

目的：验证路面附着泛化。

示例：

```text
train:
  dry / wet / snow core scenarios

test:
  ice
  dry -> ice transition
  unseen Split-μ combinations
  held-out μ numeric values
```

DS1 可以生成第 6 章定义的完整工况组合；held-out road μ 是 split 规则，表示部分 road factor、transition、Split-μ 组合只进入 validation/test，不进入对应训练 split。

### 8.3 Held-Out Vehicle / Config

目的：验证新车/新配置泛化。

示例：

```text
train:
  vehicle_A/B/C families

test:
  held-out vehicle_D
  held-out geometry region
  held-out hidden suspension/tire/actuator parameter combinations
```

### 8.4 Held-Out Target Time Windows

目的：验证 fine-tune 后不是只记住目标训练片段。

每个 target vehicle/config 至少拆分：

```text
target_finetune_window
target_validation_window
target_test_window
```

这些 window 必须时间不重叠，且最好覆盖不同但相关的工况。

### 8.5 FT0-FT6 × FTD0-FTD5 数据量划分

fine-tune 数据量轴：

```text
FTD0: 0 episode / 0 min，仅对应 FT0 no fine-tune
FTD1: 1 episode 或约 1-2 min target data
FTD2: 5 episodes 或约 5-10 min target data
FTD3: 10 episodes 或约 10-20 min target data
FTD4: 30 episodes 或约 30-60 min target data
FTD5: 100 episodes 或约 2-3 h target data
```

矩阵：

```text
rows:    FT0, FT1, FT2, FT3, FT4, FT5, FT6
columns: FTD0, FTD1, FTD2, FTD3, FTD4, FTD5
```

规则：

```text
FT0 only uses FTD0
FT1-FT6 use FTD1-FTD5
FTD1-FTD5 子集尽量 nested
所有 FT 方案从同一个 M8 final single model checkpoint 初始化
所有数据量使用相同 target vehicle/config/time-window pool
所有格子在相同 held-out target_test_window 上评估
```

预算不足时先跑：

```text
FTD1 / FTD3 / FTD5
```

确认趋势后补齐：

```text
FTD2 / FTD4
```

## 9. Episode Metadata

每条 episode 至少包含：

```yaml
episode_id:
vehicle_id:
vehicle_family:
vehicle_config_id:
scenario_id:
road_type:
mu_pattern:
split_mu_type:
transition_type:
control_script:
seed:
duration_s:
dt:
teacher_model_version:
sensor_noise_profile:
actuator_profile:
torque_observability_mode:
fixed_vehicle_context:
nominal_physics_prior:
teacher_feature_flags:
vehicle_internal_params_hash_algo:
vehicle_internal_params_hash:
observable_fields:
teacher_only_fields:
split_role:
target_window_id:
fine_tune_data_bucket:
```

说明：

```text
vehicle_internal_params_hash 使用 SHA-256，对 canonical JSON 形式的 teacher_hidden_params、teacher_feature_flags、vehicle_config hidden ids、actuator_profile、sensor_profile、environment_profile 和 road_profile hidden maps/seeds 求 hash
hash 可以用于 diagnostics，但不暴露给 student
vehicle_internal_params_hash_algo 固定为 `sha256`
teacher_feature_flags 记录 full-fidelity profile 的 enabled_features、disabled_features、downgrade_reason 和 generator_version
torque_observability_mode 记录 `tau_drv_obs_* / tau_brk_obs_*` 的来源，可取 `true_per_wheel_sensor`、`actuator_estimate`、`command_only_projection`
split_role 明确 train/val/test/held-out/fine-tune/test-window
fine_tune_data_bucket 对应 FTD0-FTD5
metadata 字段用于数据管理和审计，不进入 `time_series_observable`、`fixed_vehicle_context` 或 `nominal_physics_prior`
```

## 10. 数据质量检查

每次生成数据后必须检查：

```text
missing values
dt consistency
units
coordinate frame sign
wheel_order consistency
Fz_i >= 0
ΣFz_i reasonable
friction ellipse ratio
wheel speed plausible
torque sign vs wheel acceleration
torque_observability_mode vs tau_drv_obs/tau_brk_obs consistency
braking load transfer direction
cornering lateral load transfer direction
scenario label consistency
observable / teacher_only role consistency
teacher-only leakage
schema validator blocks teacher-only fields in student input
train/test episode overlap
target window overlap
```

B1 sanity 最少需要通过：

```text
zero-input static test
left/right symmetry test
braking monotonicity test
physics-only rollout smoke test
tiny model overfit 5-10 short sequences
```

## 11. 第一版优先级

DS0 Debug Dataset 必做：

```text
CG-SINGLE smoke subset:
  road: R00 dry, R01 wet, R03 ice
  control: L0/L1/L3/L4 × Y0/Y1/Y2

CG-SPLIT smoke subset:
  SP02 left dry / right ice
  SP08 left ice / right dry
  control: L3/L4 × Y0/Y1/Y2

CG-TRANSITION smoke subset:
  TR00 dry -> wet
  TR02 dry -> ice
  TR09 ice -> dry
  control: L0/L3/L4 × Y0/Y1

actuator / sensor smoke:
  steering lag
  brake torque delay
  wheel speed noise/bias
```

DS1 V1 Research Dataset 必做：

```text
CG-SINGLE full Cartesian product:
  R00-R03 × L0-L4 × Y0-Y4

CG-SPLIT full Cartesian product:
  SP00-SP11 × L0-L4 × Y0-Y4

CG-TRANSITION full Cartesian product:
  TR00-TR11 × L0-L4 × Y0-Y4

actuator / sensor perturbation profiles
multi-vehicle / multi-config
held-out road μ
held-out vehicle/config
held-out target time windows
FT0-FT6 × FTD0-FTD5 splits
```

DS2 Expanded Dataset 后做：

```text
extreme handling
more real vehicle classes
more load/tire aging states
MoE tire residual data
```
