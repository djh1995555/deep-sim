# TEACHER_SIMULATOR_DESIGN

**日期**：2026-05-19  
**状态**：draft v1  
**作用**：定义第一版 high-fidelity teacher simulator 的动力学范围、模块边界、内部参数、导出字段、随机化策略和 sanity 验收标准。`DATA_DESIGN.md` 定义用 teacher 生成什么数据；本文档定义 teacher 本身怎么设计。

## 1. 设计目标

teacher simulator 的目标不是成为最终部署模型，而是提供一个比 student 更高保真的、可控的、可导出内部真值的数据生成器。

必须满足：

```text
保真度高于 student double-track backbone
能生成 DS0 / DS1 所需多车、多配置、多工况数据
能导出 Fz/Fx/Fy/μ/slip/suspension/actuator 等 teacher-only 标签
支持 hidden parameter randomization 和 held-out config
支持 sim-to-real proxy perturbation
所有 observable / teacher_only 字段 schema 级隔离
```

非目标：

```text
不追求第一版完整商业级多体仿真精度
不要求 student 与 teacher 使用同一动力学结构
不把 teacher 内部真实参数暴露给 student
不使用 teacher-only 字段作为可部署模型输入
```

## 2. Teacher 与 Student 的复杂度分工

第一版 student 使用可微分 double-track 车辆动力学，核心是稳定、可解释和可训练。

teacher 需要故意更复杂，用来制造 student 需要学习的 residual：

```text
student:
  double-track body dynamics
  simplified Fz physics
  simplified tire physics
  bounded residual modules

teacher:
  6-DOF body dynamics
  nonlinear suspension / roll / pitch load transfer
  combined-slip tire with load sensitivity
  road μ map and transition
  actuator lag / saturation
  sensor noise / bias / filtering
```

因此 teacher 的内部复杂性是实验变量来源；student 只能通过 observable history、fixed_vehicle_context、nominal_physics_prior、adapter 和 residual modules 适配。

## 3. 坐标系与单位约定

必须在代码和数据 schema 中固定以下约定：

```text
body frame:
  x forward
  y left
  z up

world frame:
  z up
  yaw positive counter-clockwise around world z

wheel order:
  [FL, FR, RL, RR]

units:
  length: m
  angle: rad
  angular rate: rad/s
  velocity: m/s
  force: N
  torque: N*m
  mass: kg
  inertia: kg*m^2
```

符号 sanity 必须覆盖：

```text
制动时前轴 Fz 增加
左转时右侧轮 Fz 增加
正驱动力矩使 wheel angular acceleration 增加
Split-μ 制动产生合理 yaw moment
```

## 4. 动力学自由度

### 4.1 Teacher v0 推荐自由度

Teacher v0 使用可实现的高保真近似，而不是完整工业多体模型。

推荐状态：

```text
body pose:
  x_world, y_world, z_world
  roll phi, pitch theta, yaw psi

body velocity:
  vx, vy, vz
  p, q, r

wheel rotation:
  omega_FL, omega_FR, omega_RL, omega_RR

suspension internal states:
  z_susp_FL, z_susp_FR, z_susp_RL, z_susp_RR
  dz_susp_FL, dz_susp_FR, dz_susp_RL, dz_susp_RR

optional tire relaxation states:
  Fx_relax_i
  Fy_relax_i
```

可以理解为：

```text
6-DOF sprung body
+ 4 wheel rotational DOF
+ 4 corner suspension vertical dynamic states
+ optional tire relaxation states
```

第一版如果实现预算有限，可以先不加入完整 tire relaxation dynamics，但必须保留接口，方便 DS2 或极限操控阶段加入。

### 4.2 Student 可见状态子集

teacher 内部状态不等于 student 输入。第一版 student observable 只使用 `DATA_DESIGN.md` 中定义的字段，例如：

```text
vx, vy
roll, pitch, yaw
p, q, r
omega_i
steering / drive / brake signals
fixed_vehicle_context
nominal_physics_prior
```

`z_world / vz / suspension states / tire relaxation states / true internal params` 默认都是 teacher-only 或 metadata，不进入 student input。

## 5. 模块设计

### 5.1 Chassis Dynamics

teacher chassis 至少包含：

```text
6-DOF rigid body dynamics
gravity
roll / pitch / yaw coupling
force and moment aggregation from four contact patches
load transfer induced by acceleration, roll, pitch and suspension compression
```

输出：

```text
body acceleration
angular acceleration
roll / pitch / yaw states
force and moment diagnostics
```

最低实现要求：

```text
低速静止不漂移
直线匀速稳定
对称输入保持左右对称
横向加速度与 roll / lateral load transfer 方向一致
纵向减速度与 pitch / longitudinal load transfer 方向一致
```

### 5.2 Suspension and Load Transfer

每个角使用非线性 spring-damper 近似：

```text
F_susp_i = k_i(z_susp_i) * z_susp_i + c_i(dz_susp_i) * dz_susp_i + bump_stop_i
```

必须支持：

```text
front/rear stiffness difference
left/right small asymmetry
anti-roll bar effect
roll stiffness / damping
pitch stiffness / damping
ride height / static deflection
dynamic load transfer
```

导出 teacher-only：

```text
suspension compression z_susp_i
suspension velocity dz_susp_i
Fz_true_i
roll stiffness / damping params hash
pitch stiffness / damping params hash
```

### 5.3 Tire Model

第一版推荐使用 combined-slip tire model：

```text
Pacejka-like 或 brush-like tire
longitudinal slip ratio κ_i
lateral slip angle α_i
load sensitivity
combined slip coupling
friction ellipse / friction circle saturation
optional relaxation length
```

必须导出：

```text
slip_ratio_true_i
slip_angle_true_i
Fx_true_i
Fy_true_i
Fz_true_i
friction_usage_i
```

内部参数需要随机化但不暴露给 student：

```text
C_alpha_true
C_kappa_true
μ_peak_true
μ_slide_true
load_sensitivity
relaxation_length
tire_radius_effective_offset
```

第一版必须支持的轮胎现象：

```text
低 slip 近似线性
大 slip 饱和
combined braking + steering 下 Fx/Fy 相互挤占
低 μ 下更早饱和
Fz 增大时峰值力增大但非严格线性
```

### 5.4 Road Model

road model 需要返回每个轮胎接地点的有效附着：

```text
μ_i(t, x, y)
surface_label_i
transition_label_i
```

必须支持：

```text
uniform dry / wet / snow / ice
Split-μ left/right
front/rear μ mismatch
diagonal μ mismatch
one-wheel low-μ patch
dry -> wet / snow / ice transition
patchy μ map
```

导出 teacher-only：

```text
μ_true_i
road_surface_label_i
μ_map_id
transition_type
```

注意：`μ_true_i` 只能用于 `MuHead` auxiliary loss、oracle ablation 和 diagnostics，禁止作为 student input。

### 5.5 Steering Actuator

steering actuator 至少包含：

```text
steering ratio
first-order lag
rate limit
angle saturation
compliance / small nonlinear residual
optional deadzone
```

输入：

```text
sw_angle or steer_cmd
```

输出：

```text
delta_eff_FL_true
delta_eff_FR_true
optional delta_eff_RL_true / delta_eff_RR_true for non-front-steer future variants
```

teacher-only：

```text
actuator lag state
effective steering angle delta_eff_i
steering actuator params hash
```

### 5.6 Drive and Brake Actuators

drive/brake actuator 至少包含：

```text
torque command delay
first-order response
torque saturation
wheel-level allocation
brake bias / hydraulic split
ABS-like clipping optional
```

输出：

```text
tau_drv_i_true
tau_brk_i_true
actuator delay states
```

第一版数据可以把 `tau_drv_i / tau_brk_i` 作为 observable；如果未来真实车无法提供四轮力矩，需要在 student 侧增加 torque inference。

### 5.7 Sensor Model

sensor model 作用是把 teacher 真值变成 student 可见观测。

必须支持：

```text
Gaussian noise
constant bias
slowly drifting bias
low-pass filtering
sampling delay
quantization optional
missing sample optional
```

需要覆盖的传感器：

```text
vx / vy proxy
IMU angular rates p/q/r
roll / pitch / yaw estimate
wheel speed
steering angle
drive / brake torque or command
```

导出要求：

```text
observable_*: 加噪/滤波后的 student 可见量
true_*: teacher-only 真值，用于 diagnostics
sensor_noise_profile: metadata
```

## 6. 车辆与参数随机化

teacher 内部参数分三类。

### 6.1 Student 可见固定参数

这些参数进入 `fixed_vehicle_context`：

```text
wheelbase
track_front
track_rear
wheel_radius
wheel_order
steering_layout
drive_layout
brake_layout
```

### 6.2 Student 可见 nominal prior

这些参数作为不精确先验：

```text
mass_nominal
Ix_nominal
Iy_nominal
Iz_nominal
cg_x_nominal
cg_z_nominal
```

### 6.3 Teacher-only 真实参数

这些参数用于生成真实动力学，但不进入 student input：

```text
mass_true
Ix_true / Iy_true / Iz_true
cg_x_true / cg_z_true
roll stiffness / damping
pitch stiffness / damping
suspension nonlinear coefficients
tire stiffness / Pacejka coefficients
load sensitivity
actuator delay / saturation
sensor noise / bias / delay
road μ map
payload / load distribution
```

随机化策略：

```text
同一 vehicle_family 内改变 hidden params，模拟同车型不同车/不同状态
不同 vehicle_family 改变 geometry/layout，模拟新车型
held-out vehicle/config 保留完整 family 或隐藏参数组合
B2 sim-to-real proxy 使用系统性偏移而不是纯随机噪声
```

## 7. 导出字段

### 7.1 Observable

observable 字段以 `DATA_DESIGN.md` 为准。teacher 负责生成这些可见量：

```text
vx, vy
roll, pitch, yaw
p, q, r
omega_i
sw_angle
tau_drv_i
tau_brk_i
steer_cmd, throttle_cmd, brake_cmd
fixed_vehicle_context
nominal_physics_prior
```

### 7.2 Teacher-Only

teacher 必须导出：

```text
Fz_true_i
Fx_true_i
Fy_true_i
μ_true_i
slip_ratio_true_i
slip_angle_true_i
delta_eff_i
suspension states
actuator delay states
road surface labels
sensor true states
true mass/inertia/cg/suspension/tire params or hash
vehicle_internal_params_hash
```

使用规则：

```text
允许用于 auxiliary loss
允许用于 diagnostics
允许用于 oracle upper bound
允许用于 plotting / debugging
禁止进入 student input
```

## 8. 数值积分与采样

推荐实现：

```text
internal integration dt: 1-5 ms
export dt: 10-20 ms, corresponding to 50-100 Hz
integrator: semi-implicit Euler or RK4
episode duration: DS0 10-30 s, DS1 10-60 s
```

要求：

```text
导出时间戳严格单调
所有 observable / teacher_only 字段同一时间基准
传感器 delay 必须在 metadata 中记录
如果 internal dt 与 export dt 不同，必须明确 downsample/filter 方式
```

## 9. Scenario 接口

每个 scenario 由以下对象组成：

```yaml
vehicle_config:
  fixed_vehicle_context: ...
  nominal_physics_prior: ...
  teacher_hidden_params: ...

road_profile:
  road_type: ...
  mu_pattern: ...
  transition_type: ...
  mu_map_seed: ...

control_script:
  steering: ...
  throttle: ...
  brake: ...
  duration_s: ...

actuator_profile:
  steering_delay: ...
  brake_delay: ...
  torque_limits: ...

sensor_profile:
  noise: ...
  bias: ...
  filtering: ...
  delay: ...

split_metadata:
  episode_id: ...
  scenario_id: ...
  split_role: ...
  target_window_id: ...
```

该接口必须能覆盖 `DATA_DESIGN.md` 中 S001-S405 的 DS0/DS1 工况。

## 10. Sanity 与验收标准

### 10.1 DS0 生成前的单元测试

必须通过：

```text
zero-input static test:
  平地静止，速度和角速度不应发散

straight constant speed:
  yaw rate 接近 0，左右 Fz 对称

braking monotonicity:
  brake torque 增大时减速度增大，前轴 Fz 增加

cornering load transfer:
  左转时右侧 Fz 增加，yaw rate 方向正确

split-μ braking:
  左右 Fx 不对称，产生合理 yaw moment

friction limit:
  sqrt(Fx_i^2 + Fy_i^2) / (μ_i Fz_i) 大多数时间不超过合理范围
```

### 10.2 DS0 Debug Dataset 验收

必须通过：

```text
所有 required scenarios 可稳定生成
无 NaN / Inf
dt consistency 通过
units 和 sign tests 通过
observable / teacher_only schema 隔离通过
physics-only student smoke rollout 不立即发散
tiny black-box 和 tiny hybrid model 能 overfit 5-10 条短序列
```

### 10.3 DS1 V1 Research Dataset 验收

必须通过：

```text
多 vehicle/config family 可生成
held-out vehicle/config metadata 完整
held-out road μ metadata 完整
held-out target time window metadata 完整
FTD0-FTD5 数据量 bucket 可复现
vehicle_internal_params_hash 可追踪但不进入 student input
```

## 11. 实现阶段

### TSim-0：Kinematic / Schema Prototype

目标：

```text
跑通 scenario interface
导出完整 schema
验证坐标系、单位、wheel_order、metadata
```

可以先使用简化 tire/load transfer，只用于 pipeline smoke test。

### TSim-1：Dynamic Teacher v0

目标：

```text
6-DOF body
4-wheel rotational dynamics
nonlinear suspension/load transfer
combined-slip tire
road μ map
actuator + sensor model
DS0 全部通过
```

这是 B0 的最低可接受版本。

### TSim-2：DS1 Research Generator

目标：

```text
多车/多配置生成
hidden parameter randomization
held-out family/config split
FTD0-FTD5 target windows
B2 sim-to-real proxy perturbation
```

### TSim-3：DS2 Extension

目标：

```text
large slip / fishhook / emergency maneuvers
tire relaxation dynamics
more realistic sensor / actuator artifacts
MoE tire residual 所需极限操控覆盖
```

## 12. 与其他文档的关系

```text
EXPERIMENT_PLAN.md:
  定义为什么需要 teacher、B0/B1/B2 怎么验收、哪些实验依赖 teacher。

DATA_DESIGN.md:
  定义 teacher 需要生成哪些数据、工况、字段、split 和 metadata。

MODULE_DESIGN.md:
  定义 student 如何使用 observable 和 teacher-only auxiliary labels。

TEACHER_SIMULATOR_DESIGN.md:
  定义 teacher 本身的动力学模块、内部参数、导出真值和 sanity。
```

## 13. 第一版 Checklist

- [ ] 坐标系、单位、wheel_order 固化。
- [ ] Scenario interface 定义并可生成 episode metadata。
- [ ] 6-DOF chassis dynamics 可运行。
- [ ] 四轮转动动力学可运行。
- [ ] 悬架 / dynamic load transfer 可运行。
- [ ] combined-slip tire 可运行。
- [ ] road μ map 支持 uniform / Split-μ / transition / patchy。
- [ ] steering / drive / brake actuator delay 与 saturation 可运行。
- [ ] sensor noise / bias / filtering / delay 可运行。
- [ ] observable / teacher_only 字段导出并 schema 隔离。
- [ ] DS0 sanity 全部通过。
- [ ] DS1 最小多车/多配置数据可生成。
