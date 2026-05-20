# Teacher Simulator 设计

**日期**：2026-05-20
**状态**：当前 v3，中文化修订
**作用**：定义高保真 teacher simulator 的动力学范围、模块边界、内部参数、导出字段、随机化策略和 sanity 验收标准。`DATA_DESIGN.md` 定义用 teacher 生成什么数据；本文档定义 teacher 本身如何设计。

## 1. 设计目标

teacher simulator 的目标不是成为最终部署模型，而是提供一个比 student 更高保真的、可控的、可导出内部真值的数据生成器。

当前策略是：能稳定实现并能提高实车相似度的 feature 默认全部开启。teacher 的复杂度可以高于第一版 student；student 是否能学到这些效应，由 residual、adapter、ablation 和 fine-tune 实验验证。

必须满足：

```text
保真度高于 student double-track backbone
所有真实车相关效应默认开启，关闭任何 feature 必须写入 metadata
能生成 DS0 / DS1 所需多车、多配置、多工况数据
能导出 Fz/Fx/Fy/μ/slip/suspension/actuator 等 teacher-only 标签
支持隐藏参数随机化 hidden parameter randomization 和 held-out config
支持 sim-to-real proxy 扰动
所有 observable / teacher_only 字段 schema 级隔离
```

非目标：

```text
不要求达到商业多体仿真软件的已标定精度
不要求 student 与 teacher 使用同一动力学结构
不把 teacher 内部真实参数暴露给 student
不使用 teacher-only 字段作为可部署模型输入
```

Feature Profile（特性配置）：

```text
默认 profile: full-fidelity
禁用 feature 策略:
  仅允许用于 debug / ablation / 计算预算受限的临时降级
  必须写入 episode metadata
  不能混入正式 DS1 训练数据而不标注
```

可行性与降级策略：

```text
正式 DS1 数据默认不做 silent degradation
scenario 数值不稳定时，先减小内部 dt 或增加 sub-step 重试
重试仍失败时，将 episode 标记为 invalid，不关闭 feature 混入训练集
降级 profile 只允许用于 debug / ablation，并必须通过 teacher_feature_flags 明确区分
```

## 2. Teacher 与 Student 的复杂度分工

第一版 student 使用可微分 double-track 车辆动力学，核心是稳定、可解释和可训练。

teacher 需要故意更复杂，用来制造 student 需要学习的 residual：

```text
student:
  double-track 车体动力学
  简化 Fz 物理模型
  简化轮胎物理模型
  有界 residual modules

teacher:
  6-DOF 车体动力学
  非簧载质量、非线性悬架、roll / pitch 载荷转移
  包含载荷、camber、热、磨损效应的 transient combined-slip tire
  road μ map、路面过渡、坡度、横坡和粗糙度
  气动效应、滚动阻力和风扰动
  包含 ABS/TCS/ESC-like modulation 的 steering / drive / brake actuator dynamics
  sensor noise / bias / filtering / delay / quantization / dropout
```

因此 teacher 的内部复杂性是实验变量来源；student 只能通过 observable history、fixed_vehicle_context、nominal_physics_prior、adapter 和 residual modules 适配。

## 3. 坐标系与单位约定

必须在代码和数据 schema 中固定以下约定：

```text
车体坐标系 body frame:
  x 向前
  y 向左
  z 向上

世界坐标系 world frame:
  z 向上
  yaw 绕 world z 逆时针为正

车轮顺序 wheel order:
  [FL, FR, RL, RR]

单位:
  长度: m
  角度: rad
  角速度: rad/s
  速度: m/s
  力: N
  力矩: N*m
  质量: kg
  转动惯量: kg*m^2
```

符号 sanity 必须覆盖：

```text
制动时前轴 Fz 增加
左转时右侧轮 Fz 增加
正驱动力矩使车轮角加速度增加
Split-μ 制动产生合理 yaw moment
```

## 4. 动力学自由度

### 4.1 Teacher 默认自由度

Teacher 默认使用 full-fidelity profile。目标是尽可能接近实车数据生成机制，而不是为了贴近 student 简化模型。

推荐状态：

```text
车体位姿 body pose:
  x_world, y_world, z_world
  roll phi, pitch theta, yaw psi

车体速度 body velocity:
  vx, vy, vz
  p, q, r

车轮转动 wheel rotation:
  omega_FL, omega_FR, omega_RL, omega_RR

悬架 / 非簧载质量状态:
  z_unsprung_FL, z_unsprung_FR, z_unsprung_RL, z_unsprung_RR
  dz_unsprung_FL, dz_unsprung_FR, dz_unsprung_RL, dz_unsprung_RR

悬架内部状态:
  z_susp_FL, z_susp_FR, z_susp_RL, z_susp_RR
  dz_susp_FL, dz_susp_FR, dz_susp_RL, dz_susp_RR

tire transient states（轮胎瞬态状态）:
  Fx_relax_i
  Fy_relax_i
  tire_temperature_i
  tire_wear_i
  pressure_effective_i

actuator internal states（执行器内部状态）:
  steering_lag_state
  brake_pressure_i
  drive_torque_state_i
  ABS/TCS modulation state_i

sensor internal states（传感器内部状态）:
  bias_state
  filter_state
  timestamp_jitter_state
```

可以理解为：

```text
6-DOF 簧载车体
+ 4 个车轮转动 DOF
+ 4 个非簧载垂向 DOF
+ 4 个角点悬架垂向动态状态
+ 轮胎力 relaxation / thermal / wear / pressure 状态
+ steering / brake / drive actuator 状态
+ sensor noise / bias / delay / filter 状态
```

所有上述状态默认启用。为 debug 可以临时关闭某些子模块，但正式 DS1 / DS2 episode 必须在 metadata 中记录 feature flags，避免不同保真度数据被混用。

### 4.2 Student 可见状态子集

teacher 内部状态不等于 student 输入。第一版 student observable 只使用 `DATA_DESIGN.md` 中定义的字段，例如：

```text
vx, vy
roll, pitch, yaw
p, q, r
omega_i
steering / drive / brake 信号
fixed_vehicle_context
nominal_physics_prior
```

`z_world / vz / suspension states / tire relaxation states / true internal params` 默认都是 teacher-only 或 metadata，不进入 student input。

## 5. 模块设计

### 5.1 Chassis Dynamics（车身动力学）

teacher chassis 至少包含：

```text
6-DOF 刚体动力学
重力
roll / pitch / yaw 耦合
四个接地点的力和力矩汇总
由加速度、roll、pitch 和悬架压缩引起的载荷转移
气动阻力 / 升力 / 下压力 / pitch moment
滚动阻力
路面坡度和横坡投影
侧风扰动
载荷分布
车身柔度 / flex proxy
内部姿态使用 quaternion 或 rotation matrix 更新，对外导出 Euler 角
```

输出：

```text
车体加速度
角加速度
roll / pitch / yaw 状态
力和力矩 diagnostics
气动力 / 气动力矩 diagnostics
路面坡度 / 横坡 diagnostics
```

最低实现要求：

```text
低速静止不漂移
直线匀速稳定
对称输入保持左右对称
横向加速度与 roll / lateral load transfer 方向一致
纵向减速度与 pitch / longitudinal load transfer 方向一致
坡道上静止/匀速的重力分量方向正确
横坡上左右 Fz 分配方向正确
```

### 5.2 Suspension and Load Transfer（悬架与载荷转移）

每个角使用非线性 spring-damper 近似：

```text
k_eff_i = k_i(z_susp_i)
c_eff_i = c_i(dz_susp_i)
F_susp_i = k_eff_i * z_susp_i + c_eff_i * dz_susp_i + bump_stop_i - rebound_stop_i
```

必须支持：

```text
簧载 / 非簧载质量垂向动力学
轮胎垂向刚度 / 阻尼
前后轴刚度差异
左右小幅不对称
anti-roll bar 效应
roll 刚度 / 阻尼
pitch 刚度 / 阻尼
roll center height（侧倾中心高度）
pitch center / anti-dive / anti-squat 近似
ride height / static deflection（车身高度 / 静态压缩量）
动态载荷转移
bump stop / rebound stop（压缩 / 回弹限位）
悬架行程限位
camber gain（外倾增益）
toe compliance（前束柔度）
横向力作用下的 steering compliance
bushing compliance（衬套柔度）
粗糙路面下的 wheel hop
```

导出 teacher-only：

```text
suspension compression: z_susp_i（悬架压缩量）
suspension velocity: dz_susp_i（悬架压缩速度）
unsprung displacement / velocity（非簧载位移 / 速度）
Fz_true_i
camber_true_i
toe_true_i
roll 刚度 / 阻尼参数 hash
pitch 刚度 / 阻尼参数 hash
悬架几何参数 hash
```

### 5.3 Tire Model（轮胎模型）

默认使用 transient combined-slip tire model：

```text
Pacejka-like 或 brush-like tire
纵向滑移率 longitudinal slip ratio κ_i
侧偏角 lateral slip angle α_i
载荷敏感性 load sensitivity
combined slip coupling（联合滑移耦合）
friction ellipse / friction circle 饱和
relaxation length dynamics（松弛长度动态）
camber thrust（外倾推力）
aligning moment Mz（回正力矩）
turn slip effect（转弯滑移效应）
胎压效应
胎温效应
轮胎磨损 / 老化效应
环境温度 / 路面温度效应
低速 slip 保护
```

必须导出：

```text
slip_ratio_true_i
slip_angle_true_i
Fx_true_i
Fy_true_i
Fz_true_i
Mz_true_i
friction_usage_i
tire_temperature_i
tire_wear_i
pressure_effective_i
```

内部参数需要随机化但不暴露给 student：

```text
C_alpha_true
C_kappa_true
μ_peak_true
μ_slide_true
load_sensitivity（载荷敏感性）
relaxation_length（松弛长度）
tire_radius_effective_offset（有效轮胎半径偏移）
camber_stiffness（外倾刚度）
aligning_moment_coefficients（回正力矩系数）
temperature_sensitivity（温度敏感性）
wear_rate（磨损速率）
pressure_sensitivity（胎压敏感性）
```

第一版必须支持的轮胎现象：

```text
低 slip 近似线性
大 slip 饱和
combined braking + steering 下 Fx/Fy 相互挤占
低 μ 下更早饱和
Fz 增大时峰值力增大但非严格线性
steering transient 下轮胎力有 relaxation lag
低速起步/停车时 slip 计算不发散
胎温/磨损/胎压变化改变峰值附着和刚度
有合理回正力矩趋势
```

### 5.4 Road Model（路面模型）

road model 需要返回每个轮胎接地点的有效附着和路面几何：

```text
μ_i(t, x, y)
surface_label_i
transition_label_i
road_height_i
road_normal_i
road_grade_i
road_bank_i
roughness_i
```

必须支持：

```text
均一路面 dry / wet / snow / ice
Split-μ left/right
前后轴 μ mismatch
对角 μ mismatch
单轮 low-μ patch
dry -> wet / snow / ice transition
patchy μ map
空间平滑的 μ transition length
路面坡度 road grade
横坡 road bank
路面粗糙度 / bump / pothole profile
水膜 / 积水 patch
雪 / 冰厚度 proxy
crosswind profile metadata（侧风 profile metadata）
```

导出 teacher-only：

```text
μ_true_i
road_surface_label_i
μ_map_id
transition_type
road_height_true_i
road_normal_true_i
grade_true
bank_true
roughness_profile_id
```

注意：`μ_true_i` 只能用于 `MuHead` auxiliary loss、oracle ablation 和 diagnostics，禁止作为 student input。

### 5.5 Steering Actuator（转向执行器）

steering actuator 至少包含：

```text
steering ratio（转向传动比）
一阶滞后 first-order lag
rate limit（速率限制）
角度饱和 angle saturation
柔度 / 小幅非线性 residual
deadzone（死区）
backlash / hysteresis（间隙 / 滞回）
转向柱扭转柔度 steering column torsional compliance
助力增益变化 assist gain variation
路感反馈 proxy
左右转向不对称
```

输入：

```text
sw_angle 或 steer_cmd
```

输出：

```text
delta_eff_FL_true
delta_eff_FR_true
delta_eff_RL_true / delta_eff_RR_true，如果 steering_layout 包含后轮转向
```

teacher-only：

```text
actuator lag state（执行器滞后状态）
backlash / hysteresis state（间隙 / 滞回状态）
effective steering angle: delta_eff_i（等效转角）
steering actuator params hash（转向执行器参数 hash）
```

### 5.6 Drive and Brake Actuators（驱动与制动执行器）

drive/brake actuator 至少包含：

```text
力矩命令延迟 torque command delay
一阶响应 first-order response
力矩饱和 torque saturation
车轮级分配 wheel-level allocation
制动偏置 / 液压分路 brake bias / hydraulic split
制动液压建压 / 释放不对称
ABS-like pressure modulation（类 ABS 压力调制）
TCS-like drive torque modulation（类 TCS 驱动力矩调制）
ESC-like yaw moment intervention，当 vehicle_config 启用时生效
regen / friction brake blending（能量回收 / 摩擦制动混合）
制动温度 / fade
驱动力矩 map / motor lag
engine / motor torque map（发动机 / 电机力矩 map）
gearbox / shift delay，如果 drive_layout 需要
传动系扭转柔度 driveline torsional compliance
differential / torque vectoring 近似
```

输出：

```text
tau_drv_true_i
tau_brk_true_i
actuator delay states（执行器延迟状态）
brake_pressure_i
ABS/TCS/ESC modulation states
brake_temperature_i
regen_blend_ratio_i
gear_state / shift_state
```

力矩字段边界：

```text
tau_drv_true_i / tau_brk_true_i:
  teacher-only 真值，只能用于 diagnostics / auxiliary loss / oracle analysis

tau_drv_obs_i / tau_brk_obs_i:
  sensor model 输出的 student observable，可以来自真实四轮力矩传感、ECU 估计或 actuator inference

torque_observability_mode:
  写入 metadata，标明 observable torque 的来源和可信度
```

如果真实车无法提供四轮力矩，student 侧不能读取 `tau_drv_true_i / tau_brk_true_i`；只能使用 `tau_drv_obs_i / tau_brk_obs_i` 的估计版本，或在后续版本显式加入 actuator inference。

### 5.7 Sensor Model（传感器模型）

sensor model 作用是把 teacher 真值变成 student 可见观测。

必须支持：

```text
Gaussian noise（高斯噪声）
constant bias（常值偏置）
slowly drifting bias（慢变漂移偏置）
correlated multi-sensor bias（多传感器相关偏置）
low-pass filtering（低通滤波）
sampling delay（采样延迟）
timestamp jitter（时间戳抖动）
clock offset between sensor groups（传感器组之间的时钟偏移）
quantization（量化）
missing sample / dropout（缺采样 / 丢样）
spike / outlier（尖峰 / 异常值）
IMU mounting offset / misalignment（IMU 安装偏移 / 失准）
wheel speed encoder ticks（轮速编码器 tick）
CAN signal rate mismatch（CAN 信号频率不一致）
```

需要覆盖的传感器：

```text
vx / vy proxy
ax / ay / az proxy
IMU angular rates: p/q/r
roll / pitch / yaw 估计
轮速 wheel speed
转向角 steering angle
drive / brake torque 或 command
用于 diagnostics 的 GNSS / odometry position proxy
```

导出要求：

```text
observable_*: 加噪/滤波后的 student 可见量
true_*: teacher-only 真值，用于 diagnostics
sensor_noise_profile: 噪声 metadata
sensor_timing_profile: 时序 metadata
sensor_mounting_profile: 安装 metadata
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
tau_steer_nominal
```

### 6.3 Teacher-only 真实参数

这些参数用于生成真实动力学，但不进入 student input：

```text
mass_true
Ix_true / Iy_true / Iz_true
cg_x_true / cg_z_true
payload / load distribution（载荷分布）
chassis compliance / flex proxy（车身柔度 proxy）
roll stiffness / damping（roll 刚度 / 阻尼）
pitch stiffness / damping（pitch 刚度 / 阻尼）
suspension nonlinear coefficients（悬架非线性系数）
suspension geometry / roll center / anti-dive / anti-squat（悬架几何 / 侧倾中心 / 抗点头 / 抗下蹲）
tire stiffness / Pacejka coefficients（轮胎刚度 / Pacejka 系数）
load sensitivity（载荷敏感性）
camber / toe compliance（外倾 / 前束柔度）
tire pressure / temperature / wear state（胎压 / 胎温 / 磨损状态）
ambient / road temperature profile（环境 / 路面温度 profile）
aero drag / lift / downforce coefficients（气动阻力 / 升力 / 下压力系数）
rolling resistance coefficients（滚动阻力系数）
actuator delay / saturation（执行器延迟 / 饱和）
steering time constant / compliance / backlash / hysteresis（转向时间常数 / 柔度 / 间隙 / 滞回）
ABS/TCS/ESC thresholds and modulation gains（阈值与调制增益）
brake hydraulic pressure dynamics / fade（制动液压动态 / 热衰退）
engine / motor / gearbox maps（发动机 / 电机 / 变速箱 map）
driveline compliance / torque vectoring params（传动系柔度 / 扭矩矢量参数）
sensor noise / bias / delay（传感器噪声 / 偏置 / 延迟）
sensor mounting offset / timestamp jitter / dropout（传感器安装偏移 / 时间戳抖动 / 丢样）
road μ map
road grade / bank / roughness / height map（坡度 / 横坡 / 粗糙度 / 高程 map）
wind / crosswind profile（风 / 侧风 profile）
```

随机化策略：

```text
同一 vehicle_family 内改变 hidden params，模拟同车型不同车辆 / 不同状态
不同 vehicle_family 改变 geometry / layout，模拟新车型
held-out vehicle/config 保留完整 family 或隐藏参数组合，用于泛化测试
B2 sim-to-real proxy 使用系统性偏移而不是纯随机噪声
full-fidelity teacher_feature_flags 全部默认开启，关闭项必须进入 metadata
```

## 7. 导出字段

`DATA_DESIGN.md` 是 student-visible schema 的权威来源；本章定义 teacher 需要生成的字段角色。各模块章节中列出的输出是 diagnostics 候选项，最终是否导出、导出到 observable 还是 teacher-only，以本章和 `DATA_DESIGN.md` 为准。

### 7.1 Observable

observable 字段以 `DATA_DESIGN.md` 为准。teacher 负责生成这些可见量：

```text
vx, vy
roll, pitch, yaw
p, q, r
omega_i
sw_angle
tau_drv_obs_i
tau_brk_obs_i
steer_cmd, throttle_cmd, brake_cmd
fixed_vehicle_context
nominal_physics_prior
```

observable 是 sensor model 输出，不是 teacher 真值。`*_obs_i` 必须已经包含传感器 noise、delay、filtering、quantization、dropout 等观测效应；若需要导出真实无噪状态，必须放入 teacher-only 或 diagnostics，不可混入 student input。

### 7.2 Teacher-Only

teacher 必须导出：

```text
Fz_true_i
Fx_true_i
Fy_true_i
μ_true_i
slip_ratio_true_i
slip_angle_true_i
Mz_true_i
friction_usage_i
delta_eff_i
suspension states（悬架状态）
unsprung states（非簧载状态）
camber_true_i / toe_true_i
actuator delay states（执行器延迟状态）
tau_drv_true_i / tau_brk_true_i
brake_pressure_i
ABS/TCS/ESC modulation states
brake_temperature_i
road surface labels（路面标签）
road_height_true_i / road_normal_true_i
grade_true / bank_true
sensor true states（传感器对应真值状态）
sensor bias / delay / quantization / dropout / timestamp jitter states（传感器偏置 / 延迟 / 量化 / 丢样 / 时间戳抖动状态）
aero force / moment diagnostics（气动力 / 气动力矩 diagnostics）
真实 mass / inertia / cg / suspension / tire 参数或 hash
vehicle_internal_params_hash
teacher_feature_flags
vehicle_internal_params_hash_algo
torque_observability_mode
```

使用规则：

```text
允许用于 auxiliary loss
允许用于 diagnostics
允许用于 oracle upper bound
允许用于绘图 / debugging
禁止进入 student input
```

## 8. 数值积分与采样

推荐实现：

```text
内部积分 dt: full-fidelity profile 下 1 ms
导出 dt: 20 ms，对应 50 Hz
主积分器: partitioned semi-implicit integration
刚性事件 sub-step dt: <= 0.25 ms，用于 tire relaxation / suspension stop / ABS-TCS-ESC events
RK4 用途: 仅用于 debug profile 或 cross-check，不作为默认 full-fidelity generator
episode 时长: DS0 为 10-30 s，DS1 为 10-60 s
事件处理: road transition、ABS/TCS modulation、actuator saturation
```

要求：

```text
导出时间戳严格单调
所有 observable / teacher_only 字段同一时间基准
传感器 delay 必须在 metadata 中记录
如果内部 dt 与导出 dt 不同，必须明确 downsample / filter 方式
低速 slip、接触力、悬架限位和 actuator saturation 必须做数值保护
所有随机过程必须由 episode seed 可复现
当 slip、Fz、tire temperature、brake pressure 或 suspension travel 单步变化超过阈值时，必须触发 sub-step 或 episode retry
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
  quantization: ...
  dropout: ...
  timestamp_jitter: ...
  mounting_offset: ...

environment_profile:
  wind: ...
  road_grade_bank: ...
  roughness: ...

teacher_feature_flags:
  profile: full-fidelity
  enabled_features:
    - six_dof_chassis
    - nonlinear_suspension
    - transient_combined_slip_tire
    - tire_thermal_wear_pressure
    - road_mu_transition
    - grade_bank_roughness
    - aero_wind_crosswind
    - steering_compliance_backlash_hysteresis
    - drive_brake_actuator_dynamics
    - abs_tcs_esc_modulation
    - brake_thermal_fade
    - sensor_noise_delay_quantization_dropout
  disabled_features: []
  downgrade_reason: null
  generator_version: ...

split_metadata:
  episode_id: ...
  scenario_id: ...
  split_role: ...
  target_window_id: ...
  torque_observability_mode: true_per_wheel_sensor | actuator_estimate | command_only_projection
  vehicle_internal_params_hash_algo: sha256
  vehicle_internal_params_hash: ...
```

该接口必须能覆盖 `DATA_DESIGN.md` 中 `CG-SINGLE / CG-SPLIT / CG-TRANSITION` 的 DS0/DS1 工况组合，以及 DS2 extreme handling 扩展。

术语来源：

```text
CG-SINGLE / CG-SPLIT / CG-TRANSITION: 定义在 DATA_DESIGN.md 的工况组
FTD0-FTD5: 定义在 DATA_DESIGN.md 的 fine-tune 数据量 bucket
MoE tire residual: 定义在 MODULE_DESIGN.md 和 EXPERIMENT_PLAN.md 的 student 侧扩展模块
```

`vehicle_internal_params_hash` 定义：

```text
算法: SHA-256
输入: canonical JSON
覆盖内容:
  teacher_hidden_params（teacher 隐藏参数）
  teacher_feature_flags（teacher feature 开关）
  vehicle_config hidden ids（车辆配置隐藏 id）
  actuator_profile（执行器 profile）
  sensor_profile（传感器 profile）
  environment_profile（环境 profile）
  road_profile hidden maps / seeds（路面隐藏 map / seed）
要求:
  同一 episode seed 和同一 hidden config 必须得到相同 hash
  hash 只用于追踪和 split audit，不进入 student input
```

## 10. Sanity 与验收标准

### 10.1 DS0 生成前的单元测试

必须通过：

```text
零输入静态测试 zero-input static test:
  平地静止，速度和角速度不应发散

直线匀速测试 straight constant speed:
  yaw rate 接近 0，左右 Fz 对称

制动单调性 braking monotonicity:
  brake torque 增大时减速度增大，前轴 Fz 增加

转弯载荷转移 cornering load transfer:
  左转时右侧 Fz 增加，yaw rate 方向正确

split-μ 制动 split-μ braking:
  左右 Fx 不对称，产生合理 yaw moment

附着极限 friction limit:
  sqrt(Fx_i^2 + Fy_i^2) / (μ_i Fz_i) 大多数时间不超过合理范围

低速 slip 保护 low-speed slip guard:
  起步/停车时 slip_ratio_true_i 和 tire force 不出现数值尖峰

轮胎 relaxation 测试 tire relaxation:
  step steering / step braking 后 Fx/Fy 有合理一阶滞后，不应瞬时跳到稳态

坡度 / 横坡 road grade / bank:
  坡道重力分量、横坡左右载荷分配方向正确

粗糙路 / 悬架 rough road / suspension:
  bump/pothole 输入下 unsprung/suspension states 有界且 Fz 连续

ABS/TCS 调制 ABS/TCS modulation:
  大制动/大驱动低 μ 下 wheel slip 被调制，不应出现非物理无限抱死或空转

传感器时序 / 扰动 sensor timing / corruption:
  delay / timestamp jitter / quantization / dropout 后 observable 时间戳仍单调，metadata 可追踪
  constant bias / drift bias 不应破坏单位、符号和量纲范围检查

轮胎热 / 磨损 / 胎压 tire thermal / wear / pressure:
  长时间大 slip 后 tire_temperature_i 上升，μ / stiffness 随温度和磨损变化方向合理
  胎压变化对 effective rolling radius、vertical stiffness 和 tire force 的影响方向正确

aero / wind / crosswind 气动 / 风 / 侧风:
  drag 与速度方向相反且随速度近似二次增长
  downforce / lift 对 Fz 的影响方向正确
  crosswind 产生合理 lateral force 和 yaw moment

转向柔度 / 间隙 / 滞回 steering compliance / backlash / hysteresis:
  step steer 后 delta_eff_i 有 delay / rate limit
  小幅往返转向能体现 backlash / hysteresis，且不产生非物理跳变

制动热 / 热衰退 brake thermal / fade:
  连续大制动后 brake_temperature_i 上升
  fade 开启后同等制动命令下有效制动力不应异常增大

ESC yaw 干预 ESC yaw intervention:
  过度 yaw rate / sideslip 场景下 ESC yaw moment 方向应抵抗失稳趋势

camber / toe compliance（外倾 / 前束柔度）:
  roll、load 和 lateral force 变化时 camber_true_i / toe_true_i 方向连续且符合约定

路面 μ 过渡 road μ transition:
  dry/wet/snow/ice transition 中 μ_true_i 连续或按指定 transition_type 平滑变化
  transition 边界不应引入 Fz/Fx/Fy 数值尖峰

schema validator（schema 校验）:
  observable 字段不得包含 teacher-only 真值
  teacher-only 字段不得被 dataloader 暴露给 student input
  torque observable 必须带 torque_observability_mode
```

### 10.2 DS0 Debug Dataset 验收

必须通过：

```text
所有 required scenarios 都可稳定生成
无 NaN / Inf
dt consistency 检查通过
units 和 sign tests 检查通过
observable / teacher_only schema 隔离通过
schema validator / leakage test 通过
physics-only student smoke rollout 不立即发散
tiny black-box 和 tiny hybrid model 能 overfit 5-10 条短序列
full-fidelity teacher_feature_flags 全部开启
```

### 10.3 DS1 V1 Research Dataset 验收

必须通过：

```text
多 vehicle/config family 可生成
held-out vehicle/config metadata 完整
held-out road μ metadata 完整
held-out target time window metadata 完整
FTD0-FTD5 数据量 bucket 可复现生成
vehicle_internal_params_hash 可追踪但不进入 student input
teacher_feature_flags 可追踪，正式 DS1 不混入未标注的降级 profile
torque_observability_mode 完整且与 observable torque 字段一致
full-fidelity profile 下所有 scenario 可稳定生成
```

## 11. 实现阶段

### TSim-0：Kinematic / Schema Prototype（运动学 / Schema 原型）

目标：

```text
跑通 scenario interface
导出完整 schema
验证坐标系、单位、wheel_order、metadata
```

可以先使用简化 tire / load transfer，只用于 pipeline smoke test。

### TSim-1：Full-Fidelity Dynamic Teacher（高保真动力学 Teacher）

目标：

```text
6-DOF 车体
四轮转动动力学
非簧载质量 + 非线性 suspension / load transfer
包含 thermal / wear / pressure effects 的 transient combined-slip tire
road μ map + grade + bank + roughness
aero + rolling resistance + wind（气动 + 滚动阻力 + 风）
包含 ABS/TCS/ESC-like modulation 的 steering / drive / brake actuator
sensor noise / bias / filtering / delay / quantization / dropout（传感器噪声 / 偏置 / 滤波 / 延迟 / 量化 / 丢样）
DS0 全部通过
```

这是 B0 的最低可接受版本。

### TSim-2：DS1 Research Generator（DS1 研究数据生成器）

目标：

```text
多车/多配置生成
hidden parameter randomization（隐藏参数随机化）
held-out family/config split（held-out 车型族 / 配置划分）
FTD0-FTD5 target windows（目标时间窗）
B2 sim-to-real proxy perturbation（B2 sim-to-real proxy 扰动）
full-fidelity feature flags 和 hidden parameter hashes
```

### TSim-3：DS2 Extreme / Realism Extension（极限工况 / 真实性扩展）

目标：

```text
large slip / fishhook / emergency maneuvers（大滑移 / 鱼钩 / 紧急操控）
rough road / bump / pothole stress（粗糙路 / 凸起 / 坑洼压力测试）
更多 vehicle layouts 和 aero profiles
更强的 sensor / actuator artifact sweeps
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
- [ ] Scenario interface（工况接口）定义并可生成 episode metadata。
- [ ] 6-DOF chassis dynamics（车身动力学）可运行。
- [ ] 四轮转动动力学可运行。
- [ ] 非簧载质量、悬架几何、dynamic load transfer 可运行。
- [ ] transient combined-slip tire、low-speed slip guard、热 / 磨损 / 胎压效应可运行。
- [ ] road μ map 支持 uniform / Split-μ / transition / patchy / grade / bank / roughness。
- [ ] aero、rolling resistance、wind disturbance 可运行。
- [ ] steering / drive / brake actuator delay、saturation、ABS/TCS/ESC-like modulation 可运行。
- [ ] sensor noise / bias / filtering / delay / quantization / dropout / timestamp jitter 可运行。
- [ ] observable / teacher_only 字段导出并 schema 隔离。
- [ ] teacher_feature_flags 默认全开启并写入 metadata。
- [ ] DS0 sanity 全部通过。
- [ ] DS1 最小多车/多配置数据可生成。
