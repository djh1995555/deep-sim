# 侧倾/俯仰动力学与四轮载荷转移定向调研

**日期**：2026-05-18  
**对应决策**：采用“第二版”载荷转移方案，即显式建模 `roll, pitch, p, q` 动态，并由其修正四轮法向载荷 `Fz_i`。  
**目标**：为 hybrid vehicle dynamics 模型确定侧倾、俯仰、载荷转移、法向载荷估计与 residual learning 的工程结构。

## 1. 结论先行

之前的文献 landscape 覆盖了 hybrid dynamics、轮胎模型、TRFC 和 tire force learning，但对 **roll/pitch dynamics + four-wheel normal load** 覆盖不够细。定向调研后，建议把侧倾/俯仰和载荷转移放进车辆物理 backbone，而不是交给黑盒网络。

推荐结构：

```text
observable history
  -> history encoder
  -> ΔFz_residual / suspension latent / uncertainty

current state + input
  -> roll/pitch/load-transfer physics
  -> Fz_fl, Fz_fr, Fz_rl, Fz_rr
  -> tire physics + tire residual
  -> vehicle dynamics + small vehicle residual
```

第二版不要只用准静态公式，而应至少包含：

```text
Ixx * roll_ddot  = M_roll  - c_roll  * roll_dot  - k_roll  * roll
Iyy * pitch_ddot = M_pitch - c_pitch * pitch_dot - k_pitch * pitch
```

再由 `roll, pitch, p, q, ax, ay` 计算或修正四轮 `Fz_i`。

## 2. 为什么载荷转移必须显式建模

轮胎可产生的最大力由法向载荷和路面附着共同决定：

```text
sqrt(Fx_i^2 + Fy_i^2) <= μ_i * Fz_i
```

如果 `Fz_i` 不准，会直接导致：

- 低附着路面下轮胎力上限错误；
- Split-μ 时左右 yaw moment 错误；
- 加速/制动时前后轴能力分配错误；
- 转向时左右轮饱和顺序错误；
- rollout 中 `vy, r, roll, pitch` 漂移。

所以 `Fz_i` 是 tire physics 和 vehicle physics 之间的关键耦合变量。

## 3. 文献脉络

### 3.1 经典车辆动力学与载荷转移

经典车辆动力学教材和模型普遍将载荷转移拆成静态载荷、纵向载荷转移、横向载荷转移、roll/pitch 动态贡献。代表来源包括：

- Pacejka, *Tire and Vehicle Dynamics*：轮胎力、Magic Formula、载荷敏感性；
- Rajamani, *Vehicle Dynamics and Control*：bicycle/double-track 模型、横摆/侧偏、状态估计；
- Gillespie, *Fundamentals of Vehicle Dynamics*：载荷转移、悬架、车身姿态基础；
- Milliken & Milliken, *Race Car Vehicle Dynamics*：左右载荷转移、防倾杆、轮胎载荷敏感性。

这些不是深度学习文献，但对工程模型更重要：它们定义了 `Fz_i` 的物理边界。

### 3.2 Double-track / four-wheel vehicle model

为了支持 Split-μ 和四轮不同路面，模型不能只停留在 bicycle model。bicycle model 可用于低成本 baseline，但正式结构应采用 double-track：

```text
每个轮胎独立计算：
  slip ratio κ_i
  slip angle α_i
  normal load Fz_i
  longitudinal force Fx_i
  lateral force Fy_i

整车合力/力矩：
  ΣFx, ΣFy, Mz, Mroll, Mpitch
```

double-track 的优势：

- 支持 `μ_fl, μ_fr, μ_rl, μ_rr`；
- 支持左右轮载荷差；
- 支持左右轮制动/驱动扭矩差；
- 能表达 Split-μ 诱导的 yaw moment；
- 能把 roll/pitch 状态和 tire saturation 连接起来。

### 3.3 Tire normal force estimation

法向载荷估计文献通常走三种路线：

1. **基于车辆参数和加速度的解析估计**  
   使用 `m, L, lf, lr, h_cg, track_width, ax, ay` 等参数，计算前后/左右转移。

2. **基于 roll/pitch/suspension 的观测器**  
   使用 Kalman filter、UKF、sliding mode observer 等估计 `Fz_i` 或相关悬架状态。

3. **基于数据驱动或混合模型的估计**  
   用 ANN、fuzzy、Gaussian process、Bayesian NN 等补偿非线性悬架、轮胎垂向刚度、道路扰动和未建模效应。

对本项目，路线 1 + 2 应作为主结构，路线 3 用作 residual。

### 3.4 学习模型中的 `Fz` 使用方式

深度学习 vehicle dynamics 文献中，很多工作没有显式建模四轮 `Fz_i`，而是直接让网络学状态转移。这对常规自动驾驶或固定路面 MPC 可能够用，但对本项目不够，因为：

- 低附着和 Split-μ 的核心是每个轮胎力上限变化；
- 没有 `Fz_i` 就很难施加摩擦椭圆约束；
- NN 会把载荷转移、μ、轮胎非线性混在 latent 中，解释性和泛化性都差。

因此，更合适的是让学习模型输出：

```text
ΔFz_i, suspension_latent, c_roll correction, k_roll correction,
c_pitch correction, k_pitch correction, uncertainty
```

而不是直接输出全部 `Fz_i`。

## 4. 第二版推荐模型

### 4.1 状态

使用 requirement 中已有状态：

```text
x = [
  vx, vy,
  roll, pitch, yaw,
  p, q, r,
  omega_fl, omega_fr, omega_rl, omega_rr
]
```

控制/输入：

```text
u = [
  sw_angle,
  tau_drv_fl/fr/rl/rr,
  tau_brk_fl/fr/rl/rr,
  steer_cmd, throttle_cmd, brake_cmd
]
```

### 4.2 物理层：roll/pitch dynamics

以低自由度车身姿态模型作为主结构：

```text
roll_dot = p
pitch_dot = q

p_dot = (M_roll - c_roll * p - k_roll * roll) / Ixx
q_dot = (M_pitch - c_pitch * q - k_pitch * pitch) / Iyy
```

其中：

```text
M_roll  ≈ m * ay * h_roll + M_roll_tire + M_roll_residual
M_pitch ≈ -m * ax * h_pitch + M_pitch_tire + M_pitch_residual
```

`M_roll_tire` 和 `M_pitch_tire` 可由四轮力和几何位置计算。第一版实现时可以先用 `m*ay*h` 和 `m*ax*h` 主导，后续再加入更完整的轮胎力矩项。

### 4.3 物理层：四轮法向载荷

推荐将 `Fz_i` 分为四部分：

```text
Fz_i =
  Fz_static_i
  + ΔFz_long_i(ax, pitch, q)
  + ΔFz_lat_i(ay, roll, p)
  + ΔFz_residual_i(history)
```

其中：

```text
Fz_static_front = m * g * lr / L
Fz_static_rear  = m * g * lf / L

前后转移：
ΔFz_long_total ≈ m * ax * h_cg / L

左右转移：
ΔFz_lat_front ≈ m * ay * h_cg * lr / (L * track_front)
ΔFz_lat_rear  ≈ m * ay * h_cg * lf / (L * track_rear)
```

roll/pitch 动态修正：

```text
ΔFz_roll_dynamic  ∝ k_roll * roll + c_roll * p
ΔFz_pitch_dynamic ∝ k_pitch * pitch + c_pitch * q
```

更完整时，可引入前后轴 roll stiffness distribution：

```text
Kphi_front_ratio = k_roll_front / (k_roll_front + k_roll_rear)
Kphi_rear_ratio  = 1 - Kphi_front_ratio
```

这样横向载荷转移可在前后轴分配，而不是只按轴距几何分配。

### 4.4 神经网络 residual 的位置

推荐 residual 不直接替代 `Fz_i`，而是补偿物理层：

```text
h_t = Encoder(x_{t-K:t}, u_{t-K:t})

ΔFz_raw = FzResidualNN(h_t, ax, ay, roll, pitch, p, q)
ΔFz = bounded(ΔFz_raw)

Fz_i = project_positive_and_mass_consistent(
  Fz_phys_i + ΔFz_i
)
```

约束建议：

```text
Fz_i >= Fz_min
sum_i Fz_i ≈ m * g - m * az
|ΔFz_i| <= ρ * Fz_static_i
左右转移方向与 ay 基本一致
前后转移方向与 ax 基本一致
```

如果没有 `az`，可以先使用：

```text
sum_i Fz_i ≈ m * g
```

### 4.5 与轮胎层连接

`Fz_i` 进入 tire physics：

```text
Fx_i, Fy_i =
  TirePhysics(κ_i, α_i, Fz_i, μ_i, θ_tire_i)
  + TireResidualNN(...)
```

摩擦约束：

```text
sqrt(Fx_i^2 + Fy_i^2) <= μ_i * Fz_i
```

或者使用更柔和的 friction ellipse：

```text
(Fx_i / (μ_i Fz_i))^2 + (Fy_i / (μ_i Fz_i))^2 <= 1
```

这能自然处理低附着、Split-μ 和四轮载荷差。

## 5. 需要补齐或辨识的车辆参数

第二版模型至少需要：

```text
m                 车辆质量
Ixx, Iyy, Izz      转动惯量
lf, lr             质心到前/后轴距离
L                  轴距
track_front/rear   前后轮距
h_cg               质心高度
h_roll, h_pitch    等效侧倾/俯仰力臂
k_roll, c_roll     等效侧倾刚度/阻尼
k_pitch, c_pitch   等效俯仰刚度/阻尼
```

如果部分参数未知，推荐不要全交给 NN，而是作为可学习物理参数：

```text
θ_vehicle = learnable constrained parameters
```

例如：

```text
k_roll = softplus(raw_k_roll) + ε
c_roll = softplus(raw_c_roll) + ε
```

这样模型仍保持物理意义。

## 6. 工程实现建议

### 6.1 第一阶段实现优先级

1. 实现 double-track kinematics 和四轮 slip / slip angle；
2. 实现 roll/pitch dynamics；
3. 实现 `Fz_phys_i`；
4. 加 `FzResidualNN`，但先把幅值限制到较小范围；
5. 接 tire physics + tire residual；
6. 接 vehicle residual；
7. 用 multi-step rollout loss 训练。

### 6.2 Loss 设计

主任务：

```text
L_rollout = Σ_t ||x_pred_t - x_gt_t||
```

物理约束：

```text
L_friction = violation(friction_ellipse)
L_fz_pos = violation(Fz_i >= 0)
L_fz_sum = |ΣFz_i - m*g|
L_residual = ||ΔFz|| + ||ΔF_tire|| + ||Δx_vehicle||
L_smooth = ||z_t - z_{t-1}||
```

姿态相关：

```text
L_roll_pitch = ||roll, pitch, p, q prediction error||
```

低附着/Split-μ 场景应单独加权，避免被常规稳态数据淹没。

## 7. 相关文献与资料清单

这一批用于支撑第二版结构，后续可继续精读：

| # | 文献/资料 | 类型 | 用途 |
|---|---|---|---|
| 1 | Pacejka, *Tire and Vehicle Dynamics* | 经典教材 | tire force、load sensitivity、Magic Formula |
| 2 | Rajamani, *Vehicle Dynamics and Control* | 经典教材 | vehicle dynamics、state estimation、control-oriented models |
| 3 | Gillespie, *Fundamentals of Vehicle Dynamics* | 经典教材 | load transfer、roll/pitch、suspension basics |
| 4 | Milliken & Milliken, *Race Car Vehicle Dynamics* | 经典教材 | roll stiffness distribution、lateral load transfer |
| 5 | *Deep Dynamics: Vehicle Dynamics Modeling with a Physics-Informed Neural Network for Autonomous Racing* | 学习动力学 | physics-informed dynamics backbone |
| 6 | *Learning Vehicle Dynamics Using Neural Network Model for Autonomous Vehicle Control* | 学习动力学 | NN dynamics baseline |
| 7 | *Learning-Based Vehicle Dynamics Modeling for Autonomous Racing Using Gaussian Processes* | 学习动力学 | GP residual 和不确定性 |
| 8 | *Deep Kernel Learning for Vehicle Dynamics Identification* | 学习动力学 | dynamics uncertainty / nonlinear ID |
| 9 | *Tire-Road Friction Coefficient Estimation: Review and Research Perspectives* | TRFC 综述 | μ 估计与可辨识性 |
| 10 | *Tire Normal Force Estimation Using Artificial Neural Networks and Fuzzy Logic* | Fz 学习估计 | 法向载荷数据驱动补偿 |
| 11 | *Estimation of Wheel-Ground Contact Normal Forces of a Vehicle from Experimental Data Validation* | Fz 估计 | 实验数据验证法向载荷估计 |
| 12 | *A Hierarchical Estimator for Vehicle States and Tire Forces* | 状态/力估计 | 层级估计结构参考 |
| 13 | *Joint Estimation of Vehicle States and Tire-Road Friction Using UKF* | 状态+μ估计 | latent state observer 思路 |
| 14 | *Road Friction Estimation for Four-Wheel Independent Drive Electric Vehicles* | 四轮 μ | Split-μ / wheel-level friction |
| 15 | *Split-μ Road Identification and Vehicle Stability Control* | Split-μ | 左右附着差导致的 yaw moment |
| 16 | *Neural Network Tire Force Modeling for Automated Drifting* | tire force learning | 大侧偏和非线性轮胎区 |
| 17 | *Auxiliary-Enhanced Neural Network for Tire Lateral Force Estimation* | tire force learning | tire residual / auxiliary features |
| 18 | *Bayesian Neural Network Based Intelligent Tire Force Estimation* | 不确定性 | tire force uncertainty |
| 19 | *Physics-Constrained Recurrent Neural Networks for Dynamical Systems* | constrained sequence model | 长 rollout 约束 |
| 20 | *Neural Ordinary Differential Equations for Learning Dynamical Systems* | continuous dynamics | 可微分积分与连续时间建模 |

## 8. 对 idea generation 的约束

下一阶段生成方案时，应固定以下设计前提：

1. 使用 double-track 而不是单纯 bicycle 作为正式 backbone；
2. `roll, pitch, p, q` 进入状态更新，不只作为普通 NN 输入；
3. `Fz_i` 是显式中间变量；
4. `FzResidualNN` 只做有界小修正；
5. tire residual 是主 residual，vehicle residual 是小幅末端修正；
6. `μ_i` 与 `Fz_i` 共同约束 tire force；
7. 所有内部仿真变量只能作为辅助监督，不能作为部署输入。

