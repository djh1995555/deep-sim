# P1-P4 精读结论：物理 backbone、roll/pitch 与四轮载荷转移

**日期**：2026-05-18  
**精读范围**：Pacejka, Rajamani, Gillespie, Milliken & Milliken  
**目标**：冻结当前 hybrid dynamics 的物理主链路，特别是 double-track、roll/pitch dynamics、四轮 `Fz_i` 和 tire force 约束。

## 1. 本轮冻结结论

当前方案应采用：

```text
double-track 5-DOF body + 4-wheel rotational dynamics
+ roll/pitch second-order dynamics
+ explicit four-wheel normal loads Fz_i
+ tire model with load sensitivity and friction ellipse
+ bounded Fz residual / tire force residual
```

不要使用单纯 bicycle model 作为正式模型。bicycle 可作为 baseline，但无法充分表达：

- Split-μ 左右附着差；
- 四轮驱动/制动扭矩差；
- 左右载荷转移；
- 四轮轮速和 slip ratio；
- 四轮 `μ_i * Fz_i` 对 yaw moment 的影响。

## 2. 推荐状态与自由度

车辆物理模型采用：

```text
车身 5 DOF:
  longitudinal: vx
  lateral:      vy
  yaw:          yaw, r
  roll:         roll, p
  pitch:        pitch, q

四轮转动 4 DOF:
  omega_fl, omega_fr, omega_rl, omega_rr
```

状态向量：

```text
x = [
  vx, vy,
  yaw, r,
  roll, p,
  pitch, q,
  omega_fl, omega_fr, omega_rl, omega_rr
]
```

如果训练输出按 requirement 保持原顺序：

```text
[vx, vy, roll, pitch, yaw, p, q, r, omega_fl, omega_fr, omega_rl, omega_rr]
```

内部动力学实现可以使用更方便的顺序，但输出接口保持一致。

## 3. Double-track 主方程

车体坐标系下：

```text
m * (vx_dot - r * vy) = ΣFx_body
m * (vy_dot + r * vx) = ΣFy_body
Izz * r_dot = ΣMz
```

由四轮力计算：

```text
ΣFx_body = Σ transform_wheel_to_body(Fx_i, Fy_i, delta_i).x
ΣFy_body = Σ transform_wheel_to_body(Fx_i, Fy_i, delta_i).y

ΣMz =
  lf * (Fy_fl_body + Fy_fr_body)
  - lr * (Fy_rl_body + Fy_rr_body)
  + track_f/2 * (Fx_fr_body - Fx_fl_body)
  + track_r/2 * (Fx_rr_body - Fx_rl_body)
```

前轮转角：

```text
delta_fl, delta_fr = steering_geometry(delta_eff)
delta_eff = NN_steer(history, sw_angle, steer_cmd)
```

第一版可先令：

```text
delta_fl = delta_fr = delta_eff
```

后续再加入 Ackermann 或左右转角差。

## 4. Roll / pitch dynamics

保留二阶姿态动力学：

```text
roll_dot = p
pitch_dot = q

p_dot = (M_roll  - c_roll  * p - k_roll  * roll)  / Ixx
q_dot = (M_pitch - c_pitch * q - k_pitch * pitch) / Iyy
```

其中主激励可先采用：

```text
M_roll  = m * ay * h_roll
M_pitch = -m * ax * h_pitch
```

更完整版本再加入由四轮力和悬架几何产生的力矩：

```text
M_roll_tire  = function(Fy_i, Fz_i, roll_center, track)
M_pitch_tire = function(Fx_i, Fz_i, pitch_center, wheelbase)
```

工程建议：

- `roll, pitch, p, q` 不只是 NN 输入，而是物理状态；
- `k_roll, c_roll, k_pitch, c_pitch` 设为可学习但有正值约束的物理参数；
- 不要让 vehicle residual 直接覆盖 roll/pitch 主动力学。

## 5. 四轮法向载荷 `Fz_i`

推荐分解：

```text
Fz_i =
  Fz_static_i
  + ΔFz_long_i
  + ΔFz_lat_i
  + ΔFz_roll_pitch_i
  + ΔFz_residual_i
```

### 5.1 静态载荷

```text
Fz_front_static_total = m * g * lr / L
Fz_rear_static_total  = m * g * lf / L

Fz_fl_static = Fz_fr_static = 0.5 * Fz_front_static_total
Fz_rl_static = Fz_rr_static = 0.5 * Fz_rear_static_total
```

### 5.2 纵向载荷转移

```text
ΔFz_long_total = m * ax * h_cg / L
```

符号约定建议：

- 加速 `ax > 0`：后轴增加，前轴减少；
- 制动 `ax < 0`：前轴增加，后轴减少。

所以：

```text
front_each += -0.5 * ΔFz_long_total
rear_each  +=  0.5 * ΔFz_long_total
```

### 5.3 横向载荷转移

最小可用版本：

```text
ΔFz_lat_front = m * ay * h_cg * lr / (L * track_f)
ΔFz_lat_rear  = m * ay * h_cg * lf / (L * track_r)
```

更推荐版本：加入 roll stiffness distribution：

```text
λ_f = k_roll_front / (k_roll_front + k_roll_rear)
λ_r = 1 - λ_f

ΔFz_lat_total = m * ay * h_cg
ΔFz_lat_front = λ_f * ΔFz_lat_total / track_f
ΔFz_lat_rear  = λ_r * ΔFz_lat_total / track_r
```

侧向加速度符号需要与车体坐标统一。实现时应通过单元测试固定：

```text
ay > 0 时，外侧轮 Fz 增加，内侧轮 Fz 减少
```

### 5.4 Roll/pitch 动态修正

把姿态弹簧阻尼项转成四轮载荷修正：

```text
M_roll_susp  = k_roll * roll + c_roll * p
M_pitch_susp = k_pitch * pitch + c_pitch * q
```

然后按前后轴和左右轮分配：

```text
ΔFz_roll_front = λ_f * M_roll_susp / track_f
ΔFz_roll_rear  = λ_r * M_roll_susp / track_r

ΔFz_pitch_front =  M_pitch_susp / L
ΔFz_pitch_rear  = -M_pitch_susp / L
```

这一步的符号必须和 roll/pitch 坐标定义一致。建议实现后用静态场景校验：

- 左转/右转时外侧轮载荷增加；
- 制动时前轴载荷增加；
- 加速时后轴载荷增加。

### 5.5 `FzResidualNN`

NN 只输出小幅修正：

```text
ΔFz_residual = FzResidualNN(history, ax, ay, roll, pitch, p, q)
```

约束：

```text
|ΔFz_residual_i| <= ρ * Fz_static_i
ρ 初始建议 0.05 - 0.15
Fz_i >= Fz_min
ΣFz_i ≈ m * g
```

如果没有垂向加速度 `az`，先使用 `ΣFz_i = m*g` 投影；如果后续能获得 `az`，改为 `ΣFz_i = m*(g - az)`。

## 6. Tire model 与载荷敏感性

从 Pacejka / 经典轮胎模型得到的核心约束：

```text
Fx_i, Fy_i depends on:
  slip ratio κ_i
  slip angle α_i
  normal load Fz_i
  friction μ_i
  tire parameters θ_i
```

摩擦椭圆约束：

```text
(Fx_i / (μ_i * Fz_i))^2 + (Fy_i / (μ_i * Fz_i))^2 <= 1
```

或带形状参数：

```text
(|Fx_i| / (μ_i * Fz_i))^a + (|Fy_i| / (μ_i * Fz_i))^b <= 1
```

工程决策：

- 第一版 tire physics 可用 Fiala / Dugoff / simplified brush，而不是直接上完整 Pacejka；
- Pacejka 可作为高保真仿真器或 teacher，不一定作为可训练模型 backbone；
- `TireResidualNN` 建议输出 `ΔFx, ΔFy`，但必须经过 friction ellipse projection；
- 另一种更稳路线是输出 tire parameter residual，例如 `ΔC_alpha, ΔC_kappa, Δμ_scale`。

## 7. 与神经网络模块的边界

当前 hybrid 模型边界应固定为：

```text
NN_mu:
  输出 wheel-level μ latent 或 μ distribution

NN_steer:
  输出 delta_eff / steering lag / steering gain

FzResidualNN:
  输出小幅 ΔFz_i

TireResidualNN:
  输出 ΔFx_i, ΔFy_i 或 tire parameter residual

VehicleResidualNN:
  输出小幅 Δx 或 Δacc，用于补低自由度车辆模型误差
```

残差优先级：

```text
TireResidualNN > FzResidualNN > VehicleResidualNN
```

含义：主要误差应在力层解释，车辆状态 residual 只做末端小修正，避免退化成黑盒动力学。

## 8. 推荐实现公式接口

建议代码接口按层拆：

```python
def compute_slip(x, u, delta_eff, params):
    return kappa_i, alpha_i

def compute_load_transfer(x, ax, ay, params, fz_residual=None):
    return Fz_i

def tire_forces(kappa_i, alpha_i, Fz_i, mu_i, tire_params, tire_residual=None):
    return Fx_i, Fy_i

def vehicle_dynamics(x, Fx_i, Fy_i, delta_i, params):
    return x_dot

def integrate(x, x_dot, dt):
    return x_next
```

关键是 `Fz_i` 和 tire force 不要藏在 monolithic network 里。

## 9. 需要的数据/参数

必须参数：

```text
m, g
lf, lr, L
track_f, track_r
h_cg
Ixx, Iyy, Izz
k_roll, c_roll
k_pitch, c_pitch
wheel_radius_i
wheel_inertia_i
```

建议参数：

```text
k_roll_front, k_roll_rear
roll_center_height_front/rear
pitch_center / anti-dive / anti-squat equivalent terms
tire cornering stiffness C_alpha_i
tire longitudinal stiffness C_kappa_i
nominal μ per road class
```

未知参数处理：

```text
positive parameters: softplus(raw) + eps
bounded ratios: sigmoid(raw)
wheel-level scale: 1 + small_tanh(raw)
```

## 10. 需要立即冻结的工程决策

本轮精读后建议冻结：

1. 正式模型使用 double-track，不用 bicycle。
2. 使用 5-DOF body + 4-wheel rotational dynamics。
3. `roll/pitch` 是二阶物理状态，不只作为 NN feature。
4. `Fz_i` 显式计算，并进入 tire model。
5. 加入 roll stiffness distribution，但参数可学习。
6. `FzResidualNN` 输出小幅、质量守恒投影后的 residual。
7. `TireResidualNN` 优先在 force-level 工作，并施加 friction ellipse projection。
8. `VehicleResidualNN` 限制幅值，不能吞掉物理层。

## 11. 本轮未解决问题

下一批 P9-P12 需要解决：

- `μ_i` 是单点、分布，还是 road class + continuous scale；
- 低激励时如何表达不可辨识性；
- Split-μ 数据集如何设计；
- 路面衔接处 `μ_i(t)` 的 latent dynamics 如何约束。

下一批 P13-P17 需要解决：

- tire residual 输出 force residual 还是 tire parameter residual；
- `FzResidualNN` 是否需要辅助监督；
- 高自由度仿真器中的 tire force / normal load 是否用于 teacher loss。

## 12. 参考来源

- Hans B. Pacejka, *Tire and Vehicle Dynamics*, Elsevier / Butterworth-Heinemann.
- Rajesh Rajamani, *Vehicle Dynamics and Control*, Springer.
- Thomas D. Gillespie, *Fundamentals of Vehicle Dynamics*, SAE International.
- William F. Milliken, Douglas L. Milliken, *Race Car Vehicle Dynamics*, SAE International.

