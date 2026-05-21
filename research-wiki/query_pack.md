# Query Pack

## Project Direction

工程目标：训练用于仿真系统的高保真车辆动力学模型，第一阶段只使用真实车可观测状态与控制输入，覆盖干燥、湿滑、雨雪、低附着、Split-μ、路面衔接和长时 rollout。创新不是主目标，效果和可落地性优先。

## Current Design Anchor

采用 hybrid dynamics：

```text
double-track 5-DOF body + 4-wheel rotational dynamics
+ roll/pitch dynamics
+ explicit Fz_fl/fr/rl/rr
+ wheel-level latent μ
+ tire physics + tire residual
+ small vehicle residual
+ uncertainty-aware rollout
```

车辆物理模型：车身 5 自由度（纵向、横向、横摆、侧倾、俯仰）+ 四轮转动 4 自由度。输出状态包括 `vx, vy, roll, pitch, yaw, p, q, r, omega_fl/fr/rl/rr`。

## Top Gaps

- G1：低激励常规工况下 μ 可辨识性不足，应输出带不确定性的等效路面 latent。
- G2：Split-μ 需要四轮级或左右级 latent friction。
- G3：路面衔接需要时序 latent transition，而不是逐帧分类。
- G4：小侧偏线性区与大侧偏非线性区需要分区建模。
- G5：高自由度仿真到真实车存在 sim-to-real gap，内部变量只能作为辅助监督。
- G6：长时 rollout 稳定性比 one-step loss 更关键。
- G7：侧倾/俯仰动态与四轮法向载荷必须显式进入 hybrid dynamics。

## Deep Reading Priority

1. 物理 backbone：Pacejka, Rajamani, Gillespie, Milliken。
2. Hybrid / learned dynamics：Deep Dynamics PINN, GP vehicle dynamics, Deep Kernel Learning, long-term NN dynamics。
3. 低附着与 Split-μ：TRFC review, four-wheel friction estimation, Split-μ identification, spatio-temporal friction estimation。
4. 轮胎力与法向载荷：NN tire force for drifting, auxiliary tire lateral force estimation, BNN tire force, tire normal force estimation。
5. 不确定性和序列动力学：physics-constrained RNN, Neural ODE, Deep Ensembles。

## Design Constraints for Ideation

- 正式 backbone 用 double-track，不用单纯 bicycle。
- `roll, pitch, p, q` 进入车辆物理状态更新。
- `Fz_i` 是显式中间变量。
- `FzResidualNN` 只做有界小修正。
- tire residual 是主 residual，vehicle residual 是小幅末端修正。
- `μ_i` 与 `Fz_i` 共同约束 tire force。
- 内部仿真变量只作为辅助监督，不能作为部署输入。

## Frozen Physical Backbone Decisions

- 状态：`[vx, vy, yaw, r, roll, p, pitch, q, omega_fl, omega_fr, omega_rl, omega_rr]`，输出接口保持 requirement 中的状态顺序。
- 主方程：使用 body-frame double-track dynamics：`m(vx_dot-r*vy)=ΣFx`，`m(vy_dot+r*vx)=ΣFy`，`Izz*r_dot=ΣMz`。
- 姿态：`roll_dot=p`，`pitch_dot=q`，`p_dot=(M_roll-c_roll*p-k_roll*roll)/Ixx`，`q_dot=(M_pitch-c_pitch*q-k_pitch*pitch)/Iyy`。
- 载荷：`Fz_i = Fz_static + ΔFz_long + ΔFz_lat + ΔFz_roll_pitch + ΔFz_residual`。
- 横向载荷转移：加入 front/rear roll stiffness distribution，`λ_f=k_roll_front/(k_roll_front+k_roll_rear)`。
- 约束：`Fz_i>=Fz_min`，`ΣFz_i≈m*g`，`(Fx/(μFz))^2+(Fy/(μFz))^2<=1`。
- residual 边界：`FzResidualNN` 做 5%-15% 静态载荷量级的小修正；`VehicleResidualNN` 不允许吞掉物理层。

## Frozen Learning Decisions

- `μ_i`：四轮级输出，形式为 `μ_mean_i, μ_logvar_i`，低激励时提高不确定性。
- `TireResidualNN`：优先输出 `ΔFx_i, ΔFy_i`，输入包含 `κ_i, α_i, Fz_i, μ_i, vx, h_t, wheel_id`，输出后做 friction ellipse projection。
- `FzResidualNN`：只输出小幅 `ΔFz_i`，并做 positivity + total-load projection；仿真器 true `Fz_i` 可作为 teacher loss。
- `VehicleResidualNN`：输出小幅 acceleration/state increment residual，强正则，不能替代主动力学。
- `history encoder`：第一版优先 TCN/GRU；Transformer 等数据量足够后再评估。
- `uncertainty`：第一版用 K=3/5 deep ensemble + heteroscedastic Gaussian head。
- `training`：multi-step rollout loss 是主目标，one-step loss 是辅助。

## Active Ideas

- idea:001 RECOMMENDED — Three-group hybrid dynamics experiment. All groups keep `MuHead` for wheel-level `μ_mean/logvar`. Group 1: physical tire model only + single global residual covering tire and vehicle residuals. Group 2: add force-level tire residual `ΔFx/ΔFy` with friction ellipse projection. Group 3: add parameter-level tire residual `ΔC_alpha/ΔC_kappa/Δμ_scale`.
- idea:002 RECOMMENDED ABLATION — Parameter-Residual Tire Model. Group 3: output `ΔC_alpha/ΔC_kappa/Δμ_scale` and compare against Group 2 force residual.
- idea:003 RECOMMENDED MODULE — Bounded `FzResidualNN` with simulator teacher supervision. 验证 `Fz_true_i` teacher loss 是否改善低附着/Split-μ。
- idea:004 RECOMMENDED AFTER SINGLE MODEL — Deep ensemble uncertainty wrapper. 单模型稳定后做 K=3/5 ensemble。
- idea:005 SECOND PHASE — Slip-regime MoE tire residual. 等极限操控数据加入后再做。

## Required Ablations Added 2026-05-18

- VehicleResidualNN placement:
  - V0 no residual
  - V1 derivative/global residual before integration: `x_next = Integrator(x_t, x_dot_phys + Δx_dot)`; default recommendation
  - V2 state residual after integration: `x_next = Integrator(x_t, x_dot_phys) + Δx`
  - V3 both residuals; not recommended first
- SteeringHead:
  - S2 first-order steering lag; Rank 1 default
  - S3 S2 + bounded NN steering residual; ablation
  - S0 fixed steering ratio and S1 learnable ratio + bias are optional simple reference baselines, not required before Rank 1.

## Failed / Downgraded Ideas

- idea:006 ELIMINATED AS MAINLINE — End-to-end black-box dynamics. 仅保留为 baseline，不能作为生产主模型；baseline 包含 TCN/GRU/MLP autoregressive predictor 和 N-BEATSx direct multi-horizon predictor。
- Direct full `Fz_i` NN output — rejected; use bounded residual + projection.
- Global deterministic μ — rejected; use wheel-level distribution.
