# P1-P4 精读结论：物理 backbone、roll/pitch 与四轮载荷转移

完整内容见：`idea-stage/PHYSICS_BACKBONE_READING_20260518_192812.md`

## 已冻结建议

- 正式模型使用 double-track，不用 bicycle。
- 使用 5-DOF body + 4-wheel rotational dynamics。
- `roll/pitch` 是二阶物理状态，不只作为 NN feature。
- `Fz_i` 显式计算，并进入 tire model。
- 加入 roll stiffness distribution，但参数可学习。
- `FzResidualNN` 输出小幅、质量守恒投影后的 residual。
- `TireResidualNN` 优先在 force-level 工作，并施加 friction ellipse projection。
- `VehicleResidualNN` 限制幅值，不能吞掉物理层。

## 核心公式

```text
m * (vx_dot - r * vy) = ΣFx_body
m * (vy_dot + r * vx) = ΣFy_body
Izz * r_dot = ΣMz

roll_dot = p
pitch_dot = q
p_dot = (M_roll - c_roll*p - k_roll*roll) / Ixx
q_dot = (M_pitch - c_pitch*q - k_pitch*pitch) / Iyy

Fz_i =
  Fz_static_i
  + ΔFz_long_i
  + ΔFz_lat_i
  + ΔFz_roll_pitch_i
  + ΔFz_residual_i
```

