# 闭环车辆仿真器使用指南

新的仿真器主包是 `simulator/`。它把闭环运行、控制器、车辆模型、数据生成和可视化拆成四块：

| 路径 | 用途 |
| --- | --- |
| `simulator/simulator_app.py` | 闭环仿真主入口，负责调用 controller 和 vehicle_model。 |
| `simulator/controller/` | 控制器接口与默认实现，当前提供纵向 PID + 横向 LQR。 |
| `simulator/reference/` | 参考速度/轨迹模块，向 controller 提供每个时刻要跟踪的 `ControllerReference`。 |
| `simulator/vehicle_model/` | 车辆、轮胎、悬架、路面、执行器、传感器模型。 |
| `simulator/data_generator/` | 批量数据生成入口。 |
| `simulator/visualizer/` | controller / vehicle_model 通信变量和 debug 信号的 trace 输出与可视化接口。 |

所有命令都使用 `deep-sim` conda 环境：

```bash
conda run -n deep-sim python -m simulator.cli --request configs/simulator/demo_single.yaml
```

## 1. 闭环仿真输出

默认输出到 request 中的 `out_dir`，未指定时输出到 `output/simulation/SIM001_<scenario_id>/`。

```text
out_dir/
  manifest.json
  episodes/
    <episode_id>.npz
    <episode_id>.json
  simulation_request.json
  simulation_summary.json
  debug_trace.json
  debug_trace.csv
  debug_report.html
```

`manifest.json` 和 `episodes/` 仍是 canonical dataset 格式，可以直接被现有 validator、data loader 和后续训练/评估代码读取。

## 2. 直接运行闭环仿真

```bash
conda run -n deep-sim python -m simulator.cli \
  --dataset-config configs/datasets/ds0_minimal.yaml \
  --scenario-id manual_closed_loop_split \
  --dataset-id SIM_MANUAL_CLOSED_LOOP_SPLIT \
  --road split:dry:ice \
  --vehicle-index 1 \
  --initial-speed-mps 10.0 \
  --target-speed-mps 14.0 \
  --target-y-m 0.5 \
  --duration-s 6.0 \
  --dt-export 0.02 \
  --out-dir output/simulation/SIM001_manual_closed_loop_split
```

闭环数据流是：

```text
ReferenceProvider.query(t, state)
  -> ControllerReference(target_speed / target path)
  -> ControllerInput(state / observation / reference)
  -> Controller.compute()
  -> ControllerOutput(sw_angle / throttle_cmd / brake_cmd / debug)
  -> VehicleModel.step()
  -> next state + observation
```

## 3. Reference 配置

旧的平铺字段仍然可用：`target_speed_mps`、`target_y_m`、`target_yaw_rad`、`target_yaw_rate_rps`。不写 `reference` 时，仿真器会把这些字段转换成固定参考。

动态 reference 推荐写在 request YAML 里：

```yaml
reference:
  type: lane_change
  speed_mps: 14.0
  start_y_m: 0.0
  end_y_m: 0.5
  start_x_m: 5.0
  length_m: 25.0
  mode: spatial
```

当前支持三类 provider：

| `type` | 用途 | 关键字段 |
| --- | --- | --- |
| `fixed` | 固定目标速度 / 横向位置 / 航向角 | `speed_mps` 或 `target_speed_mps`、`target_y_m`、`target_yaw_rad` |
| `lane_change` | 平滑换道参考，可按车辆 `x_world` 或时间推进 | `speed_mps`、`start_y_m`、`end_y_m`、`start_x_m`、`length_m`、`mode` |
| `double_lane_change` | 平滑双移线参考：先横移到 `start_y_m + offset_y_m`，保持，再回到 `end_y_m` | `speed_mps`、`offset_y_m`、`first_length_m`、`hold_length_m`、`second_length_m` |
| `waypoints` | 多 waypoint 路径跟踪，按最近路径投影加 lookahead 生成目标点 | `lookahead_m`、`points: [[x, y, speed], ...]` |

CLI 也可以直接传 JSON：

```bash
conda run -n deep-sim python -m simulator.cli \
  --road split:dry:ice \
  --reference-json '{"type":"waypoints","lookahead_m":8.0,"points":[[0,0,10],[30,0,12],[60,3.5,12]]}' \
  --out-dir /tmp/deep_sim_waypoint_demo
```

`debug_trace.json/csv` 会记录每一步的 `target_x_m`、`target_speed_mps`、`target_y_m`、`target_yaw_rad`、`path_s_m` 等参考信号；episode metadata 中会保存 `controller.reference_provider`，方便追溯本次仿真使用了哪种 reference。

## 4. Plotly HTML Debug Report

默认 `write_debug_html: true` 时，闭环仿真会额外输出 `debug_report.html`。这个报告参考 `/home/mi/debug/scripts/report_generator` 的方式，用 Plotly 生成可交互 HTML，并将 Plotly JS 内嵌到单文件中，方便直接用浏览器打开。

默认面板包括：

| 面板 | 主要信号 |
| --- | --- |
| `Trajectory` | `vehicle.x_world/y_world` 与 `input.target_x_m/target_y_m` |
| `Speed Tracking` | 车速、目标速度、纵向 PID 误差、加速度命令 |
| `Lateral Tracking` | 横向位置、目标横向位置、航向角、横向/航向误差 |
| `Commands` | 转向、油门、制动命令 |
| `Vehicle State` | `vy`、`r`、最小路面 μ、最大 friction usage |
| `Reference` | reference path / lookahead / curvature 相关信号 |

可以在 request YAML 中关闭 HTML：

```yaml
write_debug_html: false
```

也可以自定义 HTML 面板信号：

```yaml
debug_html_signals:
  Speed:
    - input.vx
    - input.target_speed_mps
    - output.debug.longitudinal.speed_error
  Steering:
    - output.sw_angle
    - output.debug.lateral.raw_sw_angle
```

已有 `debug_trace.json` 时，也可以单独重新生成 HTML：

```bash
conda run -n deep-sim python -m simulator.visualizer.report \
  --trace output/simulation/SIM001_manual_closed_loop_split/debug_trace.json \
  --out output/simulation/SIM001_manual_closed_loop_split/debug_report.html
```

如果要像 `report_generator` 的 target signal 配置一样自定义面板，可以传 `--panels panels.yaml`。格式与 `debug_html_signals` 相同。

## 5. 支持的路面写法

```text
dry
single:dry
split:dry:ice
transition:dry:wet
dry->wet
```

支持的 surface：`dry`、`wet`、`snow`、`ice`。

## 6. 批量生成训练数据

批量数据生成走 `simulator.data_generator`：

```bash
conda run -n deep-sim python -m simulator.data_generator.generate \
  --config configs/datasets/ds0_minimal.yaml \
  --out output/training/SIM_DATA_ds0_debug
```

## 7. 验证输出

```bash
conda run -n deep-sim python - <<'PY'
from simulator.vehicle_model.validators import TeacherEpisodeValidator
report = TeacherEpisodeValidator().validate_dataset("output/simulation/SIM001_manual_closed_loop_split")
print(report.to_dict())
PY
```
