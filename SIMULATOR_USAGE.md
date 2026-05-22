# Simulator 目录结构与使用方法

本文档说明 `simulator/` 的代码组织、闭环仿真运行方式、训练数据生成方式、输出文件含义，以及后续替换控制器或扩展可视化信号时应该改哪里。

所有命令默认在项目根目录执行：

```bash
cd /home/mi/vibe/research/deep_sim/codex
```

所有 Python 命令都使用 `deep-sim` 环境：

```bash
conda run -n deep-sim <command>
```

## 1. 目录结构

```text
simulator/
  __init__.py
  cli.py
  simulator_app.py
  controller/
    __init__.py
    base.py
    composite.py
    longitudinal_pid.py
    lateral_lqr.py
  reference/
    __init__.py
    base.py
    factory.py
    fixed.py
    lane_change.py
    double_lane_change.py
    sinusoidal.py
    waypoints.py
  vehicle_model/
    __init__.py
    config.py
    model.py
    state.py
    scenario.py
    vehicle_params.py
    export.py
    validators.py
    validate.py
    modules/
      __init__.py
      aero.py
      drive_brake.py
      road.py
      sensors.py
      steering.py
      suspension.py
      tire.py
  data_generator/
    __init__.py
    generate.py
  visualizer/
    __init__.py
    debug_trace.py
    plotly_report.py
    report.py
```

各文件作用如下：

| 路径                                               | 作用                                                                                        |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| `simulator/__init__.py`                          | 暴露 simulator 包的主要入口，方便外部直接 import `run_closed_loop_simulation` 等 API。                     |
| `simulator/cli.py`                               | 闭环仿真的命令行入口，读取 request YAML 或 CLI 参数，合并覆盖项后调用 `simulator_app`。                             |
| `simulator/simulator_app.py`                     | 闭环仿真主流程：构建 scenario、reference、controller、vehicle model，并导出结果。                             |
| `simulator/controller/__init__.py`               | 汇总并导出 controller 子包的接口和默认控制器实现。                                                           |
| `simulator/controller/base.py`                   | 定义控制器通用数据结构：`ControllerReference`、`ControllerInput`、`ControllerOutput` 和 `Controller` 接口。 |
| `simulator/controller/composite.py`              | 组合纵向 PID 与横向 LQR，形成当前默认的闭环控制器 `PIDLQRController`。                                         |
| `simulator/controller/longitudinal_pid.py`       | 纵向 PID 控制器，根据目标速度和当前速度输出 throttle / brake 命令。                                             |
| `simulator/controller/lateral_lqr.py`            | 简化横向 LQR 控制器，根据横向误差、航向误差等输出方向盘角。                                                          |
| `simulator/reference/__init__.py`                | 汇总并导出 reference provider 构建入口和各类 reference 实现。                                            |
| `simulator/reference/base.py`                    | 定义 reference provider 抽象接口，以及 reference 配置的公共工具。                                          |
| `simulator/reference/factory.py`                 | 根据 YAML / dict 中的 `type` 字段创建 fixed、lane change、double lane change、sinusoidal 或 waypoint reference。  |
| `simulator/reference/fixed.py`                   | 固定速度 / 固定横向位置 reference，用于直线跟踪和最小 smoke。                                                  |
| `simulator/reference/lane_change.py`             | 单移线 reference，按时间或空间生成从起点横向移动到目标车道的轨迹。                                                    |
| `simulator/reference/double_lane_change.py`      | 双移线 reference，生成横移、保持、回正三段轨迹。                                                             |
| `simulator/reference/sinusoidal.py`              | 正弦路线 reference，按空间周期生成目标横向位置、航向、曲率和 yaw rate。                                             |
| `simulator/reference/waypoints.py`               | waypoint reference，读取离散轨迹点并插值出目标位置、航向、曲率和速度。                                              |
| `simulator/vehicle_model/__init__.py`            | 汇总并导出 vehicle model 子包的主要配置、模型和场景类型。                                                      |
| `simulator/vehicle_model/config.py`              | 读取和解析车辆模型 / 数据生成 YAML，构建 `TeacherSimConfig`。                                              |
| `simulator/vehicle_model/model.py`               | 高保真 teacher vehicle model 主体，负责初始化状态、推进动力学、生成 observation 和 aux labels。                   |
| `simulator/vehicle_model/state.py`               | 定义车辆状态数据结构和状态向量转换工具。                                                                      |
| `simulator/vehicle_model/scenario.py`            | 定义 road、scenario、sensor、actuator、environment 等场景配置，并生成工况矩阵。                               |
| `simulator/vehicle_model/vehicle_params.py`      | 生成不同车辆参数变体，包括轴距、轮距、质量、惯量、轮胎参数等。                                                           |
| `simulator/vehicle_model/export.py`              | 将 episode 导出为 canonical dataset 格式，包括 manifest、npz 数组和 sidecar JSON。                      |
| `simulator/vehicle_model/validators.py`          | 校验 episode / dataset 的字段完整性、时间戳、维度、数值范围和 metadata。                                        |
| `simulator/vehicle_model/validate.py`            | dataset 校验命令行入口，用于单独检查生成的数据集。                                                             |
| `simulator/vehicle_model/modules/__init__.py`    | vehicle model 子模块包标记。                                                                     |
| `simulator/vehicle_model/modules/aero.py`        | 空气动力学和风影响模型，计算阻力和风相关扰动。                                                                   |
| `simulator/vehicle_model/modules/drive_brake.py` | 驱动 / 制动执行器模型，包含命令滞后、饱和和 ABS 风格限制。                                                         |
| `simulator/vehicle_model/modules/road.py`        | 基于车辆 / 车轮坐标查询路面 μ、坡度、横坡和 roughness。                                                       |
| `simulator/vehicle_model/modules/sensors.py`     | 传感器模型，加入噪声、延迟、dropout、量化和 timestamp jitter。                                               |
| `simulator/vehicle_model/modules/steering.py`    | 转向执行器模型，包含滞后、饱和、柔度、backlash 和 hysteresis。                                                 |
| `simulator/vehicle_model/modules/suspension.py`  | 悬架、侧倾 / 俯仰和载荷转移模型。                                                                        |
| `simulator/vehicle_model/modules/tire.py`        | 轮胎模型，计算 slip、轮胎力、friction usage、热衰退和磨耗相关效应。                                               |
| `simulator/data_generator/__init__.py`           | 数据生成子包入口。                                                                                 |
| `simulator/data_generator/generate.py`           | 批量数据集生成入口，调用 vehicle model 按配置生成多工况、多车辆 canonical dataset。                                |
| `simulator/visualizer/__init__.py`               | 汇总并导出 debug trace 和可视化报告 API。                                                             |
| `simulator/visualizer/debug_trace.py`            | 记录 controller input/output 和 vehicle debug 信号，并导出 JSON / CSV / HTML。                      |
| `simulator/visualizer/plotly_report.py`          | Plotly HTML 报告生成逻辑，包括 trajectory 面板和 timeseries 面板。                                       |
| `simulator/visualizer/report.py`                 | 从已有 `debug_trace.json` 重新生成 Plotly HTML 的命令行入口。                                           |

相关配置目录：

| 路径 | 作用 |
| --- | --- |
| `configs/simulator/` | 单次闭环仿真 request 配置。 |
| `configs/datasets/` | 批量数据集生成配置。 |
| `output/simulation/` | 闭环仿真输出目录。 |
| `output/training/` | 训练实验输出目录。 |
| `data/` | 可复用数据集目录。训练数据下载或长期保留数据应放这里。 |

## 2. 当前工作逻辑

闭环仿真的数据流如下：

```text
request YAML / CLI args
  -> ClosedLoopSimulationRequest
  -> SimulatorApp
  -> build_scenario()
  -> build_reference_provider()
  -> default PIDLQRController
  -> VehicleModel.initialize()
  -> loop over t:
       ReferenceProvider.query(t, state)
       ControllerInput(state, observation, reference)
       Controller.compute()
       VehicleModel.step(command)
       DebugTrace.append(...)
  -> export canonical episode
  -> write debug_trace + HTML report
```

控制闭环中的核心数据对象：

| 对象 | 文件 | 内容 |
| --- | --- | --- |
| `ControllerReference` | `controller/base.py` | controller 每一步接收到的完整 reference 点，包括位置、速度、航向、yaw rate、曲率、路径进度和 lookahead。 |
| `ControllerInput` | `controller/base.py` | 当前仿真时间、车辆状态、scenario、reference、上一帧 observation。 |
| `ControllerOutput` | `controller/base.py` | `sw_angle`、`steer_cmd`、`throttle_cmd`、`brake_cmd` 和 debug 字段。 |
| `VehicleStepResult` | `vehicle_model/model.py` | 单步车辆动力学输出，包括 state、observation、轮胎、载荷、转向、路面接触等中间量。 |
| `DebugTrace` | `visualizer/debug_trace.py` | 保存 controller input/output 与 vehicle debug 信号，用于 CSV/JSON/HTML 可视化。 |

`ControllerReference` 与 waypoint 使用同一组 reference 点信息：

```text
x_m, y_m, z_m
speed_mps
yaw_rad
yaw_rate_rps
curvature_1pm
path_s_m
lookahead_distance_m
extra
```

`ControllerReference` 不再保留 `target_*` 兼容字段。debug trace 中 reference 统一输出到 `input.reference.*`，例如 `input.reference.x_m`、`input.reference.y_m`、`input.reference.speed_mps`。

## 3. 快速运行闭环仿真

推荐先运行已有双移线配置：

```bash
conda run -n deep-sim python -m simulator.cli --request configs/simulator/double_lane_change_5mps.yaml
```

配置中的 `out_dir` 提供输出父路径和 run 名称。默认情况下，仿真器会把时间戳插入到父路径和 run 名称之间，避免覆盖上一次结果：

```text
output/simulation/20260522_145312_123456/double_lane_change_5mps/
```

也可以运行较短的 demo：

```bash
conda run -n deep-sim python -m simulator.cli \
  --request configs/simulator/demo_single.yaml
```

## 4. 不写 YAML，直接用 CLI 运行

可以直接通过命令行覆盖主要参数：

```bash
conda run -n deep-sim python -m simulator.cli \
  --scenario-id manual_closed_loop_split \
  --road split:dry:ice \
  --vehicle-index 1 \
  --initial-speed-mps 10.0 \
  --reference-file configs/simulator/references/lane_change_14mps.yaml \
  --duration-s 6.0 \
  --dt-export 0.02 \
  --out-dir output/simulation/manual_closed_loop_split
```

常用 CLI 参数：

| 参数 | 含义 |
| --- | --- |
| `--request` | 读取闭环仿真 YAML。 |
| `--out-dir` | 输出目录模板。默认会把时间戳插入到父路径和目录名之间，例如 `output/simulation/demo` 会输出到 `output/simulation/<timestamp>/demo/`。未指定时模板为 `output/simulation/<scenario.id>/`。 |
| `--scenario-id` | 本次仿真的 scenario 名称。 |
| `--road` | 路面配置，例如 `single:dry`、`split:dry:ice`。 |
| `--vehicle-index` | 车辆参数变体编号。 |
| `--seed` | 本次 scenario 的随机种子，影响传感器噪声、扰动和 episode ID。 |
| `--initial-speed-mps` | 初始车速。 |
| `--duration-s` | 仿真时长，等价于覆盖 YAML 中的 `model.duration_s`。 |
| `--dt-internal` | 内部积分步长，等价于覆盖 YAML 中的 `model.dt_internal`。 |
| `--dt-export` | 导出采样步长，等价于覆盖 YAML 中的 `model.dt_export`。 |
| `--reference-file` | YAML reference 配置文件。推荐使用。 |
| `--reference-json` | 用 JSON 传入 reference 配置。闭环仿真必须定义 reference。 |
| `--pid-kp/--pid-ki/--pid-kd` | 纵向 PID 参数。 |
| `--lqr-gains` | 横向 LQR 四个增益。 |
| `--lqr-max-sw-angle-rad` | 横向 LQR 输出方向盘角饱和上限。 |
| `--write-debug-trace/--no-write-debug-trace` | 是否输出 debug trace。 |
| `--write-debug-html/--no-write-debug-html` | 是否输出 Plotly HTML。 |
| `--timestamped-output/--no-timestamped-output` | 是否把时间戳插入到 `out_dir` 的父路径和目录名之间。默认开启。 |

CLI 仍保留 `--duration-s`、`--dt-internal`、`--dt-export` 作为临时覆盖参数；它们不会成为顶层 request 字段，而是运行时写入 `request.model`。CLI 不再支持 `--dataset-id`、`--model-config`、`--dataset-config`、`--target-speed-mps`、`--target-y-m`、`--target-yaw-rad`、`--target-yaw-rate-rps`。

## 5. Request YAML 写法

推荐用 `configs/simulator/*.yaml` 管理单次仿真。request 的核心结构是：

- `model`：车辆模型配置。闭环仿真 YAML 通常只显式写数值积分、物理常数和数值保护字段；代码层面兼容 `TeacherSimConfig` 的 metadata / feature flag / export 字段。
- `scenario`：一个完整仿真场景，包含 road、reference、车辆初始状态和车辆变体。
- controller/debug/output：控制器、调试输出和输出目录。

一个典型 request：

```yaml
model:
  dt_internal: 0.01
  dt_export: 0.02
  duration_s: 32.0
  integrator: semi_implicit_euler
  gravity: 9.81
  coordinate_convention: body_x_forward_y_left_z_up
  numerical_limits:
    min_speed_for_slip: 0.5
    min_fz: 50.0
    max_abs_state:
      vx: 80.0
      vy: 30.0
      roll: 1.0
      pitch: 1.0
      r: 3.0
out_dir: output/simulation/double_lane_change_5mps
scenario:
  id: single_dry_vehicle0_double_lane_change_5mps
  road:
    type: single
    surface: dry
    grade_rad: 0.0
    bank_rad: 0.0
    roughness_amp_m: 0.003
  vehicle_index: 0
  seed: 21
  initial_state:
    x_m: 0.0
    y_m: 0.0
    z_m: 0.0
    yaw_rad: 0.0
    speed_mps: 5.0
  reference: configs/simulator/references/double_lane_change_5mps.yaml
pid_kp: 0.18
pid_ki: 0.025
pid_kd: 0.02
lqr_gains: [0.20, 1.60, 0.35, 1.80]
lqr_max_sw_angle_rad: 1.2
debug_stride: 4
write_debug_trace: true
write_debug_html: true
timestamped_output: true
```

`scenario.id` 不是 reference 名称，而是 road、vehicle、initial state、reference 组合后的场景 ID。比如 `single_dry_vehicle0_double_lane_change_5mps` 表示：单一 dry 路面、vehicle 0、初始 5 m/s、跟踪 double lane change reference。

`model` 会传给 `simulator.vehicle_model.config.config_from_dict()`，因此当前代码接受和 `TeacherSimConfig` 兼容的字段。`configs/simulator/*.yaml` 里的长期示例只显式写会影响闭环仿真的数值积分、物理常数和数值保护字段，未写字段使用 `default_simulator_model_config()` / `TeacherSimConfig` 默认值。闭环仿真导出的 `dataset_id` 不读取 `model.dataset_id`，而是固定由 `scenario.id` 生成 `SIM_<scenario.id>`；`schema_version` 和 `teacher_model_version` 会进入导出 manifest。`scenario.seed` 是场景随机种子，会影响仿真噪声和扰动。

### 最小可运行 request

只需要跑一个最小闭环仿真时，可以从下面开始：

```yaml
model:
  dt_internal: 0.01
  dt_export: 0.02
  duration_s: 2.0
out_dir: output/simulation/minimal_closed_loop
scenario:
  id: minimal_single_dry_fixed_10mps
  road: single:dry
  initial_state:
    x_m: 0.0
    y_m: 0.0
    yaw_rad: 0.0
    speed_mps: 10.0
  reference:
    type: fixed
    speed_mps: 10.0
    y_m: 0.0
    yaw_rad: 0.0
write_debug_html: false
```

未写出的 `model` 子字段会使用 `simulator_app.default_simulator_model_config()` / `TeacherSimConfig` 中的默认值。团队内长期保留的 request 建议像 `configs/simulator/*.yaml` 一样显式写出仿真相关的 `model` 字段，便于 review 和复现实验；metadata / export 字段仅在需要覆盖 manifest 版本号或 feature flags 时再写。

### 字段边界

simulator YAML 的顶层 request 字段只描述闭环仿真本身：车辆、路面、reference、初始状态、控制器、噪声/执行器/风、数值积分和 debug 输出。`model` 内部为了复用 `TeacherSimConfig`，仍兼容少量 metadata、feature flag 和 export 字段。
字段含义：

| 字段                            | 类型          | 含义                                                                                                                            | 示例 / 说明                                                      |
| ----------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `model`                       | mapping     | 底层车辆模型配置。包含仿真步长、时长、积分器、重力、坐标约定、数值保护，并兼容 `TeacherSimConfig` 的 metadata / feature flag / export 字段。                                                 | 见下方 `model` 字段表。                                             |
| `out_dir`                     | string      | 输出目录模板。默认会把时间戳插入到父路径和目录名之间保存本次结果。                                                                                                 | `output/simulation/double_lane_change_5mps`                  |
| `scenario`                    | mapping     | 场景集合，包含 road、reference、车辆变体、随机种子和车辆初始状态。                                                                                    | 见下方 `scenario` 字段表。                                          |
| `sensor_noise_scale`          | float       | 传感器噪声整体缩放。                                                                                                                    | `0.0` 可近似关闭噪声；`1.0` 为默认。                                     |
| `timestamp_jitter_std_s`      | float       | 导出 timestamp 抖动标准差，单位秒。                                                                                                       | 用于模拟真实采样时间抖动。                                                |
| `sensor_dropout_probability`  | float       | 传感器 dropout 概率。                                                                                                               | `0.0` 表示不丢点。                                                 |
| `sensor_quantization`         | float       | 传感器量化步长。                                                                                                                      | 默认 `1e-5`。                                                   |
| `torque_observability_mode`   | string      | 导出扭矩观测的模式。                                                                                                                    | 当前常用 `actuator_estimate`。                                    |
| `steering_saturation_rad`     | float       | 转向执行器饱和上限，单位 rad。                                                                                                             | 限制有效转向角。                                                     |
| `actuator_sensor_delay_steps` | int         | 执行器/传感器延迟步数。                                                                                                                  | `1` 表示延迟一个内部 step。                                           |
| `wind_x_mps`                  | float       | 世界坐标 x 方向风速，单位 m/s。                                                                                                           | 影响空气动力学。                                                     |
| `wind_y_mps`                  | float       | 世界坐标 y 方向风速，单位 m/s。                                                                                                           | 默认示例有轻微横风。                                                   |
| `lqr_gains`                   | list[float] | 横向 LQR 简化控制器的 4 个增益。                                                                                                          | 当前顺序由 `lateral_lqr.py` 使用，建议整组调参。                            |
| `lqr_max_sw_angle_rad`        | float       | LQR 输出方向盘角饱和上限，单位 rad。                                                                                                        | 默认 `1.2`。                                                    |
| `pid_kp`                      | float       | 纵向 PID 比例增益。                                                                                                                  | 不写时默认 `0.18`。                                                |
| `pid_ki`                      | float       | 纵向 PID 积分增益。                                                                                                                  | 不写时默认 `0.025`。                                               |
| `pid_kd`                      | float       | 纵向 PID 微分增益。                                                                                                                  | 不写时默认 `0.02`。                                                |
| `debug_stride`                | int         | debug trace 采样步长。                                                                                                             | `4` 表示每 4 个内部 step 记录一次 debug。                               |
| `write_debug_trace`           | bool        | 是否输出 `debug_trace.json` 和 `debug_trace.csv`。                                                                                  | 推荐保持 `true`，方便排查。                                            |
| `write_debug_html`            | bool        | 是否输出 Plotly `debug_report.html`。                                                                                              | 批量跑很多仿真时可以设为 `false`。                                        |
| `debug_html_signals`          | mapping     | 自定义 HTML 报告面板和信号列表。                                                                                                           | 不写则使用默认面板。                                                   |
| `timestamped_output`          | bool        | 是否把时间戳插入到 `out_dir` 的父路径和目录名之间。                                                                                                    | 默认 `true`，防止重跑覆盖旧结果。                                         |

`model` 内部字段：

| `model` 字段                            | 含义                                        |
| ------------------------------------- | ----------------------------------------- |
| `dt_internal`                         | 车辆动力学内部积分步长，单位秒。越小越稳定但越慢。                 |
| `dt_export`                           | 导出采样间隔，单位秒。必须是 `dt_internal` 的整数倍。        |
| `duration_s`                          | 本次仿真的总时长，单位秒。                             |
| `integrator`                          | 数值积分器。当前 v0 使用 `semi_implicit_euler`。     |
| `gravity`                             | 重力加速度，单位 m/s^2。                           |
| `coordinate_convention`               | 坐标系约定。当前只支持 `body_x_forward_y_left_z_up`。 |
| `dataset_id`                          | `TeacherSimConfig` 兼容字段。批量数据生成会用它写 manifest；闭环仿真会忽略它并使用 `SIM_<scenario.id>`。 |
| `schema_version`                      | 导出 manifest 和 episode metadata 的 schema 版本。 |
| `teacher_model_version`               | 导出 manifest 和 episode metadata 的车辆模型版本。 |
| `seed`                                | `TeacherSimConfig` 默认随机种子。闭环仿真通常使用 `scenario.seed`。 |
| `default_feature_flags`               | 写入 episode metadata 的 teacher feature flags。 |
| `numerical_limits.min_speed_for_slip` | 计算轮胎 slip 时使用的最低速度保护。                     |
| `numerical_limits.min_fz`             | 垂向载荷下限保护，避免数值异常。                          |
| `numerical_limits.max_abs_state`      | 状态量安全边界，超过后会触发数值失败。                       |
| `export.include_teacher_aux_labels`   | `TeacherSimConfig` 兼容字段。当前 exporter 始终写 canonical episode 中已有的 aux arrays。 |
| `export.include_metadata`             | `TeacherSimConfig` 兼容字段。当前 exporter 始终写 sidecar metadata。 |
| `export.resample_policy`              | `TeacherSimConfig` 兼容字段。当前导出采样由 `dt_export / dt_internal` 的 hold-last 步长控制。 |

`scenario` 内部字段：

| `scenario` 字段 | 含义 |
| --- | --- |
| `id` | 场景 ID。建议同时体现 road、vehicle、初始状态和 reference，例如 `single_dry_vehicle0_double_lane_change_5mps`。 |
| `road` | 路面配置。可以是简写字符串，也可以是 mapping。推荐长期配置使用 mapping。 |
| `vehicle_index` | 车辆参数变体编号，用于从 `vehicle_params.py` 选择车辆。 |
| `seed` | 场景随机种子，影响传感器噪声、扰动和 episode ID。 |
| `initial_state` | 车辆初始状态，包含位置、姿态、速度等。 |
| `reference` | reference 配置。推荐写成 reference YAML 文件路径，也支持 inline mapping。 |

`scenario` 只接受上表字段。`scenario_id`、`dataset_id`、`split_role`、`target_speed_mps` 等旧字段即使写在 `scenario` 内部也会被拒绝。

`scenario.initial_state` 支持：

| 字段 | 含义 |
| --- | --- |
| `x_m` / `y_m` / `z_m` | 初始世界坐标位置。 |
| `yaw_rad` / `pitch_rad` / `roll_rad` | 初始姿态。 |
| `speed_mps` | 初始纵向速度。 |
| `vy_mps` / `vz_mps` | 初始横向/垂向速度。 |
| `yaw_rate_rps` / `pitch_rate_rps` / `roll_rate_rps` | 初始角速度。 |

`initial_state` 也只接受上表字段，不再支持 `x`、`y`、`yaw`、`vx`、`p/q/r` 等别名。

`scenario.road` 推荐 mapping 写法：

```yaml
road:
  type: split
  left: dry
  right: ice
  split_boundary_y_m: 0.0
  split_start_x_m: 0.0
  split_end_x_m: 120.0
  split_default_surface: dry
  grade_rad: 0.0
  bank_rad: 0.0
  roughness_amp_m: 0.003
```

也可以临时用简写：

```yaml
road: single:dry
road: split:dry:ice
road: transition:dry:wet
road: dry->wet
```

`scenario.reference` 推荐指向文件：

```yaml
scenario:
  reference: configs/simulator/references/double_lane_change_5mps.yaml
```

reference 文件可以是参数式轨迹：

```yaml
type: double_lane_change
speed_mps: 5.0
start_y_m: 0.0
offset_y_m: 3.5
end_y_m: 0.0
start_x_m: 5.0
first_length_m: 50.0
hold_length_m: 15.0
second_length_m: 50.0
mode: spatial
```

也可以是 waypoint 轨迹。每个点必须显式写出 `x_m`、`y_m`、`yaw_rad`、`curvature_1pm`、`speed_mps`；可选 `z_m`、`yaw_rate_rps`、`path_s_m`、`lookahead_distance_m`、`extra`。构建 provider 时会按几何路径重新计算每个 waypoint 的 `path_s_m`，并填入当前 `lookahead_m`。

```yaml
type: waypoints
lookahead_m: 8.0
points:
  - {x_m: 0.0, y_m: 0.0, yaw_rad: 0.0, curvature_1pm: 0.0, speed_mps: 5.0}
  - {x_m: 30.0, y_m: 0.0, yaw_rad: 0.0, curvature_1pm: 0.0, speed_mps: 5.0}
  - {x_m: 60.0, y_m: 3.5, yaw_rad: 0.1, curvature_1pm: 0.006, speed_mps: 5.0}
```

`reference` 内部字段由具体类型决定。双移线 reference 中：

| `reference` 字段 | 含义 |
| --- | --- |
| `type` | reference provider 类型。`double_lane_change` 表示双移线。 |
| `speed_mps` | reference 目标速度，单位 m/s。 |
| `start_y_m` | 双移线开始时的横向位置。 |
| `offset_y_m` | 第一段横移的偏移量。 |
| `end_y_m` | 第二段结束后的横向位置。 |
| `start_x_m` | 双移线开始的纵向位置。 |
| `first_length_m` | 第一段横移长度。 |
| `hold_length_m` | 中间保持段长度。 |
| `second_length_m` | 第二段回正长度。 |
| `mode` | reference 推进方式。`spatial` 表示按车辆 `x_world` 位置推进。 |

### sinusoidal

正弦路线 reference。`sin`、`sine` 和 `sinusoidal` 都会映射到同一个 provider：

```yaml
type: sinusoidal
speed_mps: 5.0
center_y_m: 0.0
amplitude_m: 5.0
period_m: 20.0
start_x_m: 0.0
phase_rad: 0.0
mode: spatial
```

其中目标路线为 `y = center_y_m + amplitude_m * sin(2π * (x - start_x_m) / period_m + phase_rad)`。当前实现只支持 `mode: spatial`。

`timestamped_output` 默认是 `true`。因此 `out_dir: output/simulation/double_lane_change_5mps` 的实际输出会变成：

```text
output/simulation/<timestamp>/double_lane_change_5mps/
```

如果确实需要写入固定目录，可以显式关闭：

```yaml
timestamped_output: false
```

或使用 CLI：

```bash
conda run -n deep-sim python -m simulator.cli \
  --request configs/simulator/demo_single.yaml \
  --no-timestamped-output
```

## 6. 路面配置

当前支持以下简写：

```text
dry
single:dry
split:dry:ice
transition:dry:wet
dry->wet
```

支持的 surface：

```text
dry, wet, snow, ice
```

说明：

| 写法 | 含义 |
| --- | --- |
| `dry` / `single:dry` | 所有车轮都在 dry 路面。 |
| `split:dry:ice` | split-μ 地图，左侧 dry，右侧 ice。车辆横向运动后，车轮会按世界坐标自动查询当前所在路面。 |
| `transition:dry:wet` / `dry->wet` | 从 dry 到 wet 的路面过渡。 |

split road 和 transition road 都由 `vehicle_model/modules/road.py` 计算。路面查询使用车轮世界坐标，因此车辆从 split 区域驶出或横向跨过边界后，μ 会自动变化。

## 7. Reference 配置

闭环仿真必须写 `scenario.reference`。推荐做法是让它指向 `configs/simulator/references/*.yaml` 中的 reference 文件。固定速度/固定横向位置也要通过 `type: fixed` 表达，不再支持顶层 `target_speed_mps`、`target_y_m`、`target_yaw_rad`、`target_yaw_rate_rps`。

### fixed

固定目标速度、横向位置、航向：

```yaml
type: fixed
speed_mps: 10.0
y_m: 0.0
yaw_rad: 0.0
yaw_rate_rps: 0.0
```

### lane_change

单次平滑换道：

```yaml
type: lane_change
speed_mps: 14.0
start_y_m: 0.0
end_y_m: 0.5
start_x_m: 5.0
length_m: 25.0
mode: spatial
```

### double_lane_change

双移线：先横移、保持、再回到目标横向位置。

```yaml
type: double_lane_change
speed_mps: 5.0
start_y_m: 0.0
offset_y_m: 3.5
end_y_m: 0.0
start_x_m: 5.0
first_length_m: 50.0
hold_length_m: 15.0
second_length_m: 50.0
mode: spatial
```

### waypoints

多 waypoint 路径跟踪。waypoint 与 `ControllerReference` 拥有同一组 reference 点信息；配置中必须给出 `x_m`、`y_m`、`yaw_rad`、`curvature_1pm`、`speed_mps`，可选 `z_m`、`yaw_rate_rps`、`path_s_m`、`lookahead_distance_m`、`extra`。

```yaml
type: waypoints
lookahead_m: 8.0
points:
  - {x_m: 0.0, y_m: 0.0, yaw_rad: 0.0, curvature_1pm: 0.0, speed_mps: 10.0}
  - {x_m: 30.0, y_m: 0.0, yaw_rad: 0.0, curvature_1pm: 0.0, speed_mps: 12.0}
  - {x_m: 60.0, y_m: 3.5, yaw_rad: 0.12, curvature_1pm: 0.006, speed_mps: 12.0}
```

CLI 中也可以直接传 JSON：

```bash
conda run -n deep-sim python -m simulator.cli \
  --road split:dry:ice \
  --reference-json '{"type":"waypoints","lookahead_m":8.0,"points":[{"x_m":0,"y_m":0,"yaw_rad":0,"curvature_1pm":0,"speed_mps":10},{"x_m":30,"y_m":0,"yaw_rad":0,"curvature_1pm":0,"speed_mps":12},{"x_m":60,"y_m":3.5,"yaw_rad":0.12,"curvature_1pm":0.006,"speed_mps":12}]}' \
  --out-dir output/simulation/waypoint_demo
```

## 8. 输出文件

闭环仿真输出目录结构：

```text
output/simulation/<timestamp>/<run_name>/
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

文件含义：

| 文件 | 含义 |
| --- | --- |
| `manifest.json` | canonical dataset manifest，可直接被现有 data loader / validator 读取。 |
| `episodes/<episode_id>.npz` | 观测数组与 teacher-only aux labels。 |
| `episodes/<episode_id>.json` | episode sidecar，包含 metadata、固定车辆 context、nominal prior。 |
| `simulation_request.json` | 本次仿真的 request 快照，包含 resolved output path / dataset ID / scenario ID。 |
| `simulation_summary.json` | 本次仿真的摘要，包括 final state、observable range、输出路径。 |
| `debug_trace.json` | 每个 debug step 的结构化 controller / vehicle debug 信号。 |
| `debug_trace.csv` | 扁平化 debug trace，便于表格工具读取。 |
| `debug_report.html` | Plotly 交互式 HTML 报告，包含 reference 轨迹和车辆实际轨迹对比图。 |

`manifest.json` 和 `episodes/` 遵循 canonical dataset 格式，因此闭环仿真结果也可以作为后续分析或训练输入。

## 9. 可视化

默认 `write_debug_trace: true`、`write_debug_html: true` 时，仿真会自动输出：

```text
debug_trace.json
debug_trace.csv
debug_report.html
```

`debug_report.html` 是单文件 Plotly HTML，可以直接用浏览器打开。报告内的 `Trajectory` 面板用于对比 reference 轨迹和车辆实际轨迹，支持缩放、平移、hover 查看坐标和时间。

默认面板：

| 面板 | 主要信号 |
| --- | --- |
| `Trajectory` | 实际轨迹和 reference 轨迹。 |
| `Speed Tracking` | `vx`、目标速度、速度误差、加速度命令。 |
| `Lateral Tracking` | 横向位置、目标横向位置、航向角、横向误差、航向误差。 |
| `Sideslip` | `beta_deg`、横向速度、横摆率、横向加速度、侧倾角和航向角。 |
| `Commands` | 转向、油门、制动命令。 |
| `Vehicle State` | `vy`、`r`、四轮 μ、最小 μ、最大 friction usage。 |
| `Tire Slip Teacher States` | 四轮 slip angle 和 slip ratio。 |
| `Tire Force Teacher States` | 四轮 `Fy_true` 和 `Fz_true`。 |
| `Friction Usage` | 四轮 friction usage。 |
| `Reference` | path progress、lookahead、curvature 等 reference 信号。 |

可以在 request YAML 中自定义面板：

```yaml
debug_html_signals:
  Speed:
    - input.vx
    - input.reference.speed_mps
    - output.debug.longitudinal.speed_error
  Steering:
    - output.sw_angle
    - output.debug.lateral.raw_sw_angle
```

已有 `debug_trace.json` 时，可以单独重新生成 HTML：

```bash
conda run -n deep-sim python -m simulator.visualizer.report \
  --trace output/simulation/<timestamp>/double_lane_change_5mps/debug_trace.json \
  --out output/simulation/<timestamp>/double_lane_change_5mps/debug_report.html
```

如果要从 YAML/JSON 读取自定义面板：

```bash
conda run -n deep-sim python -m simulator.visualizer.report \
  --trace output/simulation/<timestamp>/double_lane_change_5mps/debug_trace.json \
  --panels path/to/debug_panels.yaml
```

## 10. 批量生成训练数据

批量数据生成入口是 `simulator.data_generator.generate`，它调用 `vehicle_model`，根据 `configs/datasets/*.yaml` 中的 scenario set 批量生成 canonical dataset。

示例：

```bash
conda run -n deep-sim python -m simulator.data_generator.generate \
  --config configs/datasets/ds0_minimal.yaml \
  --out data/ds0_minimal_regen
```

常用数据配置：

| 配置 | 用途 |
| --- | --- |
| `configs/datasets/ds0_minimal.yaml` | 最小 smoke 数据集。 |
| `configs/datasets/ds1_v1.yaml` | 多车、多工况 base 数据集。 |
| `configs/datasets/ds1_proxy_v1.yaml` | 带 proxy perturbation 的数据集。 |
| `configs/datasets/ds1_proxy_ft_v1.yaml` | fine-tune 目标车 / 目标时间段数据集。 |
| `configs/datasets/ds2_extreme_v0.yaml` | 极限工况 / MoE tire residual 后续使用的数据集。 |

所有 `scenario_set` 的输出目录至少包含：

```text
manifest.json
episodes/
generation_summary.json
```

`ds1`、`ds1_proxy`、`ds2_extreme` 会额外输出：

```text
scenario_matrix.json
split_manifest.json
scenario_coverage_report.json
dataset_qa_report.json
```

`ds1_proxy` 会再额外输出：

```text
perturbation_profiles.json
proxy_target_windows.json
proxy_distribution_report.json
```

长期保留、可复用的数据集应放在 `data/` 下；临时实验或运行过程产物应放在 `output/` 下。

## 11. 验证仿真输出

验证一个闭环仿真输出：

```bash
conda run -n deep-sim python - <<'PY'
from simulator.vehicle_model.validators import TeacherEpisodeValidator

report = TeacherEpisodeValidator().validate_dataset(
    "output/simulation/<timestamp>/double_lane_change_5mps"
)
print(report.to_dict())
PY
```

也可以通过实验入口生成/验证数据集，训练实验会自动调用 validator。

## 12. 如何替换控制器

控制器接口定义在 `simulator/controller/base.py`：

```python
class Controller(Protocol):
    def reset(self, scenario: ScenarioConfig) -> None:
        ...

    def compute(self, controller_input: ControllerInput) -> ControllerOutput:
        ...
```

要替换控制器有两种方式：

1. 新建一个实现 `reset()` 和 `compute()` 的 controller 类。
2. 在 Python 中直接调用 `run_closed_loop_simulation(request, controller=custom_controller)`。

默认 CLI 当前使用 `_default_controller()` 构建 `PIDLQRController`。如果希望 CLI 也支持新控制器，需要扩展 `ClosedLoopSimulationRequest` 和 `_default_controller()`。

默认控制器结构：

| 文件 | 作用 |
| --- | --- |
| `controller/longitudinal_pid.py` | 纵向速度 PID，输出加速度命令，再映射到油门/制动。 |
| `controller/lateral_lqr.py` | 横向 LQR，根据横向误差、航向误差等输出方向盘角。 |
| `controller/composite.py` | 把纵向 PID 和横向 LQR 组合成 `PIDLQRController`。 |

## 13. 如何扩展车辆模型

车辆模型主循环在 `vehicle_model/model.py`。高层流程是：

```text
initialize(scenario)
  -> build initial state
step(runtime, t, command)
  -> road query
  -> steering model
  -> drive/brake model
  -> suspension/load transfer
  -> tire force model
  -> aero model
  -> integrate vehicle state
  -> sensor observation
```

扩展点：

| 需求 | 修改位置 |
| --- | --- |
| 新路面类型 / μ 地图 | `vehicle_model/modules/road.py` 和 `vehicle_model/scenario.py` |
| 新轮胎模型 | `vehicle_model/modules/tire.py` |
| 新载荷转移 / 悬架模型 | `vehicle_model/modules/suspension.py` |
| 新执行器模型 | `vehicle_model/modules/drive_brake.py`、`vehicle_model/modules/steering.py` |
| 新传感器噪声 / 延迟 | `vehicle_model/modules/sensors.py` |
| 新车辆参数变体 | `vehicle_model/vehicle_params.py` |
| 新 scenario 组合 | `vehicle_model/scenario.py` |

## 14. 常见任务

### 重新跑 5 m/s 双移线

```bash
conda run -n deep-sim python -m simulator.cli \
  --request configs/simulator/double_lane_change_5mps.yaml
```

查看结果：

```text
output/simulation/<timestamp>/double_lane_change_5mps/debug_report.html
```

### 跑 split-μ 闭环仿真

```bash
conda run -n deep-sim python -m simulator.cli \
  --request configs/simulator/demo_single.yaml \
  --road split:dry:ice \
  --scenario-id demo_split_mu \
  --out-dir output/simulation/demo_split_mu
```

### 关闭 HTML，只输出数据

```bash
conda run -n deep-sim python -m simulator.cli \
  --request configs/simulator/demo_single.yaml \
  --no-write-debug-html \
  --out-dir output/simulation/demo_no_html
```

### 生成一个训练数据集

```bash
conda run -n deep-sim python -m simulator.data_generator.generate \
  --config configs/datasets/ds1_v1.yaml \
  --out data/ds1_v1_regen
```

### 只重新生成 HTML 报告

```bash
conda run -n deep-sim python -m simulator.visualizer.report \
  --trace output/simulation/<timestamp>/double_lane_change_5mps/debug_trace.json
```

## 15. 注意事项

- `simulator/` 已替代旧的 `teacher_simulator`。后续不要再新增 `teacher_simulator` 目录。
- 闭环仿真输出默认属于 `output/simulation/`；训练实验输出属于 `output/training/`。
- 可复用数据集放 `data/`，不要放 `output/`。
- `configs/datasets/` 是数据集生成配置；`configs/simulator/` 是单次闭环仿真 request 配置。
- 当前默认控制器只是 baseline：纵向 PID + 横向 LQR。后续真实控制器、MPC 或外部控制器应通过 `Controller` 接口接入。
- 如果仿真出现 `FloatingPointError`，优先检查 `road`、速度、控制器增益、`dt_internal` 和转向饱和参数。
