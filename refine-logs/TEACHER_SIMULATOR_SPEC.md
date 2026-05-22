# TEACHER_SIMULATOR_SPEC

**日期**：2026-05-21  
**状态**：implementation spec v0  
**作用**：定义 high-fidelity teacher simulator 的代码接口、状态、模块、配置、导出字段和验收测试。

## 0. 文档边界

`TEACHER_SIMULATOR_DESIGN.md` 解释 teacher simulator 的物理设计和保真度目标；本文档定义实现必须满足的接口和测试。字段 schema 以 `DATA_SCHEMA_SPEC.md` 为准。

Teacher simulator 的目标：

```text
生成比 student 更高保真的多车、多工况 episode
默认启用真实车相关复杂效应
导出 student-visible observable、teacher_aux_labels 和 episode_metadata
支持 DS0 / DS1 / DS2 数据生成
支持 sim-to-real proxy perturbation target windows
```

## 1. Package Boundary

建议实现模块：

```text
teacher_simulator/
  config.py
  scenario.py
  vehicle_params.py
  state.py
  simulator.py
  export.py
  validators.py
  modules/
    chassis.py
    suspension.py
    tire.py
    road.py
    steering.py
    drive_brake.py
    aero.py
    sensors.py
```

Public API：

```python
sim = TeacherSimulator(config: TeacherSimConfig)
episode = sim.run_episode(scenario: ScenarioConfig) -> EpisodeRecord
validator = TeacherEpisodeValidator(schema: DataSchema)
validator.validate(episode) -> ValidationReport
```

## 2. Core Data Objects

### 2.1 TeacherSimConfig

Required fields:

```yaml
TeacherSimConfig:
  dt_internal: float
  dt_export: float
  integrator: rk4 | semi_implicit_euler | dopri5
  gravity: float
  coordinate_convention: body_x_forward_y_left_z_up
  default_feature_flags: TeacherFeatureFlags
  numerical_limits:
    min_speed_for_slip: float
    min_fz: float
    max_abs_state: map
  export:
    include_teacher_aux_labels: true
    include_metadata: true
    resample_policy: hold_last | linear | anti_alias
```

Validation:

```text
dt_internal > 0
dt_export >= dt_internal
dt_export / dt_internal is integer for v0
gravity > 0
```

### 2.2 ScenarioConfig

Required fields:

```yaml
ScenarioConfig:
  vehicle_config: VehicleConfig
  road_profile: RoadProfile
  control_script: ControlScript
  actuator_profile: ActuatorProfile
  sensor_profile: SensorProfile
  environment_profile: EnvironmentProfile
  teacher_feature_flags: TeacherFeatureFlags
  split_metadata: SplitMetadata
```

`ScenarioConfig` must cover `CG-SINGLE`, `CG-SPLIT`, `CG-TRANSITION` and later DS2 extreme scenarios.

### 2.3 EpisodeRecord

Output must match `DATA_SCHEMA_SPEC.md`:

```yaml
EpisodeRecord:
  metadata: EpisodeMetadata
  fixed_vehicle_context: FixedVehicleContext
  nominal_physics_prior: NominalPhysicsPrior
  time_series_observable: TimeSeriesObservable
  teacher_aux_labels: TeacherAuxLabels
```

No simulator module may write directly to student model input. Export is mediated by `export.py` and schema validation.

## 3. Runtime Environment

Teacher simulator 的所有命令必须在 Miniforge / conda 虚拟环境中执行。默认使用仓库根目录：

```text
environment.yml
```

默认环境名：

```text
deep-sim
```

创建和激活：

```bash
conda env create -f environment.yml
conda activate deep-sim
python -m pip install -r requirements.txt
```

自动化执行时必须使用已激活环境，或显式使用：

```bash
conda run -n deep-sim python -m teacher_simulator.generate --config configs/datasets/ds0.yaml --out data/ds0
```

如果环境不存在或无法激活，teacher generation / validation run 必须失败或标记为 `blocked`，不能退回系统 Python。

## 4. Teacher State

Teacher internal state is higher-dimensional than student state.

Minimum default state groups:

```text
chassis:
  world position: x_world, y_world, z_world
  body velocity: vx, vy, vz
  orientation: roll, pitch, yaw
  body angular velocity: p, q, r

wheels:
  omega_fl, omega_fr, omega_rl, omega_rr

suspension / unsprung:
  per-wheel suspension displacement and velocity
  unsprung vertical state
  tire vertical deflection

tire internal:
  relaxation states for longitudinal/lateral slip
  tire temperature / wear / pressure states if enabled

actuator:
  steering delay/compliance/backlash/hysteresis states
  drive torque lag/saturation states
  brake pressure/thermal/fade states
  ABS/TCS/ESC modulation states

sensor:
  bias, delay buffers, filter states, dropout/timestamp jitter states
```

Student visible subset is exported only through sensor model outputs defined in `DATA_SCHEMA_SPEC.md`.

## 5. Module Interfaces

### 5.1 Chassis Dynamics

Interface:

```python
chassis_derivatives = ChassisDynamics.step(
    state: TeacherState,
    forces: WheelForces,
    aero: AeroForces,
    road: RoadContact,
    params: TeacherVehicleParams,
) -> ChassisDerivatives
```

Required behavior:

```text
6-DOF teacher chassis by default
body-frame longitudinal/lateral/yaw dynamics available for student projection
roll/pitch dynamics coupled to load transfer
grade/bank/roughness affect normal force and body acceleration
```

### 5.2 Suspension And Load Transfer

Interface:

```python
load_state = SuspensionModel.step(
    state: TeacherState,
    road_contact: RoadContact,
    params: TeacherVehicleParams,
) -> SuspensionLoadState
```

Required outputs:

```text
Fz_true_i
suspension states
unsprung states
camber_true_i
toe_true_i
load transfer diagnostics
```

Requirements:

```text
Fz_true_i must be nonnegative after contact handling
front/rear and left/right load transfer signs must pass braking/turning smoke tests
rough-road and contact events must be represented in teacher_aux_labels or diagnostics
```

### 5.3 Tire Model

Interface:

```python
tire_out = TireModel.step(
    state: TeacherState,
    wheel_inputs: WheelInputs,
    road_mu: WheelRoadMu,
    load_state: SuspensionLoadState,
    tire_params: TeacherTireParams,
) -> TireForces
```

Required outputs:

```text
Fx_true_i
Fy_true_i
Fz_true_i passthrough or coupled output
Mz_true_i optional
mu_true_i
slip_ratio_true_i
slip_angle_true_i
friction_usage_i
camber_true_i / toe_true_i
```

Requirements:

```text
combined slip supported
transient slip / relaxation state supported by default
force saturation compatible with μ_i * Fz_i
tire thermal / wear / pressure feature flags default enabled for full-fidelity profile
```

### 5.4 Road Model

Interface:

```python
road_contact = RoadModel.query(
    wheel_world_positions: float[4, 3],
    t: float,
    road_profile: RoadProfile,
) -> RoadContact
```

Required profiles:

```text
single μ: dry / wet / snow / ice
split μ: left/right ordered maps
transition μ: A -> B and B -> A transitions
grade / bank
roughness / road height / road normal
patchy μ and wheel-level μ map for DS2
```

Road labels and true μ are teacher auxiliary labels, not student input.

### 5.5 Steering Actuator

Interface:

```python
steering_out = SteeringActuator.step(
    command: SteeringCommand,
    state: SteeringActuatorState,
    params: TeacherActuatorParams,
) -> SteeringOutput
```

Required outputs:

```text
delta_eff_fl
delta_eff_fr
actuator delay / compliance / backlash / hysteresis states
```

Requirements:

```text
first-order lag, saturation, compliance, backlash and hysteresis enabled by default
temperature/aging drift represented via hidden params or profile
student only sees sensor-corrupted sw_angle / steer_cmd, not delta_eff_i
```

### 5.6 Drive And Brake Actuators

Interface:

```python
torque_out = DriveBrakeActuator.step(
    throttle_cmd: float,
    brake_cmd: float,
    state: DriveBrakeState,
    params: TeacherActuatorParams,
) -> WheelTorqueOutput
```

Required outputs:

```text
tau_drv_true_i
tau_brk_true_i
brake_pressure_i
brake_temperature_i
ABS/TCS/ESC modulation states
tau_drv_obs_i / tau_brk_obs_i after sensor/estimator path
```

`torque_observability_mode` must be one of:

```text
true_per_wheel_sensor
actuator_estimate
command_only_projection
```

### 5.7 Aero And Environment

Interface:

```python
aero = AeroModel.step(
    state: TeacherState,
    environment: EnvironmentProfile,
    params: TeacherVehicleParams,
) -> AeroForces
```

Requirements:

```text
drag, lift/downforce, pitch/yaw/roll moments supported
wind and crosswind profiles supported
diagnostics exported as teacher_aux_labels
```

### 5.8 Sensor Model

Interface:

```python
observable = SensorModel.observe(
    true_state: TeacherState,
    true_outputs: TeacherAuxLabels,
    sensor_profile: SensorProfile,
    t: float,
) -> TimeSeriesObservableSample
```

Required effects:

```text
noise
bias
delay
filtering
quantization
dropout
timestamp jitter
mounting offset
```

Requirements:

```text
observable timestamps remain monotonic
sensor true states and sensor error states are teacher_aux_labels
observable values are never aliases of noiseless true variables unless profile explicitly sets zero noise/delay for DS0 debug
```

## 6. Simulation Loop

Required execution order per internal step:

```text
1. evaluate control script at t
2. update steering actuator
3. update drive/brake actuators
4. query road contact at wheel positions
5. update suspension/load transfer
6. compute tire slip and tire forces
7. compute aero/environment forces
8. integrate chassis/wheel/suspension/tire/actuator states
9. run sensor model and append observable sample if export tick
10. append teacher_aux_labels and metadata references
```

Integrator requirements:

```text
DS0 may use semi-implicit Euler for debug if stable
DS1 default should use RK4 or equivalent stable integrator
internal dt must be small enough to resolve actuator and tire transient states
all exported data resampled to dt_export
```

## 7. Vehicle Parameter Randomization

Teacher hidden params must support randomization over:

```text
true mass / inertia / cg
suspension stiffness / damping / anti-roll distribution
tire stiffness / friction behavior / relaxation length
tire thermal / wear / pressure
actuator delay / saturation / hysteresis
sensor bias / noise / delay / dropout
aero coefficients
road roughness / grade / bank / μ maps
```

Student-visible fields remain limited to:

```text
fixed_vehicle_context
nominal_physics_prior
time_series_observable
```

Hidden parameters must be hashed into `vehicle_internal_params_hash` for audit and split validation.

## 8. Export Contract

`export.py` must produce:

```text
episode metadata
fixed_vehicle_context
nominal_physics_prior
time_series_observable
teacher_aux_labels
schema version
teacher simulator version
```

Export formats:

```text
v0 required: parquet or npz + JSON/YAML metadata
v0 recommended: one episode per file or shard with index manifest
```

Manifest fields:

```yaml
dataset_id:
schema_version:
teacher_model_version:
episodes:
  - episode_id:
    path:
    split_role:
    scenario_id:
    vehicle_config_id:
    vehicle_internal_params_hash:
```

## 9. Scenario Matrix Support

Teacher must generate:

```text
DS0:
  representative smoke cases from CG-SINGLE / CG-SPLIT / CG-TRANSITION

DS1:
  full factorized matrix:
    5 longitudinal factors
    5 lateral factors
    4 single road factors
    12 split μ factors
    12 transition μ factors

DS2:
  DS1 plus extreme handling, patchy μ, wheel-level μ maps and expanded vehicle/load/tire states
```

Scenario IDs must follow:

```text
{road_factor_id}-{longitudinal_factor_id}-{lateral_factor_id}
```

## 10. Validation Gates

### 10.1 Unit Tests

Required before DS0:

```text
coordinate convention tests
wheel_order tests
braking increases front Fz on flat road
left turn increases right-side Fz under standard convention
split μ braking creates expected yaw moment sign
friction usage remains finite
sensor delay and timestamp jitter preserve monotonic timestamps
schema validator blocks teacher_aux_labels from student input
metadata remains available but excluded from model input
```

### 10.2 DS0 Acceptance

DS0 must pass:

```text
5-10 representative episodes generated
all required observable fields present
all required teacher_aux_labels present
metadata complete
no NaN/Inf
Fz_i nonnegative except explicitly marked contact-loss cases
friction ellipse diagnostics finite
tiny black-box and tiny hybrid can overfit 5-10 short sequences
```

### 10.3 DS1 Acceptance

DS1 must pass:

```text
multi-vehicle / multi-config generation complete
CG-SINGLE / CG-SPLIT / CG-TRANSITION coverage complete
held-out road μ metadata complete
held-out vehicle/config metadata complete
held-out target time window metadata complete
FTD0-FTD5 buckets reproducible
vehicle_internal_params_hash traceable and excluded from student input
teacher_feature_flags all enabled for full-fidelity unless explicitly downgraded
torque_observability_mode complete and consistent with torque observable fields
```

## 11. Required CLI

Implementation should expose:

```bash
conda activate deep-sim
python -m teacher_simulator.generate --config configs/datasets/ds0.yaml --out data/ds0
python -m teacher_simulator.generate --config configs/datasets/ds1.yaml --out data/ds1
python -m teacher_simulator.validate --dataset data/ds0 --schema configs/schema_v0.yaml
python -m teacher_simulator.validate --dataset data/ds1 --schema configs/schema_v0.yaml
```

Non-interactive execution:

```bash
conda run -n deep-sim python -m teacher_simulator.generate --config configs/datasets/ds0.yaml --out data/ds0
```

Exit codes:

```text
0: pass
1: validation failure
2: configuration/schema error
3: numerical integration failure
```

## 12. Versioning

Every exported episode must include:

```text
schema_version
teacher_model_version
generator_git_commit if available
teacher_feature_flags.generator_version
```

Any feature downgrade must set:

```yaml
teacher_feature_flags:
  disabled_features: [...]
  downgrade_reason: "..."
```
