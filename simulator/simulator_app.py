import json
import os
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from simulator.controller import (
    Controller,
    ControllerInput,
    ControllerReference,
    CoupledMPCConfig,
    LateralLQRConfig,
    LongitudinalPIDConfig,
    SimulationController,
    SimulationControllerConfig,
)
from simulator.reference import build_reference_provider
from simulator.vehicle_model.config import config_from_dict, load_yaml
from simulator.vehicle_model.export import export_dataset
from simulator.vehicle_model.model import VehicleModel, VehicleStepResult
from simulator.vehicle_model.scenario import (
    ROAD_MU,
    ActuatorProfile,
    ControlScript,
    EnvironmentProfile,
    InitialState,
    RoadProfile,
    ScenarioConfig,
    SensorProfile,
    SplitMetadata,
)
from simulator.vehicle_model.vehicle_params import make_vehicle_config_variant
from simulator.visualizer import DebugTrace


def default_simulator_model_config() -> Dict[str, Any]:
    return {
        "schema_version": "v0",
        "teacher_model_version": "teacher_simulator_v0",
        "dt_internal": 0.01,
        "dt_export": 0.02,
        "duration_s": 12.0,
        "integrator": "semi_implicit_euler",
        "gravity": 9.81,
        "coordinate_convention": "body_x_forward_y_left_z_up",
        "default_feature_flags": {
            "tire_relaxation": True,
            "tire_thermal_wear_pressure": True,
            "suspension_unsprung": True,
            "steering_lag_compliance_backlash_hysteresis": True,
            "drive_brake_lag_saturation_abs": True,
            "aero_wind": True,
            "sensor_noise_delay_filter_quantization": True,
            "generator_version": "teacher_simulator_v0",
            "disabled_features": [],
            "downgrade_reason": "",
        },
        "numerical_limits": {
            "min_speed_for_slip": 0.5,
            "min_fz": 50.0,
            "max_abs_state": {
                "vx": 80.0,
                "vy": 30.0,
                "roll": 1.0,
                "pitch": 1.0,
                "r": 3.0,
            },
        },
        "export": {
            "include_teacher_aux_labels": True,
            "include_metadata": True,
            "resample_policy": "hold_last",
        },
    }


def default_simulator_scenario() -> Dict[str, Any]:
    return {
        "id": None,
        "road": {"type": "single", "surface": "dry"},
        "vehicle_index": 0,
        "seed": 0,
        "initial_state": {
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": 0.0,
            "yaw_rad": 0.0,
            "speed_mps": 12.0,
        },
        "reference": None,
    }


@dataclass
class ClosedLoopSimulationRequest:
    model: Dict[str, Any] = field(default_factory=default_simulator_model_config)
    scenario: Dict[str, Any] = field(default_factory=default_simulator_scenario)
    out_dir: Optional[str] = None
    sensor_noise_scale: float = 1.0
    timestamp_jitter_std_s: float = 0.00015
    sensor_dropout_probability: float = 0.0
    sensor_quantization: float = 1e-5
    torque_observability_mode: str = "actuator_estimate"
    steering_saturation_rad: float = 0.62
    actuator_sensor_delay_steps: int = 1
    wind_x_mps: float = 0.0
    wind_y_mps: float = 0.5
    controller_type: str = "modular"
    pid_kp: float = 0.18
    pid_ki: float = 0.025
    pid_kd: float = 0.02
    lqr_gains: Sequence[float] = (0.18, 0.85, 0.04, 0.16)
    lqr_max_sw_angle_rad: float = 1.2
    mpc_config: Dict[str, Any] = field(default_factory=dict)
    debug_stride: int = 1
    write_debug_trace: bool = True
    write_debug_html: bool = True
    debug_html_signals: Optional[Dict[str, Sequence[str]]] = None
    timestamped_output: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClosedLoopSimulationRequest":
        valid = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - valid)
        if unknown:
            raise ValueError("unknown simulator request fields: %s" % ", ".join(unknown))
        request = cls(**data)
        if not isinstance(request.model, dict):
            raise ValueError("model must be a mapping")
        if not isinstance(request.scenario, dict):
            raise ValueError("scenario must be a mapping")
        _validate_scenario_schema(request.scenario)
        return request

    def with_overrides(self, overrides: Dict[str, Any]) -> "ClosedLoopSimulationRequest":
        clean = {key: value for key, value in overrides.items() if value is not None}
        if not clean:
            return self
        return replace(self, **clean)


class SimulatorApp:
    def __init__(
        self,
        request: ClosedLoopSimulationRequest,
        controller: Optional[Controller] = None,
    ) -> None:
        self.request = request
        self.config = config_from_dict(request.model)
        self.config.validate()
        self.vehicle_model = VehicleModel(self.config)
        self.scenario = build_scenario(request)
        reference_config = _reference_config(request)
        if reference_config is None:
            raise ValueError("simulator request must define reference")
        fallback_reference = ControllerReference()
        self.reference_provider = build_reference_provider(
            reference_config,
            fallback_reference,
        )
        self.controller = controller or _default_controller(request)
        self.debug_trace = DebugTrace()

    def run(self) -> Dict[str, Any]:
        runtime = self.vehicle_model.initialize(self.scenario)
        self.controller.reset(self.scenario)
        self.reference_provider.reset(self.scenario)

        obs_rows: List[Dict[str, float]] = []
        aux_rows: Dict[str, List[Any]] = {}
        road_labels: List[List[str]] = []
        last_observation: Optional[Dict[str, float]] = None
        last_result: Optional[VehicleStepResult] = None
        n_steps = int(round(self.config.duration_s / self.config.dt_internal))
        export_stride = int(round(self.config.dt_export / self.config.dt_internal))

        for step in range(n_steps + 1):
            t = step * self.config.dt_internal
            reference = self.reference_provider.query(t, runtime.state)
            controller_input = ControllerInput(
                t=t,
                dt=self.config.dt_internal,
                state=runtime.state,
                scenario=self.scenario,
                reference=reference,
                observation=last_observation,
            )
            controller_output = self.controller.compute(controller_input)
            result = self.vehicle_model.step(
                runtime,
                t,
                controller_output.command_dict(),
            )
            last_observation = result.observation
            last_result = result
            if self.request.write_debug_trace and step % max(1, self.request.debug_stride) == 0:
                self.debug_trace.append(
                    t=t,
                    controller_input=controller_input.debug_dict(),
                    controller_output={
                        **controller_output.command_dict(),
                        "debug": controller_output.debug,
                    },
                    vehicle_debug=_vehicle_debug(result),
                )

            if step % export_stride == 0:
                obs_rows.append(result.observation)
                self.vehicle_model._append_aux(
                    aux_rows,
                    result.tire,
                    result.load_state,
                    result.torques,
                    result.steering,
                    result.road_contact,
                    result.aero,
                )
                road_labels.append(list(result.road_contact["labels"]))

        episode = self.vehicle_model.build_episode(
            scenario=self.scenario,
            obs_rows=obs_rows,
            aux_rows=aux_rows,
            road_labels=road_labels,
            final_state=runtime.state,
            last_modules_present=last_result is not None,
        )
        episode["metadata"]["control_mode"] = _control_mode(self.request)
        episode["metadata"]["controller"] = {
            "type": self.controller.__class__.__name__,
            "controller_type": self.request.controller_type,
            "reference_provider": self.reference_provider.describe(),
        }
        return self._export(episode)

    def _export(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        output_root = self.request.out_dir or _default_out_dir(self.scenario.scenario_id)
        out_dir = _resolve_output_dir(output_root, self.request.timestamped_output)
        dataset_id = "SIM_%s" % _safe_stem(self.scenario.scenario_id)
        manifest = export_dataset(
            [episode],
            out_dir,
            dataset_id=dataset_id,
            schema_version=self.config.schema_version,
            teacher_model_version=self.config.teacher_model_version,
        )
        request_payload = asdict(self.request)
        request_payload.update(
            {
                "resolved_out_dir": out_dir,
                "resolved_output_root": output_root,
                "resolved_dataset_id": dataset_id,
                "resolved_scenario_id": self.scenario.scenario_id,
            }
        )
        _write_json(os.path.join(out_dir, "simulation_request.json"), request_payload)
        debug_trace_json_path = os.path.join(out_dir, "debug_trace.json")
        debug_trace_csv_path = os.path.join(out_dir, "debug_trace.csv")
        debug_report_path = os.path.join(out_dir, "debug_report.html")
        if self.request.write_debug_trace:
            self.debug_trace.write_json(debug_trace_json_path)
            self.debug_trace.write_csv(debug_trace_csv_path)
            if self.request.write_debug_html:
                self.debug_trace.write_html(
                    debug_report_path,
                    panels=self.request.debug_html_signals,
                    title="Simulator Debug Report - %s" % self.scenario.scenario_id,
                )
        summary = _simulation_summary(out_dir, dataset_id, manifest, episode)
        if self.request.write_debug_trace:
            summary["debug_trace_json_path"] = debug_trace_json_path
            summary["debug_trace_csv_path"] = debug_trace_csv_path
            if self.request.write_debug_html:
                summary["debug_report_path"] = debug_report_path
        _write_json(os.path.join(out_dir, "simulation_summary.json"), summary)
        return summary


def load_simulation_request(path: str) -> ClosedLoopSimulationRequest:
    data = load_yaml(path)
    _resolve_reference_file(data, os.path.dirname(path))
    return ClosedLoopSimulationRequest.from_dict(data)


def run_closed_loop_simulation(
    request: ClosedLoopSimulationRequest,
    controller: Optional[Controller] = None,
) -> Dict[str, Any]:
    return SimulatorApp(request, controller=controller).run()


def build_scenario(request: ClosedLoopSimulationRequest) -> ScenarioConfig:
    scenario_data = _scenario_data(request)
    vehicle_index = int(scenario_data.get("vehicle_index", 0))
    vehicle = make_vehicle_config_variant(vehicle_index)
    road = build_road_profile(request)
    scenario_id = scenario_data.get("id")
    scenario_id = str(scenario_id) if scenario_id else _default_scenario_id(
        vehicle_index, road
    )
    initial_state = _initial_state_from_scenario(scenario_data)
    initial_speed = (
        12.0 if initial_state.speed_mps is None else float(initial_state.speed_mps)
    )
    return ScenarioConfig(
        scenario_id=scenario_id,
        vehicle_config=vehicle,
        road_profile=road,
        control_script=ControlScript(
            longitudinal="closed_loop_speed_pid",
            lateral="closed_loop_lqr",
            initial_speed_mps=initial_speed,
            longitudinal_id="CL-PID",
            lateral_id="CL-LQR",
        ),
        initial_state=initial_state,
        actuator_profile=ActuatorProfile(
            torque_observability_mode=request.torque_observability_mode,
            steering_saturation_rad=request.steering_saturation_rad,
            sensor_delay_steps=request.actuator_sensor_delay_steps,
        ),
        sensor_profile=SensorProfile(
            noise_scale=request.sensor_noise_scale,
            timestamp_jitter_std_s=request.timestamp_jitter_std_s,
            dropout_probability=request.sensor_dropout_probability,
            quantization=request.sensor_quantization,
        ),
        environment_profile=EnvironmentProfile(
            wind_x_mps=request.wind_x_mps,
            wind_y_mps=request.wind_y_mps,
        ),
        split_metadata=SplitMetadata(split_role="simulation"),
        validation_case="closed_loop_simulation",
        seed=int(scenario_data.get("seed", 0)),
        scenario_group="SIM-CLOSED-LOOP",
        longitudinal_factor_id="CL-PID",
        lateral_factor_id="CL-LQR",
    )


def build_road_profile(request: ClosedLoopSimulationRequest) -> RoadProfile:
    scenario_data = _scenario_data(request)
    road_spec = scenario_data.get("road", {"type": "single", "surface": "dry"})
    if isinstance(road_spec, dict):
        return _build_road_profile_from_mapping(road_spec)
    road_spec = str(road_spec).strip()
    if "->" in road_spec and ":" not in road_spec:
        left, right = [part.strip() for part in road_spec.split("->", 1)]
        road_spec = "transition:%s:%s" % (left, right)
    parts = [part.strip() for part in road_spec.split(":") if part.strip()]
    if len(parts) == 1 and parts[0] in ROAD_MU:
        parts = ["single", parts[0]]
    if not parts:
        raise ValueError("road must not be empty")
    road_type = parts[0]
    for surface in parts[1:]:
        _validate_surface(surface)
    common = {
        "transition_start_s": 4.0,
        "transition_duration_s": 3.0,
        "grade_rad": 0.0,
        "bank_rad": 0.0,
        "roughness_amp_m": 0.003,
        "scenario_group": "SIM-CLOSED-LOOP",
    }
    if road_type == "single" and len(parts) == 2:
        return RoadProfile(
            road_type="single",
            mu_pattern=parts[1],
            single_surface=parts[1],
            **common,
        )
    if road_type == "split" and len(parts) == 3:
        return RoadProfile(
            road_type="split",
            mu_pattern="split",
            split_left=parts[1],
            split_right=parts[2],
            **common,
        )
    if road_type == "transition" and len(parts) == 3:
        return RoadProfile(
            road_type="transition",
            mu_pattern="transition",
            transition_from=parts[1],
            transition_to=parts[2],
            **common,
        )
    raise ValueError(
        "unsupported road spec '%s'; use dry, single:dry, split:dry:ice, or transition:dry:wet"
        % road_spec
    )


def _scenario_data(request: ClosedLoopSimulationRequest) -> Dict[str, Any]:
    if not isinstance(request.scenario, dict):
        raise ValueError("scenario must be a mapping")
    _validate_scenario_schema(request.scenario)
    return request.scenario


def _validate_scenario_schema(scenario: Dict[str, Any]) -> None:
    allowed = {"id", "road", "vehicle_index", "seed", "initial_state", "reference"}
    unknown = sorted(set(scenario) - allowed)
    if unknown:
        raise ValueError("unknown simulator scenario fields: %s" % ", ".join(unknown))
    initial_state = scenario.get("initial_state")
    if initial_state is not None:
        if not isinstance(initial_state, dict):
            raise ValueError("scenario.initial_state must be a mapping")
        allowed_initial = {
            "x_m",
            "y_m",
            "z_m",
            "speed_mps",
            "vy_mps",
            "vz_mps",
            "roll_rad",
            "pitch_rad",
            "yaw_rad",
            "roll_rate_rps",
            "pitch_rate_rps",
            "yaw_rate_rps",
        }
        unknown_initial = sorted(set(initial_state) - allowed_initial)
        if unknown_initial:
            raise ValueError(
                "unknown simulator scenario.initial_state fields: %s"
                % ", ".join(unknown_initial)
            )


def _reference_config(request: ClosedLoopSimulationRequest) -> Optional[Dict[str, Any]]:
    reference = _scenario_data(request).get("reference")
    if reference is None:
        return None
    if isinstance(reference, str):
        return _load_reference_file(reference, None)
    if isinstance(reference, dict):
        if "file" in reference:
            loaded = _load_reference_file(str(reference["file"]), None)
            return {**loaded, **{k: v for k, v in reference.items() if k != "file"}}
        return reference
    raise ValueError("scenario.reference must be a mapping or file path")


def _resolve_reference_file(data: Dict[str, Any], base_dir: str) -> None:
    scenario = data.get("scenario")
    if not isinstance(scenario, dict):
        return
    reference = scenario.get("reference")
    if isinstance(reference, str):
        scenario["reference"] = _load_reference_file(reference, base_dir)
    elif isinstance(reference, dict) and "file" in reference:
        loaded = _load_reference_file(str(reference["file"]), base_dir)
        scenario["reference"] = {
            **loaded,
            **{k: v for k, v in reference.items() if k != "file"},
        }


def _load_reference_file(path: str, base_dir: Optional[str]) -> Dict[str, Any]:
    candidates = [path]
    if base_dir and not os.path.isabs(path):
        candidates.insert(0, os.path.join(base_dir, path))
    for candidate in candidates:
        if os.path.exists(candidate):
            data = load_yaml(candidate)
            data.setdefault("source_path", candidate)
            return data
    raise ValueError("reference file not found: %s" % path)


def _initial_state_from_scenario(scenario_data: Dict[str, Any]) -> InitialState:
    data = dict(scenario_data.get("initial_state", {}) or {})
    return InitialState(
        x_m=float(data.get("x_m", 0.0)),
        y_m=float(data.get("y_m", 0.0)),
        z_m=float(data.get("z_m", 0.0)),
        speed_mps=(
            None if data.get("speed_mps") is None else float(data.get("speed_mps"))
        ),
        vy_mps=float(data.get("vy_mps", 0.0)),
        vz_mps=float(data.get("vz_mps", 0.0)),
        roll_rad=float(data.get("roll_rad", 0.0)),
        pitch_rad=float(data.get("pitch_rad", 0.0)),
        yaw_rad=float(data.get("yaw_rad", 0.0)),
        roll_rate_rps=float(data.get("roll_rate_rps", 0.0)),
        pitch_rate_rps=float(data.get("pitch_rate_rps", 0.0)),
        yaw_rate_rps=float(data.get("yaw_rate_rps", 0.0)),
    )


def _build_road_profile_from_mapping(data: Dict[str, Any]) -> RoadProfile:
    road_type = str(data.get("type", "single"))
    common = {
        "transition_start_s": float(data.get("transition_start_s", 4.0)),
        "transition_duration_s": float(data.get("transition_duration_s", 3.0)),
        "transition_start_x_m": float(data.get("transition_start_x_m", 25.0)),
        "transition_length_m": float(data.get("transition_length_m", 35.0)),
        "split_boundary_y_m": float(data.get("split_boundary_y_m", 0.0)),
        "split_start_x_m": float(data.get("split_start_x_m", 0.0)),
        "split_end_x_m": float(data.get("split_end_x_m", 120.0)),
        "split_default_surface": str(data.get("split_default_surface", "dry")),
        "grade_rad": float(data.get("grade_rad", 0.0)),
        "bank_rad": float(data.get("bank_rad", 0.0)),
        "roughness_amp_m": float(data.get("roughness_amp_m", 0.003)),
        "scenario_group": "SIM-CLOSED-LOOP",
    }
    if road_type == "single":
        surface = str(data.get("surface", data.get("single_surface", "dry")))
        _validate_surface(surface)
        return RoadProfile(
            road_type="single",
            mu_pattern=surface,
            single_surface=surface,
            **common,
        )
    if road_type == "split":
        left = str(data.get("left", data.get("split_left", "dry")))
        right = str(data.get("right", data.get("split_right", "ice")))
        _validate_surface(left)
        _validate_surface(right)
        return RoadProfile(
            road_type="split",
            mu_pattern="split",
            split_left=left,
            split_right=right,
            **common,
        )
    if road_type == "transition":
        start = str(data.get("from", data.get("transition_from", "dry")))
        end = str(data.get("to", data.get("transition_to", "wet")))
        _validate_surface(start)
        _validate_surface(end)
        return RoadProfile(
            road_type="transition",
            mu_pattern="transition",
            transition_from=start,
            transition_to=end,
            **common,
        )
    raise ValueError("unsupported road type: %s" % road_type)


def _default_controller(request: ClosedLoopSimulationRequest) -> SimulationController:
    gains = tuple(float(x) for x in request.lqr_gains)
    if len(gains) != 4:
        raise ValueError("lqr_gains must contain 4 values")
    return SimulationController(
        SimulationControllerConfig(
            controller_type=request.controller_type,
            longitudinal=LongitudinalPIDConfig(
                kp=request.pid_kp,
                ki=request.pid_ki,
                kd=request.pid_kd,
            ),
            lateral=LateralLQRConfig(
                gains=gains,
                max_sw_angle_rad=request.lqr_max_sw_angle_rad,
            ),
            mpc=CoupledMPCConfig.from_dict(request.mpc_config),
        )
    )


def _control_mode(request: ClosedLoopSimulationRequest) -> str:
    if request.controller_type == "modular":
        return "closed_loop_pid_lqr"
    if request.controller_type == "coupled_mpc":
        return "closed_loop_coupled_mpc"
    return "closed_loop_%s" % _safe_stem(request.controller_type)


def _validate_surface(surface: str) -> None:
    if surface not in ROAD_MU:
        raise ValueError(
            "unsupported road surface '%s'; supported: %s"
            % (surface, ", ".join(sorted(ROAD_MU)))
        )


def _default_scenario_id(vehicle_index: int, road: RoadProfile) -> str:
    return "closed_loop_v%d_%s_speed_pid_lqr" % (
        vehicle_index,
        road.factor_id,
    )


def _default_out_dir(scenario_id: str) -> str:
    return os.path.join("output", "simulation", _safe_stem(scenario_id))


def _resolve_output_dir(output_root: str, timestamped_output: bool) -> str:
    if not timestamped_output:
        return output_root
    output_root = os.path.normpath(output_root)
    parent, run_name = os.path.split(output_root)
    if not parent:
        parent = "."
    if not run_name:
        raise ValueError("out_dir must include a run directory name")
    for index in range(100):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        suffix = "" if index == 0 else "_%02d" % index
        candidate = os.path.join(parent, stamp + suffix, run_name)
        if not os.path.exists(candidate):
            return candidate
    raise RuntimeError("could not allocate timestamped simulation output directory")


def _safe_stem(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in ["_", "-", "."]:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep)


_WHEEL_SUFFIXES = ("fl", "fr", "rl", "rr")


def _vehicle_debug(result: VehicleStepResult) -> Dict[str, Any]:
    mu = np.asarray(result.road_contact["mu"], dtype=np.float64)
    beta_rad = float(np.arctan2(result.state.vy, max(abs(result.state.vx), 1e-6)))
    debug = {
        "vx": float(result.state.vx),
        "vy": float(result.state.vy),
        "vz": float(result.state.vz),
        "x_world": float(result.state.x_world),
        "y_world": float(result.state.y_world),
        "z_world": float(result.state.z_world),
        "roll": float(result.state.roll),
        "pitch": float(result.state.pitch),
        "yaw": float(result.state.yaw),
        "p": float(result.state.p),
        "q": float(result.state.q),
        "r": float(result.state.r),
        "beta_rad": beta_rad,
        "beta_deg": float(np.degrees(beta_rad)),
        "ax_body_mps2": float(result.state.prev_ax),
        "ay_body_mps2": float(result.state.prev_ay),
        "mu_min": float(np.min(result.tire["mu_true_i"])),
        "mu_fl": float(mu[0]),
        "mu_fr": float(mu[1]),
        "mu_rl": float(mu[2]),
        "mu_rr": float(mu[3]),
        "friction_usage_max": float(np.max(result.tire["friction_usage_i"])),
    }
    _add_wheel_debug(debug, "mu", result.tire["mu_true_i"])
    _add_wheel_debug(debug, "slip_angle", result.tire["slip_angle_true_i"])
    _add_wheel_debug(debug, "slip_ratio", result.tire["slip_ratio_true_i"])
    _add_wheel_debug(debug, "friction_usage", result.tire["friction_usage_i"])
    _add_wheel_debug(debug, "Fx_true", result.tire["Fx_true_i"])
    _add_wheel_debug(debug, "Fy_true", result.tire["Fy_true_i"])
    _add_wheel_debug(debug, "Fz_true", result.load_state["Fz_true_i"])
    _add_wheel_debug(debug, "Mz_true", result.tire["Mz_true_i"])
    _add_wheel_debug(debug, "camber_true", result.load_state["camber_true_i"])
    _add_wheel_debug(debug, "toe_true", result.load_state["toe_true_i"])
    _add_wheel_debug(debug, "road_height_true", result.road_contact["height"])
    _add_wheel_xy_debug(debug, result.road_contact["wheel_xy"])
    delta_eff = np.asarray(result.steering["delta_eff_i"], dtype=np.float64)
    debug["delta_eff_fl"] = float(delta_eff[0])
    debug["delta_eff_fr"] = float(delta_eff[1])
    return debug


def _add_wheel_debug(
    debug: Dict[str, Any],
    prefix: str,
    values: Any,
) -> None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    for suffix, value in zip(_WHEEL_SUFFIXES, arr):
        debug["%s_%s" % (prefix, suffix)] = float(value)


def _add_wheel_xy_debug(debug: Dict[str, Any], values: Any) -> None:
    arr = np.asarray(values, dtype=np.float64)
    for suffix, xy in zip(_WHEEL_SUFFIXES, arr):
        debug["road_wheel_x_%s" % suffix] = float(xy[0])
        debug["road_wheel_y_%s" % suffix] = float(xy[1])


def _simulation_summary(
    out_dir: str,
    dataset_id: str,
    manifest: Dict[str, Any],
    episode: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = episode["metadata"]
    obs = episode["time_series_observable"]
    return {
        "status": "success",
        "out_dir": out_dir,
        "manifest_path": os.path.join(out_dir, "manifest.json"),
        "dataset_id": dataset_id,
        "episode_count": len(manifest["episodes"]),
        "episode_id": metadata["episode_id"],
        "scenario_id": metadata["scenario_id"],
        "control_mode": metadata.get("control_mode"),
        "controller": metadata.get("controller"),
        "vehicle_config_id": metadata["vehicle_config_id"],
        "road_type": metadata["road_type"],
        "road_factor_id": metadata["road_factor_id"],
        "sample_count": metadata["sample_count"],
        "duration_s": metadata["duration_s"],
        "dt_export": metadata["dt"],
        "final_state_preview": metadata.get("final_state_preview", {}),
        "observable_ranges": _array_stats(
            obs,
            ["vx", "vy", "r", "roll", "pitch", "sw_angle", "throttle_cmd", "brake_cmd"],
        ),
    }


def _array_stats(source: Dict[str, Any], keys: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for key in keys:
        if key not in source:
            continue
        arr = np.asarray(source[key], dtype=np.float64)
        stats[key] = {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }
    return stats


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
