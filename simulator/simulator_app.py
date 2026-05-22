import json
import os
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from simulator.controller import (
    Controller,
    ControllerInput,
    ControllerReference,
    LateralLQRConfig,
    LongitudinalPIDConfig,
    PIDLQRController,
    PIDLQRControllerConfig,
)
from simulator.reference import build_reference_provider
from simulator.vehicle_model.config import load_dataset_config, load_yaml
from simulator.vehicle_model.export import export_dataset
from simulator.vehicle_model.model import VehicleModel, VehicleStepResult
from simulator.vehicle_model.scenario import (
    ROAD_MU,
    ActuatorProfile,
    ControlScript,
    EnvironmentProfile,
    RoadProfile,
    ScenarioConfig,
    SensorProfile,
    SplitMetadata,
)
from simulator.vehicle_model.vehicle_params import make_vehicle_config_variant
from simulator.visualizer import DebugTrace


@dataclass
class ClosedLoopSimulationRequest:
    dataset_config: str = "configs/datasets/ds0_minimal.yaml"
    out_dir: Optional[str] = None
    scenario_id: Optional[str] = None
    dataset_id: Optional[str] = None
    road: str = "single:dry"
    vehicle_index: int = 0
    seed: int = 0
    initial_speed_mps: float = 12.0
    duration_s: Optional[float] = None
    dt_internal: Optional[float] = None
    dt_export: Optional[float] = None
    split_role: str = "simulation"
    transition_start_s: float = 4.0
    transition_duration_s: float = 3.0
    road_grade_rad: float = 0.0
    road_bank_rad: float = 0.0
    road_roughness_amp_m: float = 0.003
    sensor_noise_scale: float = 1.0
    timestamp_jitter_std_s: float = 0.00015
    sensor_dropout_probability: float = 0.0
    sensor_quantization: float = 1e-5
    torque_observability_mode: str = "actuator_estimate"
    steering_saturation_rad: float = 0.62
    actuator_sensor_delay_steps: int = 1
    wind_x_mps: float = 0.0
    wind_y_mps: float = 0.5
    target_speed_mps: float = 14.0
    target_y_m: float = 0.0
    target_yaw_rad: float = 0.0
    target_yaw_rate_rps: float = 0.0
    reference: Optional[Dict[str, Any]] = None
    pid_kp: float = 0.18
    pid_ki: float = 0.025
    pid_kd: float = 0.02
    lqr_gains: Sequence[float] = (0.18, 0.85, 0.04, 0.16)
    lqr_max_sw_angle_rad: float = 1.2
    debug_stride: int = 1
    write_debug_trace: bool = True
    write_debug_html: bool = True
    debug_html_signals: Optional[Dict[str, Sequence[str]]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClosedLoopSimulationRequest":
        data = _normalize_request_dict(data)
        valid = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - valid)
        if unknown:
            raise ValueError("unknown simulator request fields: %s" % ", ".join(unknown))
        return cls(**data)

    def with_overrides(self, overrides: Dict[str, Any]) -> "ClosedLoopSimulationRequest":
        clean = {key: value for key, value in overrides.items() if value is not None}
        if not clean:
            return self
        return replace(self, **clean)

    def to_reference(self) -> ControllerReference:
        return ControllerReference(
            target_speed_mps=self.target_speed_mps,
            target_y_m=self.target_y_m,
            target_yaw_rad=self.target_yaw_rad,
            target_yaw_rate_rps=self.target_yaw_rate_rps,
        )


class SimulatorApp:
    def __init__(
        self,
        request: ClosedLoopSimulationRequest,
        controller: Optional[Controller] = None,
    ) -> None:
        self.request = request
        self.config = load_dataset_config(request.dataset_config)
        if request.duration_s is not None:
            self.config.duration_s = float(request.duration_s)
        if request.dt_internal is not None:
            self.config.dt_internal = float(request.dt_internal)
        if request.dt_export is not None:
            self.config.dt_export = float(request.dt_export)
        self.config.validate()
        self.vehicle_model = VehicleModel(self.config)
        self.scenario = build_scenario(request)
        fallback_reference = request.to_reference()
        self.reference_provider = build_reference_provider(
            request.reference,
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
        episode["metadata"]["control_mode"] = "closed_loop_pid_lqr"
        episode["metadata"]["controller"] = {
            "type": self.controller.__class__.__name__,
            "reference_provider": self.reference_provider.describe(),
        }
        return self._export(episode)

    def _export(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        out_dir = self.request.out_dir or _default_out_dir(self.scenario.scenario_id)
        dataset_id = self.request.dataset_id or "SIM_%s" % _safe_stem(
            self.scenario.scenario_id
        )
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
    return ClosedLoopSimulationRequest.from_dict(load_yaml(path))


def _normalize_request_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(data)
    if "teacher_config" in normalized and "dataset_config" not in normalized:
        normalized["dataset_config"] = normalized.pop("teacher_config")
    return normalized


def run_closed_loop_simulation(
    request: ClosedLoopSimulationRequest,
    controller: Optional[Controller] = None,
) -> Dict[str, Any]:
    return SimulatorApp(request, controller=controller).run()


def build_scenario(request: ClosedLoopSimulationRequest) -> ScenarioConfig:
    vehicle = make_vehicle_config_variant(request.vehicle_index)
    road = build_road_profile(request)
    scenario_id = request.scenario_id or _default_scenario_id(request, road)
    return ScenarioConfig(
        scenario_id=scenario_id,
        vehicle_config=vehicle,
        road_profile=road,
        control_script=ControlScript(
            longitudinal="closed_loop_speed_pid",
            lateral="closed_loop_lqr",
            initial_speed_mps=request.initial_speed_mps,
            longitudinal_id="CL-PID",
            lateral_id="CL-LQR",
        ),
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
        split_metadata=SplitMetadata(split_role=request.split_role),
        validation_case="closed_loop_simulation",
        seed=request.seed,
        scenario_group="SIM-CLOSED-LOOP",
        longitudinal_factor_id="CL-PID",
        lateral_factor_id="CL-LQR",
    )


def build_road_profile(request: ClosedLoopSimulationRequest) -> RoadProfile:
    road_spec = request.road.strip()
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
        "transition_start_s": request.transition_start_s,
        "transition_duration_s": request.transition_duration_s,
        "grade_rad": request.road_grade_rad,
        "bank_rad": request.road_bank_rad,
        "roughness_amp_m": request.road_roughness_amp_m,
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
        % request.road
    )


def _default_controller(request: ClosedLoopSimulationRequest) -> PIDLQRController:
    gains = tuple(float(x) for x in request.lqr_gains)
    if len(gains) != 4:
        raise ValueError("lqr_gains must contain 4 values")
    return PIDLQRController(
        PIDLQRControllerConfig(
            longitudinal=LongitudinalPIDConfig(
                kp=request.pid_kp,
                ki=request.pid_ki,
                kd=request.pid_kd,
            ),
            lateral=LateralLQRConfig(
                gains=gains,
                max_sw_angle_rad=request.lqr_max_sw_angle_rad,
            ),
        )
    )


def _validate_surface(surface: str) -> None:
    if surface not in ROAD_MU:
        raise ValueError(
            "unsupported road surface '%s'; supported: %s"
            % (surface, ", ".join(sorted(ROAD_MU)))
        )


def _default_scenario_id(request: ClosedLoopSimulationRequest, road: RoadProfile) -> str:
    return "closed_loop_v%d_%s_speed_pid_lqr" % (
        request.vehicle_index,
        road.factor_id,
    )


def _default_out_dir(scenario_id: str) -> str:
    return os.path.join("output", "simulation", "SIM001_%s" % _safe_stem(scenario_id))


def _safe_stem(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in ["_", "-", "."]:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep)


def _vehicle_debug(result: VehicleStepResult) -> Dict[str, Any]:
    return {
        "vx": float(result.state.vx),
        "vy": float(result.state.vy),
        "x_world": float(result.state.x_world),
        "y_world": float(result.state.y_world),
        "yaw": float(result.state.yaw),
        "r": float(result.state.r),
        "mu_min": float(np.min(result.tire["mu_true_i"])),
        "friction_usage_max": float(np.max(result.tire["friction_usage_i"])),
    }


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
