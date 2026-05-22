from dataclasses import dataclass, fields
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from simulator.controller.base import (
    ControllerInput,
    ControllerOutput,
    ControllerReference,
)
from simulator.vehicle_model.modules.road import RoadModel
from simulator.vehicle_model.scenario import ScenarioConfig
from simulator.vehicle_model.state import TeacherState


@dataclass
class CoupledMPCConfig:
    horizon_steps: int = 10
    prediction_dt: float = 0.08
    accel_min_mps2: float = -5.0
    accel_max_mps2: float = 2.5
    accel_span_mps2: float = 3.0
    accel_samples: int = 7
    max_sw_angle_rad: float = 1.2
    steering_span_rad: float = 1.2
    steering_samples: int = 9
    speed_weight: float = 1.0
    lateral_weight: float = 4.0
    heading_weight: float = 3.0
    longitudinal_weight: float = 0.05
    accel_effort_weight: float = 0.04
    steering_effort_weight: float = 0.03
    accel_rate_weight: float = 0.08
    steering_rate_weight: float = 0.12
    friction_excess_weight: float = 20.0
    terminal_weight: float = 2.0
    lateral_feedback_gain: float = 0.25
    heading_feedback_gain: float = 1.1
    speed_feedback_gain: float = 0.4
    throttle_feedforward: float = 0.035
    accel_to_throttle: float = 0.35
    decel_to_brake: float = 0.45
    friction_safety_factor: float = 0.85

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "CoupledMPCConfig":
        if not data:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("coupled MPC config must be a mapping")
        valid = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - valid)
        if unknown:
            raise ValueError(
                "unknown coupled MPC config fields: %s" % ", ".join(unknown)
            )
        return cls(**data)


class CoupledMPCController:
    def __init__(self, config: CoupledMPCConfig | None = None) -> None:
        self.config = config or CoupledMPCConfig()
        self.road = RoadModel()
        self.prev_accel_cmd = 0.0
        self.prev_sw_angle = 0.0

    def reset(self, scenario: ScenarioConfig) -> None:
        self.prev_accel_cmd = 0.0
        self.prev_sw_angle = 0.0

    def compute(self, controller_input: ControllerInput) -> ControllerOutput:
        cfg = self.config
        accel_center = self._accel_center(controller_input)
        steering_center = self._steering_center(controller_input)
        accel_candidates = _candidate_grid(
            center=accel_center,
            lower=cfg.accel_min_mps2,
            upper=cfg.accel_max_mps2,
            span=cfg.accel_span_mps2,
            samples=cfg.accel_samples,
            extra=(0.0, self.prev_accel_cmd),
        )
        steering_candidates = _candidate_grid(
            center=steering_center,
            lower=-cfg.max_sw_angle_rad,
            upper=cfg.max_sw_angle_rad,
            span=cfg.steering_span_rad,
            samples=cfg.steering_samples,
            extra=(0.0, self.prev_sw_angle),
        )

        best_cost = float("inf")
        best_accel = accel_center
        best_sw_angle = steering_center
        best_debug: Dict[str, float] = {}
        for accel_cmd in accel_candidates:
            for sw_angle in steering_candidates:
                cost, rollout_debug = self._rollout_cost(
                    controller_input,
                    float(accel_cmd),
                    float(sw_angle),
                )
                if cost < best_cost:
                    best_cost = cost
                    best_accel = float(accel_cmd)
                    best_sw_angle = float(sw_angle)
                    best_debug = rollout_debug

        pedals = self._accel_to_pedals(best_accel)
        self.prev_accel_cmd = best_accel
        self.prev_sw_angle = best_sw_angle
        debug = {
            "controller": "CoupledMPCController",
            "horizon_steps": cfg.horizon_steps,
            "prediction_dt": cfg.prediction_dt,
            "candidate_count": len(accel_candidates) * len(steering_candidates),
            "best_cost": best_cost,
            "accel_cmd": best_accel,
            "sw_angle": best_sw_angle,
            "accel_center": accel_center,
            "steering_center": steering_center,
            **best_debug,
        }
        return ControllerOutput(
            sw_angle=best_sw_angle,
            steer_cmd=best_sw_angle,
            throttle_cmd=pedals["throttle_cmd"],
            brake_cmd=pedals["brake_cmd"],
            debug=debug,
        )

    def _accel_center(self, controller_input: ControllerInput) -> float:
        cfg = self.config
        error = controller_input.reference.speed_mps - controller_input.state.vx
        return float(
            np.clip(
                cfg.speed_feedback_gain * error,
                cfg.accel_min_mps2,
                cfg.accel_max_mps2,
            )
        )

    def _steering_center(self, controller_input: ControllerInput) -> float:
        cfg = self.config
        state = controller_input.state
        reference = controller_input.reference
        vehicle = controller_input.scenario.vehicle_config
        steering_ratio = _steering_ratio(controller_input.scenario)
        delta_ff = np.arctan(vehicle.wheelbase * reference.curvature_1pm)
        sw_ff = steering_ratio * delta_ff
        _, e_lat, e_yaw = _tracking_errors(
            state.x_world,
            state.y_world,
            state.yaw,
            reference.x_m,
            reference.y_m,
            reference.yaw_rad,
        )
        feedback = (
            -cfg.lateral_feedback_gain * e_lat
            - cfg.heading_feedback_gain * e_yaw
        )
        return float(
            np.clip(sw_ff + feedback, -cfg.max_sw_angle_rad, cfg.max_sw_angle_rad)
        )

    def _rollout_cost(
        self,
        controller_input: ControllerInput,
        accel_cmd: float,
        sw_angle: float,
    ) -> Tuple[float, Dict[str, float]]:
        cfg = self.config
        scenario = controller_input.scenario
        vehicle = scenario.vehicle_config
        reference = controller_input.reference
        steering_ratio = _steering_ratio(scenario)
        delta = float(np.clip(sw_angle / steering_ratio, -0.75, 0.75))
        x = float(controller_input.state.x_world)
        y = float(controller_input.state.y_world)
        yaw = float(controller_input.state.yaw)
        speed = max(float(controller_input.state.vx), 0.05)
        ref_x = float(reference.x_m)
        ref_y = float(reference.y_m)
        ref_yaw = float(reference.yaw_rad)
        ref_speed = max(float(reference.speed_mps), 0.0)
        ref_yaw_rate = _reference_yaw_rate(reference)
        cost = 0.0
        max_abs_lateral_error = 0.0
        max_friction_usage_proxy = 0.0
        terminal_cost = 0.0

        for step in range(max(1, int(cfg.horizon_steps))):
            speed = float(
                np.clip(
                    speed + accel_cmd * cfg.prediction_dt,
                    0.05,
                    max(ref_speed + 15.0, 35.0),
                )
            )
            yaw_rate = speed / max(vehicle.wheelbase, 1e-6) * np.tan(delta)
            yaw = _wrap_angle(yaw + yaw_rate * cfg.prediction_dt)
            x += speed * np.cos(yaw) * cfg.prediction_dt
            y += speed * np.sin(yaw) * cfg.prediction_dt

            ref_yaw = _wrap_angle(ref_yaw + ref_yaw_rate * cfg.prediction_dt)
            ref_x += ref_speed * np.cos(ref_yaw) * cfg.prediction_dt
            ref_y += ref_speed * np.sin(ref_yaw) * cfg.prediction_dt

            e_long, e_lat, e_yaw = _tracking_errors(
                x,
                y,
                yaw,
                ref_x,
                ref_y,
                ref_yaw,
            )
            speed_error = speed - ref_speed
            lateral_acc = (
                speed * speed * np.tan(delta) / max(vehicle.wheelbase, 1e-6)
            )
            friction_usage = self._friction_usage_proxy(
                controller_input.t + (step + 1) * cfg.prediction_dt,
                x,
                y,
                yaw,
                accel_cmd,
                lateral_acc,
                scenario,
            )
            step_cost = (
                cfg.speed_weight * speed_error * speed_error
                + cfg.lateral_weight * e_lat * e_lat
                + cfg.heading_weight * e_yaw * e_yaw
                + cfg.longitudinal_weight * e_long * e_long
                + cfg.accel_effort_weight * accel_cmd * accel_cmd
                + cfg.steering_effort_weight * sw_angle * sw_angle
                + cfg.friction_excess_weight * max(0.0, friction_usage - 1.0) ** 2
            )
            cost += step_cost
            terminal_cost = step_cost
            max_abs_lateral_error = max(max_abs_lateral_error, abs(e_lat))
            max_friction_usage_proxy = max(max_friction_usage_proxy, friction_usage)

        accel_rate = accel_cmd - self.prev_accel_cmd
        steering_rate = sw_angle - self.prev_sw_angle
        cost += cfg.accel_rate_weight * accel_rate * accel_rate
        cost += cfg.steering_rate_weight * steering_rate * steering_rate
        cost += cfg.terminal_weight * terminal_cost
        return float(cost), {
            "predicted_max_abs_lateral_error_m": float(max_abs_lateral_error),
            "predicted_max_friction_usage": float(max_friction_usage_proxy),
        }

    def _friction_usage_proxy(
        self,
        t: float,
        x: float,
        y: float,
        yaw: float,
        accel_cmd: float,
        lateral_acc: float,
        scenario: ScenarioConfig,
    ) -> float:
        cfg = self.config
        probe_state = TeacherState(x_world=x, y_world=y, yaw=yaw)
        road = self.road.query(
            t,
            scenario.road_profile,
            probe_state,
            scenario.vehicle_config,
        )
        mu = max(float(np.min(road["mu"])), 0.03)
        limit = max(mu * 9.81 * cfg.friction_safety_factor, 1e-3)
        return float(
            np.sqrt((accel_cmd / limit) ** 2 + (lateral_acc / limit) ** 2)
        )

    def _accel_to_pedals(self, accel_cmd: float) -> Dict[str, float]:
        cfg = self.config
        if accel_cmd >= 0.0:
            throttle = cfg.throttle_feedforward + cfg.accel_to_throttle * accel_cmd
            brake = 0.0
        else:
            throttle = 0.0
            brake = cfg.decel_to_brake * (-accel_cmd)
        return {
            "throttle_cmd": float(np.clip(throttle, 0.0, 1.0)),
            "brake_cmd": float(np.clip(brake, 0.0, 1.0)),
        }


def _candidate_grid(
    center: float,
    lower: float,
    upper: float,
    span: float,
    samples: int,
    extra: Sequence[float],
) -> np.ndarray:
    samples = max(1, int(samples))
    lo = max(float(lower), float(center) - float(span))
    hi = min(float(upper), float(center) + float(span))
    grid = np.linspace(lo, hi, samples, dtype=np.float64)
    values = [*grid.tolist(), float(center), *[float(value) for value in extra]]
    clipped = [float(np.clip(value, lower, upper)) for value in values]
    return np.unique(np.round(np.asarray(clipped, dtype=np.float64), decimals=8))


def _tracking_errors(
    x: float,
    y: float,
    yaw: float,
    ref_x: float,
    ref_y: float,
    ref_yaw: float,
) -> Tuple[float, float, float]:
    dx = x - ref_x
    dy = y - ref_y
    cos_yaw = np.cos(ref_yaw)
    sin_yaw = np.sin(ref_yaw)
    e_long = cos_yaw * dx + sin_yaw * dy
    e_lat = -sin_yaw * dx + cos_yaw * dy
    e_yaw = _wrap_angle(yaw - ref_yaw)
    return float(e_long), float(e_lat), float(e_yaw)


def _reference_yaw_rate(reference: ControllerReference) -> float:
    if abs(reference.yaw_rate_rps) > 1e-9:
        return float(reference.yaw_rate_rps)
    return float(reference.speed_mps * reference.curvature_1pm)


def _steering_ratio(scenario: ScenarioConfig) -> float:
    layout = scenario.vehicle_config.fixed_vehicle_context["steering_layout"]
    return max(float(layout["steering_ratio_nominal"]), 1e-6)


def _wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)
