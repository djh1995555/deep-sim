from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from simulator.controller.base import ControllerReference
from simulator.vehicle_model.scenario import ScenarioConfig
from simulator.vehicle_model.state import TeacherState


@dataclass
class DoubleLaneChangeReferenceConfig:
    speed_mps: float = 5.0
    start_y_m: float = 0.0
    offset_y_m: float = 3.5
    end_y_m: float = 0.0
    start_x_m: float = 5.0
    first_length_m: float = 18.0
    hold_length_m: float = 8.0
    second_length_m: float = 18.0
    start_time_s: float = 0.0
    mode: str = "spatial"


class DoubleLaneChangeReferenceProvider:
    def __init__(self, config: DoubleLaneChangeReferenceConfig) -> None:
        self.config = config

    def reset(self, scenario: ScenarioConfig) -> None:
        pass

    def query(self, t: float, state: TeacherState) -> ControllerReference:
        cfg = self.config
        if cfg.mode == "temporal":
            distance_m = max(0.0, (t - cfg.start_time_s) * cfg.speed_mps)
        else:
            distance_m = state.x_world - cfg.start_x_m
        target_y, slope, second_derivative, phase, progress = _profile_at_distance(
            distance_m,
            cfg,
        )
        yaw = float(np.arctan(slope))
        curvature = float(second_derivative / np.power(1.0 + slope * slope, 1.5))
        return ControllerReference(
            x_m=float(state.x_world),
            y_m=float(target_y),
            speed_mps=float(cfg.speed_mps),
            yaw_rad=yaw,
            yaw_rate_rps=float(curvature * cfg.speed_mps),
            curvature_1pm=curvature,
            path_s_m=float(max(0.0, distance_m)),
            lookahead_distance_m=0.0,
            extra={
                "provider": "double_lane_change",
                "phase": phase,
                "progress": progress,
                "mode": cfg.mode,
            },
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "double_lane_change",
            "config": self.config.__dict__,
        }


def _profile_at_distance(
    distance_m: float,
    cfg: DoubleLaneChangeReferenceConfig,
) -> tuple[float, float, float, str, float]:
    first_len = max(cfg.first_length_m, 1e-6)
    hold_len = max(cfg.hold_length_m, 0.0)
    second_len = max(cfg.second_length_m, 1e-6)
    peak_y = cfg.start_y_m + cfg.offset_y_m

    if distance_m <= 0.0:
        return cfg.start_y_m, 0.0, 0.0, "before", 0.0
    if distance_m < first_len:
        p = float(np.clip(distance_m / first_len, 0.0, 1.0))
        y, slope, second = _smooth_segment(cfg.start_y_m, peak_y, first_len, p)
        return y, slope, second, "first_shift", p
    hold_end = first_len + hold_len
    if distance_m < hold_end:
        p = float(np.clip((distance_m - first_len) / max(hold_len, 1e-6), 0.0, 1.0))
        return peak_y, 0.0, 0.0, "hold", p
    second_end = hold_end + second_len
    if distance_m < second_end:
        p = float(np.clip((distance_m - hold_end) / second_len, 0.0, 1.0))
        y, slope, second = _smooth_segment(peak_y, cfg.end_y_m, second_len, p)
        return y, slope, second, "second_shift", p
    return cfg.end_y_m, 0.0, 0.0, "after", 1.0


def _smooth_segment(
    y0: float,
    y1: float,
    length_m: float,
    p: float,
) -> tuple[float, float, float]:
    delta = y1 - y0
    y = y0 + delta * _smoothstep(p)
    slope = delta * _smoothstep_derivative(p) / length_m
    second_derivative = delta * _smoothstep_second_derivative(p) / (length_m * length_m)
    return float(y), float(slope), float(second_derivative)


def _smoothstep(p: float) -> float:
    return p * p * (3.0 - 2.0 * p)


def _smoothstep_derivative(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return 6.0 * p * (1.0 - p)


def _smoothstep_second_derivative(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return 6.0 - 12.0 * p
