from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from simulator.controller.base import ControllerReference
from simulator.vehicle_model.scenario import ScenarioConfig
from simulator.vehicle_model.state import TeacherState


@dataclass
class SinusoidalReferenceConfig:
    speed_mps: float = 5.0
    center_y_m: float = 0.0
    amplitude_m: float = 5.0
    period_m: float = 20.0
    start_x_m: float = 0.0
    phase_rad: float = 0.0
    mode: str = "spatial"


class SinusoidalReferenceProvider:
    def __init__(self, config: SinusoidalReferenceConfig) -> None:
        if config.period_m <= 0.0:
            raise ValueError("sinusoidal reference period_m must be > 0")
        if config.mode != "spatial":
            raise ValueError("sinusoidal reference currently supports spatial mode only")
        self.config = config

    def reset(self, scenario: ScenarioConfig) -> None:
        pass

    def query(self, t: float, state: TeacherState) -> ControllerReference:
        cfg = self.config
        distance_m = state.x_world - cfg.start_x_m
        wave_number = 2.0 * np.pi / cfg.period_m
        theta = wave_number * distance_m + cfg.phase_rad
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        target_y = cfg.center_y_m + cfg.amplitude_m * sin_theta
        slope = cfg.amplitude_m * wave_number * cos_theta
        second_derivative = -cfg.amplitude_m * wave_number * wave_number * sin_theta
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
                "provider": "sinusoidal",
                "mode": cfg.mode,
                "theta_rad": float(theta),
                "period_m": float(cfg.period_m),
                "amplitude_m": float(cfg.amplitude_m),
            },
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "sinusoidal",
            "config": self.config.__dict__,
        }
