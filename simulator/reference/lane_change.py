from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from simulator.controller.base import ControllerReference
from simulator.vehicle_model.scenario import ScenarioConfig
from simulator.vehicle_model.state import TeacherState


@dataclass
class LaneChangeReferenceConfig:
    speed_mps: float = 14.0
    start_y_m: float = 0.0
    end_y_m: float = 3.5
    start_x_m: float = 20.0
    length_m: float = 35.0
    start_time_s: float = 0.0
    duration_s: float = 3.0
    mode: str = "spatial"


class LaneChangeReferenceProvider:
    def __init__(self, config: LaneChangeReferenceConfig) -> None:
        self.config = config

    def reset(self, scenario: ScenarioConfig) -> None:
        pass

    def query(self, t: float, state: TeacherState) -> ControllerReference:
        cfg = self.config
        if cfg.mode == "temporal":
            progress = (t - cfg.start_time_s) / max(cfg.duration_s, 1e-6)
            denominator = max(cfg.duration_s, 1e-6)
        else:
            progress = (state.x_world - cfg.start_x_m) / max(cfg.length_m, 1e-6)
            denominator = max(cfg.length_m, 1e-6)
        p = float(np.clip(progress, 0.0, 1.0))
        smooth = _smoothstep(p)
        slope = _smoothstep_derivative(p) * (cfg.end_y_m - cfg.start_y_m) / denominator
        target_y = cfg.start_y_m + (cfg.end_y_m - cfg.start_y_m) * smooth
        if cfg.mode == "temporal":
            yaw = float(np.arctan2(slope, max(cfg.speed_mps, 1e-6)))
            yaw_rate = 0.0
        else:
            yaw = float(np.arctan(slope))
            yaw_rate = 0.0
        return ControllerReference(
            x_m=float(state.x_world),
            y_m=float(target_y),
            speed_mps=float(cfg.speed_mps),
            yaw_rad=yaw,
            yaw_rate_rps=yaw_rate,
            curvature_1pm=0.0,
            path_s_m=float(max(0.0, state.x_world - cfg.start_x_m)),
            lookahead_distance_m=0.0,
            extra={
                "provider": "lane_change",
                "progress": p,
                "mode": cfg.mode,
            },
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "lane_change",
            "config": self.config.__dict__,
        }


def _smoothstep(p: float) -> float:
    return p * p * (3.0 - 2.0 * p)


def _smoothstep_derivative(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return 6.0 * p * (1.0 - p)
