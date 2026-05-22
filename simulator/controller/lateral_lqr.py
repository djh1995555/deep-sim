from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from simulator.controller.base import ControllerInput
from simulator.vehicle_model.scenario import ScenarioConfig


@dataclass
class LateralLQRConfig:
    gains: Tuple[float, float, float, float] = (0.18, 0.85, 0.04, 0.16)
    max_sw_angle_rad: float = 1.2


class LateralLQRController:
    def __init__(self, config: LateralLQRConfig | None = None) -> None:
        self.config = config or LateralLQRConfig()

    def reset(self, scenario: ScenarioConfig) -> None:
        pass

    def compute_steering(self, controller_input: ControllerInput) -> Dict[str, float]:
        state = controller_input.state
        ref = controller_input.reference
        e_y = state.y_world - ref.y_m
        e_yaw = _wrap_angle(state.yaw - ref.yaw_rad)
        e_vy = state.vy
        e_r = state.r - ref.yaw_rate_rps
        k_y, k_yaw, k_vy, k_r = self.config.gains
        raw = -(k_y * e_y + k_yaw * e_yaw + k_vy * e_vy + k_r * e_r)
        sw_angle = float(
            np.clip(raw, -self.config.max_sw_angle_rad, self.config.max_sw_angle_rad)
        )
        return {
            "sw_angle": sw_angle,
            "lateral_error_m": float(e_y),
            "heading_error_rad": float(e_yaw),
            "vy_error_mps": float(e_vy),
            "yaw_rate_error_rps": float(e_r),
            "raw_sw_angle": float(raw),
        }


def _wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)
