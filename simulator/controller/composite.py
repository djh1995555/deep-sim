from dataclasses import dataclass, field
from typing import Any, Dict

from simulator.controller.base import (
    ControllerInput,
    ControllerOutput,
    ControllerReference,
)
from simulator.controller.lateral_lqr import LateralLQRConfig, LateralLQRController
from simulator.controller.longitudinal_pid import (
    LongitudinalPIDConfig,
    LongitudinalPIDController,
)
from simulator.vehicle_model.scenario import ScenarioConfig


@dataclass
class PIDLQRControllerConfig:
    longitudinal: LongitudinalPIDConfig = field(default_factory=LongitudinalPIDConfig)
    lateral: LateralLQRConfig = field(default_factory=LateralLQRConfig)


class PIDLQRController:
    def __init__(self, config: PIDLQRControllerConfig | None = None) -> None:
        self.config = config or PIDLQRControllerConfig()
        self.longitudinal = LongitudinalPIDController(self.config.longitudinal)
        self.lateral = LateralLQRController(self.config.lateral)

    def reset(self, scenario: ScenarioConfig) -> None:
        self.longitudinal.reset(scenario)
        self.lateral.reset(scenario)

    def compute(self, controller_input: ControllerInput) -> ControllerOutput:
        long_debug = self.longitudinal.compute_accel_command(controller_input)
        pedals = self.longitudinal.command_to_pedals(long_debug["accel_cmd"])
        lat_debug = self.lateral.compute_steering(controller_input)
        debug: Dict[str, Any] = {
            "controller": "PIDLQRController",
            "reference": _reference_debug(controller_input.reference),
            "longitudinal": long_debug,
            "lateral": lat_debug,
        }
        return ControllerOutput(
            sw_angle=lat_debug["sw_angle"],
            steer_cmd=lat_debug["sw_angle"],
            throttle_cmd=pedals["throttle_cmd"],
            brake_cmd=pedals["brake_cmd"],
            debug=debug,
        )


def _reference_debug(reference: ControllerReference) -> Dict[str, Any]:
    return {
        "target_x_m": reference.target_x_m,
        "target_speed_mps": reference.target_speed_mps,
        "target_y_m": reference.target_y_m,
        "target_yaw_rad": reference.target_yaw_rad,
        "target_yaw_rate_rps": reference.target_yaw_rate_rps,
        "target_curvature_1pm": reference.target_curvature_1pm,
        "path_s_m": reference.path_s_m,
        "lookahead_distance_m": reference.lookahead_distance_m,
        "extra": reference.extra,
    }
