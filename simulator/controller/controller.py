from dataclasses import dataclass, field
from typing import Any, Dict

from simulator.controller.base import (
    ControllerInput,
    ControllerOutput,
    ControllerReference,
)
from simulator.controller.coupled_mpc import CoupledMPCConfig, CoupledMPCController
from simulator.controller.lateral_lqr import LateralLQRConfig, LateralLQRController
from simulator.controller.longitudinal_pid import (
    LongitudinalPIDConfig,
    LongitudinalPIDController,
)
from simulator.vehicle_model.scenario import ScenarioConfig


@dataclass
class SimulationControllerConfig:
    controller_type: str = "modular"
    longitudinal_type: str = "pid"
    lateral_type: str = "lqr"
    longitudinal: LongitudinalPIDConfig = field(default_factory=LongitudinalPIDConfig)
    lateral: LateralLQRConfig = field(default_factory=LateralLQRConfig)
    mpc: CoupledMPCConfig = field(default_factory=CoupledMPCConfig)


class SimulationController:
    def __init__(self, config: SimulationControllerConfig | None = None) -> None:
        self.config = config or SimulationControllerConfig()
        self.coupled = _build_coupled_controller(self.config)
        self.longitudinal = None
        self.lateral = None
        if self.coupled is None:
            self.longitudinal = _build_longitudinal_controller(self.config)
            self.lateral = _build_lateral_controller(self.config)

    def reset(self, scenario: ScenarioConfig) -> None:
        if self.coupled is not None:
            self.coupled.reset(scenario)
            return
        assert self.longitudinal is not None
        assert self.lateral is not None
        self.longitudinal.reset(scenario)
        self.lateral.reset(scenario)

    def compute(self, controller_input: ControllerInput) -> ControllerOutput:
        if self.coupled is not None:
            return self.coupled.compute(controller_input)
        assert self.longitudinal is not None
        assert self.lateral is not None
        long_debug = self.longitudinal.compute_accel_command(controller_input)
        pedals = self.longitudinal.command_to_pedals(long_debug["accel_cmd"])
        lat_debug = self.lateral.compute_steering(controller_input)
        debug: Dict[str, Any] = {
            "controller": "SimulationController",
            "controller_type": self.config.controller_type,
            "longitudinal_type": self.config.longitudinal_type,
            "lateral_type": self.config.lateral_type,
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


def _build_coupled_controller(
    config: SimulationControllerConfig,
) -> CoupledMPCController | None:
    if config.controller_type == "modular":
        return None
    if config.controller_type == "coupled_mpc":
        return CoupledMPCController(config.mpc)
    raise ValueError("unsupported simulation controller: %s" % config.controller_type)


def _build_longitudinal_controller(
    config: SimulationControllerConfig,
) -> LongitudinalPIDController:
    if config.longitudinal_type == "pid":
        return LongitudinalPIDController(config.longitudinal)
    raise ValueError(
        "unsupported longitudinal controller: %s" % config.longitudinal_type
    )


def _build_lateral_controller(
    config: SimulationControllerConfig,
) -> LateralLQRController:
    if config.lateral_type == "lqr":
        return LateralLQRController(config.lateral)
    raise ValueError("unsupported lateral controller: %s" % config.lateral_type)


def _reference_debug(reference: ControllerReference) -> Dict[str, Any]:
    return {
        "x_m": reference.x_m,
        "y_m": reference.y_m,
        "z_m": reference.z_m,
        "speed_mps": reference.speed_mps,
        "yaw_rad": reference.yaw_rad,
        "yaw_rate_rps": reference.yaw_rate_rps,
        "curvature_1pm": reference.curvature_1pm,
        "path_s_m": reference.path_s_m,
        "lookahead_distance_m": reference.lookahead_distance_m,
        "extra": reference.extra,
    }
