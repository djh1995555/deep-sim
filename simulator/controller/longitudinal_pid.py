from dataclasses import dataclass
from typing import Dict

import numpy as np

from simulator.controller.base import ControllerInput
from simulator.vehicle_model.scenario import ScenarioConfig


@dataclass
class LongitudinalPIDConfig:
    kp: float = 0.18
    ki: float = 0.025
    kd: float = 0.02
    integral_limit: float = 12.0
    accel_to_throttle: float = 0.35
    decel_to_brake: float = 0.45
    throttle_feedforward: float = 0.035


class LongitudinalPIDController:
    def __init__(self, config: LongitudinalPIDConfig | None = None) -> None:
        self.config = config or LongitudinalPIDConfig()
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def reset(self, scenario: ScenarioConfig) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def compute_accel_command(self, controller_input: ControllerInput) -> Dict[str, float]:
        cfg = self.config
        error = controller_input.reference.target_speed_mps - controller_input.state.vx
        self.integral = float(
            np.clip(
                self.integral + error * controller_input.dt,
                -cfg.integral_limit,
                cfg.integral_limit,
            )
        )
        derivative = 0.0
        if self.initialized and controller_input.dt > 0:
            derivative = (error - self.prev_error) / controller_input.dt
        self.prev_error = error
        self.initialized = True
        accel_cmd = cfg.kp * error + cfg.ki * self.integral + cfg.kd * derivative
        return {
            "speed_error": float(error),
            "speed_error_integral": float(self.integral),
            "speed_error_derivative": float(derivative),
            "accel_cmd": float(accel_cmd),
        }

    def command_to_pedals(self, accel_cmd: float) -> Dict[str, float]:
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
