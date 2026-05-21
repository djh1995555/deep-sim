from typing import Dict

import numpy as np

from teacher_simulator.scenario import ActuatorProfile
from teacher_simulator.state import TeacherState
from teacher_simulator.vehicle_params import VehicleConfig


class SteeringActuator:
    def step(
        self,
        sw_angle: float,
        state: TeacherState,
        vehicle: VehicleConfig,
        actuator: ActuatorProfile,
        dt: float,
    ) -> Dict[str, np.ndarray]:
        hidden = vehicle.hidden_params
        ratio = vehicle.fixed_vehicle_context["steering_layout"][
            "steering_ratio_nominal"
        ]
        target = sw_angle / ratio
        backlash = float(hidden["steering_backlash_rad"])
        if abs(target - state.steering_delta[0]) < backlash:
            target = state.steering_delta[0]
        target *= float(hidden["steering_compliance_scale"])
        target = float(
            np.clip(
                target,
                -actuator.steering_saturation_rad,
                actuator.steering_saturation_rad,
            )
        )
        tau = max(float(hidden["steering_tau_true"]), 1e-3)
        alpha = dt / (tau + dt)
        state.steering_delta += alpha * (target - state.steering_delta)
        return {
            "delta_eff_i": state.steering_delta.copy(),
            "actuator_delay_states": np.array(
                [target, tau, backlash, alpha], dtype=np.float64
            ),
        }
