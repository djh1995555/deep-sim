from typing import Dict

import numpy as np

from simulator.vehicle_model.scenario import ActuatorProfile
from simulator.vehicle_model.state import TeacherState
from simulator.vehicle_model.vehicle_params import WHEEL_ORDER, VehicleConfig


class DriveBrakeActuator:
    def step(
        self,
        throttle_cmd: float,
        brake_cmd: float,
        state: TeacherState,
        vehicle: VehicleConfig,
        actuator: ActuatorProfile,
        dt: float,
    ) -> Dict[str, np.ndarray]:
        hidden = vehicle.hidden_params
        drive_target = np.zeros(4, dtype=np.float64)
        for idx, wheel in enumerate(WHEEL_ORDER):
            if wheel in vehicle.driven_wheels:
                drive_target[idx] = (
                    throttle_cmd * float(hidden["max_drive_torque_per_wheel"])
                )

        brake_target = np.zeros(4, dtype=np.float64)
        front_bias = float(hidden["brake_bias_front"])
        for idx, wheel in enumerate(WHEEL_ORDER):
            axle_scale = front_bias / 0.5 if wheel in ["FL", "FR"] else (1.0 - front_bias) / 0.5
            brake_target[idx] = (
                brake_cmd
                * float(hidden["max_brake_torque_per_wheel"])
                * axle_scale
            )

        drive_alpha = dt / (float(hidden["drive_tau_true"]) + dt)
        brake_alpha = dt / (float(hidden["brake_tau_true"]) + dt)
        state.drive_torque += drive_alpha * (drive_target - state.drive_torque)
        state.brake_torque += brake_alpha * (brake_target - state.brake_torque)

        brake_pressure = state.brake_torque / max(
            float(hidden["max_brake_torque_per_wheel"]), 1e-6
        )
        brake_temperature = 35.0 + 28.0 * brake_pressure + 2.0 * np.abs(state.omega)
        modulation = np.clip(brake_pressure - 0.82, 0.0, 1.0)
        return {
            "tau_drv_true_i": state.drive_torque.copy(),
            "tau_brk_true_i": state.brake_torque.copy(),
            "brake_pressure_i": brake_pressure.copy(),
            "brake_temperature_i": brake_temperature.copy(),
            "abs_tcs_esc_modulation_states": modulation.copy(),
        }
