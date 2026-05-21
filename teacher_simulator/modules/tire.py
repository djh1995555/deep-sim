from typing import Dict

import numpy as np

from teacher_simulator.state import TeacherState
from teacher_simulator.vehicle_params import VehicleConfig


class TireModel:
    def step(
        self,
        state: TeacherState,
        steering: Dict[str, np.ndarray],
        torques: Dict[str, np.ndarray],
        road_contact: Dict[str, object],
        load_state: Dict[str, np.ndarray],
        vehicle: VehicleConfig,
        min_speed_for_slip: float,
    ) -> Dict[str, np.ndarray]:
        wheelbase = vehicle.wheelbase
        b = vehicle.cg_x
        a = wheelbase - b
        pos = np.array(
            [
                [a, vehicle.track_front / 2.0],
                [a, -vehicle.track_front / 2.0],
                [-b, vehicle.track_rear / 2.0],
                [-b, -vehicle.track_rear / 2.0],
            ],
            dtype=np.float64,
        )
        deltas = np.array(
            [steering["delta_eff_i"][0], steering["delta_eff_i"][1], 0.0, 0.0],
            dtype=np.float64,
        )
        fz = load_state["Fz_true_i"]
        mu = np.asarray(road_contact["mu"], dtype=np.float64)
        tau_drv = torques["tau_drv_true_i"]
        tau_brk = torques["tau_brk_true_i"]
        radius = vehicle.wheel_radius
        fx_body = np.zeros(4, dtype=np.float64)
        fy_body = np.zeros(4, dtype=np.float64)
        fx_wheel = np.zeros(4, dtype=np.float64)
        fy_wheel = np.zeros(4, dtype=np.float64)
        slip_ratio = np.zeros(4, dtype=np.float64)
        slip_angle = np.zeros(4, dtype=np.float64)
        friction_usage = np.zeros(4, dtype=np.float64)

        for i in range(4):
            x_i, y_i = pos[i]
            vx_i = state.vx - state.r * y_i
            vy_i = state.vy + state.r * x_i
            delta = deltas[i]
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            vx_w = vx_i * cos_d + vy_i * sin_d
            vy_w = -vx_i * sin_d + vy_i * cos_d
            denom = max(abs(vx_w), min_speed_for_slip)
            slip_ratio[i] = (radius * state.omega[i] - vx_w) / denom
            slip_angle[i] = delta - np.arctan2(vy_i, max(abs(vx_i), min_speed_for_slip))

            fx_cmd = tau_drv[i] / radius - tau_brk[i] / radius
            cornering = (
                vehicle.hidden_params["cornering_stiffness_front"]
                if i < 2
                else vehicle.hidden_params["cornering_stiffness_rear"]
            )
            fy_raw = cornering * slip_angle[i] * (fz[i] / 4500.0)
            cap = max(mu[i] * fz[i], 1.0)
            fx = np.clip(fx_cmd, -0.98 * cap, 0.98 * cap)
            fy = fy_raw
            usage = np.sqrt((fx / cap) ** 2 + (fy / cap) ** 2)
            if usage > 1.0:
                fx /= usage
                fy /= usage
                usage = 1.0
            fx_wheel[i] = fx
            fy_wheel[i] = fy
            fx_body[i] = fx * cos_d - fy * sin_d
            fy_body[i] = fx * sin_d + fy * cos_d
            friction_usage[i] = usage

        mz = pos[:, 0] * fy_body - pos[:, 1] * fx_body
        return {
            "Fx_true_i": fx_body,
            "Fy_true_i": fy_body,
            "Fx_wheel_i": fx_wheel,
            "Fy_wheel_i": fy_wheel,
            "Mz_true_i": mz,
            "mu_true_i": mu,
            "slip_ratio_true_i": slip_ratio,
            "slip_angle_true_i": slip_angle,
            "friction_usage_i": friction_usage,
            "wheel_positions_body": pos,
        }
