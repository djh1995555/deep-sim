from typing import Dict

import numpy as np

from simulator.vehicle_model.vehicle_params import VehicleConfig


class SuspensionModel:
    def step(
        self,
        ax: float,
        ay: float,
        road_contact: Dict[str, object],
        vehicle: VehicleConfig,
        gravity: float,
    ) -> Dict[str, np.ndarray]:
        mass = vehicle.mass
        wheelbase = vehicle.wheelbase
        cg_z = vehicle.cg_z
        b = vehicle.cg_x
        a = wheelbase - b
        front_static = mass * gravity * b / wheelbase
        rear_static = mass * gravity * a / wheelbase
        fz = np.array(
            [
                front_static / 2.0,
                front_static / 2.0,
                rear_static / 2.0,
                rear_static / 2.0,
            ],
            dtype=np.float64,
        )

        d_long = -mass * ax * cg_z / wheelbase
        d_long *= float(vehicle.hidden_params.get("suspension_longitudinal_transfer_scale", 1.0))
        fz[0:2] += d_long / 2.0
        fz[2:4] -= d_long / 2.0

        track_front = vehicle.track_front
        track_rear = vehicle.track_rear
        d_lat_front = mass * ay * cg_z * 0.55 / max(track_front, 1e-6)
        d_lat_rear = mass * ay * cg_z * 0.45 / max(track_rear, 1e-6)
        lat_scale = float(vehicle.hidden_params.get("suspension_lateral_transfer_scale", 1.0))
        d_lat_front *= lat_scale
        d_lat_rear *= lat_scale
        fz[0] -= d_lat_front / 2.0
        fz[1] += d_lat_front / 2.0
        fz[2] -= d_lat_rear / 2.0
        fz[3] += d_lat_rear / 2.0
        fz = np.maximum(fz, 20.0)

        height = np.asarray(road_contact["height"], dtype=np.float64)
        suspension_disp = -height[:, None] * np.array([[1.0, 0.0]])
        unsprung = np.column_stack([height, np.zeros(4, dtype=np.float64)])
        return {
            "Fz_true_i": fz,
            "suspension_states": suspension_disp,
            "unsprung_states": unsprung,
            "camber_true_i": np.array(
                [-0.006, -0.006, -0.004, -0.004], dtype=np.float64
            )
            - 0.03 * height,
            "toe_true_i": np.array([0.001, -0.001, 0.0, 0.0], dtype=np.float64),
            "load_transfer_diagnostics": np.array(
                [d_long, d_lat_front, d_lat_rear], dtype=np.float64
            ),
        }
