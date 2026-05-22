from typing import Dict

import numpy as np

from simulator.vehicle_model.scenario import EnvironmentProfile
from simulator.vehicle_model.state import TeacherState
from simulator.vehicle_model.vehicle_params import VehicleConfig


class AeroModel:
    def step(
        self,
        state: TeacherState,
        environment: EnvironmentProfile,
        vehicle: VehicleConfig,
    ) -> Dict[str, np.ndarray]:
        hidden = vehicle.hidden_params
        rel_vx = state.vx - environment.wind_x_mps
        rel_vy = state.vy - environment.wind_y_mps
        speed = np.sqrt(rel_vx * rel_vx + rel_vy * rel_vy)
        drag = -0.5 * hidden["air_density"] * hidden["drag_coefficient_area"] * speed * rel_vx
        side = -0.08 * speed * rel_vy
        downforce = -0.03 * speed * speed
        return {
            "force_moment": np.array([drag, side, downforce, 0.0, 0.0, 0.02 * side], dtype=np.float64)
        }
