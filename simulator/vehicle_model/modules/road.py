from typing import Dict, List

import numpy as np

from simulator.vehicle_model.scenario import ROAD_MU, RoadProfile
from simulator.vehicle_model.state import TeacherState
from simulator.vehicle_model.vehicle_params import WHEEL_ORDER, VehicleConfig


class RoadModel:
    def query(
        self,
        t: float,
        road_profile: RoadProfile,
        state: TeacherState | None = None,
        vehicle: VehicleConfig | None = None,
    ) -> Dict[str, object]:
        wheel_xy = (
            self.wheel_world_xy(state, vehicle)
            if state is not None and vehicle is not None
            else self._default_wheel_xy()
        )
        if road_profile.road_type == "single":
            labels = [str(road_profile.single_surface)] * 4
            mu = np.array([ROAD_MU[label] for label in labels], dtype=np.float64)
        elif road_profile.road_type == "split":
            labels = [
                self._split_surface_for_wheel(xy[0], xy[1], road_profile)
                for xy in wheel_xy
            ]
            mu = np.array([ROAD_MU[label] for label in labels], dtype=np.float64)
        elif road_profile.road_type == "transition":
            labels, mu = self._transition_surfaces(t, wheel_xy, road_profile)
        else:
            raise ValueError("unsupported road_type: %s" % road_profile.road_type)

        return {
            "mu": mu,
            "labels": labels,
            "height": self._rough_height(t, road_profile),
            "normal": np.tile(np.array([0.0, 0.0, 1.0]), (4, 1)),
            "grade": road_profile.grade_rad,
            "bank": road_profile.bank_rad,
            "wheel_xy": wheel_xy,
        }

    @staticmethod
    def wheel_world_xy(state: TeacherState, vehicle: VehicleConfig) -> np.ndarray:
        cg_to_front = vehicle.wheelbase - vehicle.cg_x
        cg_to_rear = -vehicle.cg_x
        local = np.array(
            [
                [cg_to_front, vehicle.track_front / 2.0],
                [cg_to_front, -vehicle.track_front / 2.0],
                [cg_to_rear, vehicle.track_rear / 2.0],
                [cg_to_rear, -vehicle.track_rear / 2.0],
            ],
            dtype=np.float64,
        )
        c = np.cos(state.yaw)
        s = np.sin(state.yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        origin = np.array([state.x_world, state.y_world], dtype=np.float64)
        return local @ rot.T + origin

    @staticmethod
    def _default_wheel_xy() -> np.ndarray:
        return np.array(
            [
                [1.4, 0.8],
                [1.4, -0.8],
                [-1.4, 0.8],
                [-1.4, -0.8],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _split_surface_for_wheel(x: float, y: float, road_profile: RoadProfile) -> str:
        if x < road_profile.split_start_x_m or x > road_profile.split_end_x_m:
            return str(road_profile.split_default_surface)
        if y >= road_profile.split_boundary_y_m:
            return str(road_profile.split_left)
        return str(road_profile.split_right)

    def _transition_surfaces(
        self,
        t: float,
        wheel_xy: np.ndarray,
        road_profile: RoadProfile,
    ) -> tuple[List[str], np.ndarray]:
        labels = []
        mu_values = []
        for x, _ in wheel_xy:
            spatial_progress = np.clip(
                (x - road_profile.transition_start_x_m)
                / max(road_profile.transition_length_m, 1e-6),
                0.0,
                1.0,
            )
            temporal_progress = np.clip(
                (t - road_profile.transition_start_s)
                / max(road_profile.transition_duration_s, 1e-6),
                0.0,
                1.0,
            )
            progress = max(float(spatial_progress), float(temporal_progress))
            mu_from = ROAD_MU[str(road_profile.transition_from)]
            mu_to = ROAD_MU[str(road_profile.transition_to)]
            mu_values.append((1.0 - progress) * mu_from + progress * mu_to)
            labels.append(
                "%s_to_%s" % (
                    road_profile.transition_from,
                    road_profile.transition_to,
                )
            )
        return labels, np.asarray(mu_values, dtype=np.float64)

    @staticmethod
    def _rough_height(t: float, road_profile: RoadProfile) -> np.ndarray:
        amp = road_profile.roughness_amp_m
        phases = np.array([0.0, 0.7, 1.9, 2.6])
        return amp * np.sin(2.0 * np.pi * 1.1 * t + phases)
