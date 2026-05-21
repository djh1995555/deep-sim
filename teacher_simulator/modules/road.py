from typing import Dict, List

import numpy as np

from teacher_simulator.scenario import ROAD_MU, RoadProfile


class RoadModel:
    def query(self, t: float, road_profile: RoadProfile) -> Dict[str, object]:
        if road_profile.road_type == "single":
            labels = [road_profile.single_surface] * 4
        elif road_profile.road_type == "split":
            labels = [
                road_profile.split_left,
                road_profile.split_right,
                road_profile.split_left,
                road_profile.split_right,
            ]
        elif road_profile.road_type == "transition":
            progress = np.clip(
                (t - road_profile.transition_start_s)
                / max(road_profile.transition_duration_s, 1e-6),
                0.0,
                1.0,
            )
            mu_from = ROAD_MU[road_profile.transition_from]
            mu_to = ROAD_MU[road_profile.transition_to]
            mu = (1.0 - progress) * mu_from + progress * mu_to
            label = "%s_to_%s" % (
                road_profile.transition_from,
                road_profile.transition_to,
            )
            return {
                "mu": np.full(4, mu, dtype=np.float64),
                "labels": [label] * 4,
                "height": self._rough_height(t, road_profile),
                "normal": np.tile(np.array([0.0, 0.0, 1.0]), (4, 1)),
                "grade": road_profile.grade_rad,
                "bank": road_profile.bank_rad,
            }
        else:
            raise ValueError("unsupported road_type: %s" % road_profile.road_type)

        label_list: List[str] = [str(label) for label in labels]
        return {
            "mu": np.array([ROAD_MU[label] for label in label_list], dtype=np.float64),
            "labels": label_list,
            "height": self._rough_height(t, road_profile),
            "normal": np.tile(np.array([0.0, 0.0, 1.0]), (4, 1)),
            "grade": road_profile.grade_rad,
            "bank": road_profile.bank_rad,
        }

    @staticmethod
    def _rough_height(t: float, road_profile: RoadProfile) -> np.ndarray:
        amp = road_profile.roughness_amp_m
        phases = np.array([0.0, 0.7, 1.9, 2.6])
        return amp * np.sin(2.0 * np.pi * 1.1 * t + phases)
