from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from simulator.controller.base import ControllerReference
from simulator.vehicle_model.scenario import ScenarioConfig
from simulator.vehicle_model.state import TeacherState


@dataclass
class Waypoint:
    x_m: float
    y_m: float
    speed_mps: float


@dataclass
class WaypointReferenceConfig:
    waypoints: List[Waypoint]
    lookahead_m: float = 8.0


class WaypointReferenceProvider:
    def __init__(self, config: WaypointReferenceConfig) -> None:
        if len(config.waypoints) < 2:
            raise ValueError("waypoint reference requires at least two waypoints")
        self.config = config
        self.points = np.asarray(
            [[p.x_m, p.y_m, p.speed_mps] for p in config.waypoints],
            dtype=np.float64,
        )
        deltas = np.diff(self.points[:, :2], axis=0)
        lengths = np.linalg.norm(deltas, axis=1)
        if np.any(lengths <= 1e-9):
            raise ValueError("consecutive waypoints must be distinct")
        self.segment_lengths = lengths
        self.s = np.concatenate([[0.0], np.cumsum(lengths)])

    def reset(self, scenario: ScenarioConfig) -> None:
        pass

    def query(self, t: float, state: TeacherState) -> ControllerReference:
        nearest_s = self._project_state_to_path_s(state)
        target_s = min(float(self.s[-1]), nearest_s + self.config.lookahead_m)
        target = self._interpolate_at_s(target_s)
        yaw = self._yaw_at_s(target_s)
        speed = float(target[2])
        return ControllerReference(
            target_x_m=float(target[0]),
            target_y_m=float(target[1]),
            target_speed_mps=speed,
            target_yaw_rad=yaw,
            target_yaw_rate_rps=0.0,
            target_curvature_1pm=0.0,
            path_s_m=float(target_s),
            lookahead_distance_m=float(self.config.lookahead_m),
            extra={
                "provider": "waypoints",
                "nearest_s_m": float(nearest_s),
            },
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "waypoints",
            "lookahead_m": self.config.lookahead_m,
            "waypoints": [p.__dict__ for p in self.config.waypoints],
        }

    def _project_state_to_path_s(self, state: TeacherState) -> float:
        point = np.array([state.x_world, state.y_world], dtype=np.float64)
        best_dist = float("inf")
        best_s = 0.0
        for idx in range(len(self.segment_lengths)):
            a = self.points[idx, :2]
            b = self.points[idx + 1, :2]
            ab = b - a
            u = float(np.clip(np.dot(point - a, ab) / np.dot(ab, ab), 0.0, 1.0))
            projection = a + u * ab
            dist = float(np.linalg.norm(point - projection))
            if dist < best_dist:
                best_dist = dist
                best_s = float(self.s[idx] + u * self.segment_lengths[idx])
        return best_s

    def _interpolate_at_s(self, target_s: float) -> np.ndarray:
        if target_s <= 0.0:
            return self.points[0].copy()
        if target_s >= self.s[-1]:
            return self.points[-1].copy()
        idx = int(np.searchsorted(self.s, target_s, side="right") - 1)
        u = (target_s - self.s[idx]) / self.segment_lengths[idx]
        return (1.0 - u) * self.points[idx] + u * self.points[idx + 1]

    def _yaw_at_s(self, target_s: float) -> float:
        if target_s >= self.s[-1]:
            idx = len(self.segment_lengths) - 1
        else:
            idx = int(np.searchsorted(self.s, target_s, side="right") - 1)
        delta = self.points[idx + 1, :2] - self.points[idx, :2]
        return float(np.arctan2(delta[1], delta[0]))


def parse_waypoint(value: Any, default_speed_mps: float) -> Waypoint:
    if isinstance(value, dict):
        return Waypoint(
            x_m=float(value["x_m"] if "x_m" in value else value["x"]),
            y_m=float(value["y_m"] if "y_m" in value else value["y"]),
            speed_mps=float(
                value["speed_mps"]
                if "speed_mps" in value
                else value.get("speed", default_speed_mps)
            ),
        )
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) >= 2
    ):
        speed = float(value[2]) if len(value) >= 3 else default_speed_mps
        return Waypoint(x_m=float(value[0]), y_m=float(value[1]), speed_mps=speed)
    raise ValueError("waypoint must be a mapping or sequence [x, y, speed?]")
