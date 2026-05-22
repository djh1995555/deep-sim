from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from simulator.controller.base import ControllerReference
from simulator.vehicle_model.scenario import ScenarioConfig
from simulator.vehicle_model.state import TeacherState


@dataclass
class Waypoint:
    x_m: float
    y_m: float
    z_m: float = 0.0
    speed_mps: float = 0.0
    yaw_rad: float = 0.0
    yaw_rate_rps: float = 0.0
    curvature_1pm: float = 0.0
    path_s_m: float = 0.0
    lookahead_distance_m: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


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
            [
                [
                    p.x_m,
                    p.y_m,
                    p.z_m,
                    p.speed_mps,
                    p.yaw_rad,
                    p.yaw_rate_rps,
                    p.curvature_1pm,
                ]
                for p in config.waypoints
            ],
            dtype=np.float64,
        )
        deltas = np.diff(self.points[:, :2], axis=0)
        lengths = np.linalg.norm(deltas, axis=1)
        if np.any(lengths <= 1e-9):
            raise ValueError("consecutive waypoints must be distinct")
        self.segment_lengths = lengths
        self.s = np.concatenate([[0.0], np.cumsum(lengths)])
        for waypoint, path_s in zip(self.config.waypoints, self.s):
            waypoint.path_s_m = float(path_s)
            waypoint.lookahead_distance_m = float(self.config.lookahead_m)

    def reset(self, scenario: ScenarioConfig) -> None:
        pass

    def query(self, t: float, state: TeacherState) -> ControllerReference:
        nearest_s = self._project_state_to_path_s(state)
        target_s = min(float(self.s[-1]), nearest_s + self.config.lookahead_m)
        target = self._interpolate_at_s(target_s)
        yaw = self._yaw_at_s(target_s, target)
        return ControllerReference(
            x_m=float(target[0]),
            y_m=float(target[1]),
            z_m=float(target[2]),
            speed_mps=float(target[3]),
            yaw_rad=yaw,
            yaw_rate_rps=float(target[5]),
            curvature_1pm=float(target[6]),
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

    def _yaw_at_s(self, target_s: float, target: np.ndarray) -> float:
        if not np.isnan(target[4]):
            return float(target[4])
        if target_s >= self.s[-1]:
            idx = len(self.segment_lengths) - 1
        else:
            idx = int(np.searchsorted(self.s, target_s, side="right") - 1)
        delta = self.points[idx + 1, :2] - self.points[idx, :2]
        return float(np.arctan2(delta[1], delta[0]))


def parse_waypoint(value: Any) -> Waypoint:
    if not isinstance(value, dict):
        raise ValueError("waypoint must be a mapping with explicit fields")
    required = {"x_m", "y_m", "yaw_rad", "curvature_1pm", "speed_mps"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError("waypoint missing required fields: %s" % ", ".join(missing))
    allowed = required | {"z_m", "yaw_rate_rps", "path_s_m", "lookahead_distance_m", "extra"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError("unknown waypoint fields: %s" % ", ".join(unknown))
    extra = value.get("extra", {})
    if not isinstance(extra, dict):
        raise ValueError("waypoint extra must be a mapping")
    return Waypoint(
        x_m=float(value["x_m"]),
        y_m=float(value["y_m"]),
        z_m=float(value.get("z_m", 0.0)),
        speed_mps=float(value["speed_mps"]),
        yaw_rad=float(value["yaw_rad"]),
        yaw_rate_rps=float(value.get("yaw_rate_rps", 0.0)),
        curvature_1pm=float(value["curvature_1pm"]),
        path_s_m=float(value.get("path_s_m", 0.0)),
        lookahead_distance_m=float(value.get("lookahead_distance_m", 0.0)),
        extra=dict(extra),
    )
