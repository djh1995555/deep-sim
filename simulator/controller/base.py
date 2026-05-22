from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

from simulator.vehicle_model.scenario import ScenarioConfig
from simulator.vehicle_model.state import TeacherState


@dataclass
class ControllerReference:
    target_x_m: float = 0.0
    target_speed_mps: float = 14.0
    target_y_m: float = 0.0
    target_yaw_rad: float = 0.0
    target_yaw_rate_rps: float = 0.0
    target_curvature_1pm: float = 0.0
    path_s_m: float = 0.0
    lookahead_distance_m: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ControllerInput:
    t: float
    dt: float
    state: TeacherState
    scenario: ScenarioConfig
    reference: ControllerReference
    observation: Optional[Dict[str, float]] = None

    def debug_dict(self) -> Dict[str, float]:
        return {
            "t": float(self.t),
            "vx": float(self.state.vx),
            "vy": float(self.state.vy),
            "x_world": float(self.state.x_world),
            "y_world": float(self.state.y_world),
            "yaw": float(self.state.yaw),
            "r": float(self.state.r),
            "target_x_m": float(self.reference.target_x_m),
            "target_speed_mps": float(self.reference.target_speed_mps),
            "target_y_m": float(self.reference.target_y_m),
            "target_yaw_rad": float(self.reference.target_yaw_rad),
            "target_yaw_rate_rps": float(self.reference.target_yaw_rate_rps),
            "target_curvature_1pm": float(self.reference.target_curvature_1pm),
            "path_s_m": float(self.reference.path_s_m),
            "lookahead_distance_m": float(self.reference.lookahead_distance_m),
        }


@dataclass
class ControllerOutput:
    sw_angle: float
    throttle_cmd: float
    brake_cmd: float
    steer_cmd: Optional[float] = None
    debug: Dict[str, Any] = field(default_factory=dict)

    def command_dict(self) -> Dict[str, float]:
        steer = self.sw_angle if self.steer_cmd is None else self.steer_cmd
        return {
            "sw_angle": float(self.sw_angle),
            "steer_cmd": float(steer),
            "throttle_cmd": float(self.throttle_cmd),
            "brake_cmd": float(self.brake_cmd),
        }


class Controller(Protocol):
    def reset(self, scenario: ScenarioConfig) -> None:
        ...

    def compute(self, controller_input: ControllerInput) -> ControllerOutput:
        ...
