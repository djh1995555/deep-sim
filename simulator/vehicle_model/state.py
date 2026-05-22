from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class TeacherState:
    x_world: float = 0.0
    y_world: float = 0.0
    z_world: float = 0.0
    vx: float = 14.0
    vy: float = 0.0
    vz: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    p: float = 0.0
    q: float = 0.0
    r: float = 0.0
    omega: np.ndarray = field(
        default_factory=lambda: np.array([42.0, 42.0, 42.0, 42.0], dtype=np.float64)
    )
    steering_delta: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0], dtype=np.float64)
    )
    drive_torque: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float64)
    )
    brake_torque: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float64)
    )
    prev_ax: float = 0.0
    prev_ay: float = 0.0

    def limited_dict(self) -> Dict[str, float]:
        return {
            "vx": self.vx,
            "vy": self.vy,
            "roll": self.roll,
            "pitch": self.pitch,
            "r": self.r,
        }
