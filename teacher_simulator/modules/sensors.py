from typing import Dict, Optional

import numpy as np

from teacher_simulator.scenario import SensorProfile
from teacher_simulator.state import TeacherState
from teacher_simulator.vehicle_params import VehicleConfig


class SensorModel:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.last_timestamp: Optional[float] = None

    def observe(
        self,
        t: float,
        state: TeacherState,
        command: Dict[str, float],
        torques: Dict[str, np.ndarray],
        vehicle: VehicleConfig,
        sensor_profile: SensorProfile,
    ) -> Dict[str, float]:
        hidden = vehicle.hidden_params
        bias = hidden["sensor_bias"]
        std = hidden["sensor_noise_std"]
        scale = sensor_profile.noise_scale
        timestamp = t + self.rng.normal(0.0, sensor_profile.timestamp_jitter_std_s)
        if self.last_timestamp is not None:
            timestamp = max(timestamp, self.last_timestamp + 1e-6)
        self.last_timestamp = timestamp

        def noise(name: str) -> float:
            return float(self.rng.normal(0.0, std.get(name, 0.0) * scale))

        obs = {
            "timestamp": float(timestamp),
            "vx": state.vx + bias.get("vx", 0.0) + noise("vx"),
            "vy": state.vy + bias.get("vy", 0.0) + noise("vy"),
            "roll": state.roll + noise("roll"),
            "pitch": state.pitch + noise("pitch"),
            "yaw": state.yaw + noise("yaw"),
            "p": state.p + noise("p"),
            "q": state.q + noise("q"),
            "r": state.r + bias.get("r", 0.0) + noise("r"),
            "sw_angle": command["sw_angle"] + bias.get("sw_angle", 0.0) + noise("sw_angle"),
            "steer_cmd": command["steer_cmd"] + noise("cmd"),
            "throttle_cmd": command["throttle_cmd"] + noise("cmd"),
            "brake_cmd": command["brake_cmd"] + noise("cmd"),
        }
        for idx, suffix in enumerate(["fl", "fr", "rl", "rr"]):
            obs["omega_%s" % suffix] = state.omega[idx] + noise("omega")
            obs["tau_drv_obs_%s" % suffix] = torques["tau_drv_true_i"][idx] + noise("torque")
            obs["tau_brk_obs_%s" % suffix] = torques["tau_brk_true_i"][idx] + noise("torque")

        quant = sensor_profile.quantization
        if quant > 0:
            for key, value in list(obs.items()):
                obs[key] = float(np.round(value / quant) * quant)
        return obs
