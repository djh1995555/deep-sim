import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List


WHEEL_ORDER = ["FL", "FR", "RL", "RR"]


@dataclass
class VehicleConfig:
    vehicle_id: str = "vehicle_A_debug"
    vehicle_family: str = "vehicle_A"
    vehicle_config_id: str = "vehicle_A_debug_base"
    fixed_vehicle_context: Dict[str, Any] = field(default_factory=dict)
    nominal_physics_prior: Dict[str, float] = field(default_factory=dict)
    hidden_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def wheelbase(self) -> float:
        return float(self.fixed_vehicle_context["wheelbase"])

    @property
    def wheel_radius(self) -> float:
        return float(self.fixed_vehicle_context["wheel_radius"])

    @property
    def track_front(self) -> float:
        return float(self.fixed_vehicle_context["track_front"])

    @property
    def track_rear(self) -> float:
        return float(self.fixed_vehicle_context["track_rear"])

    @property
    def mass(self) -> float:
        return float(self.hidden_params["mass_true"])

    @property
    def iz(self) -> float:
        return float(self.hidden_params["Iz_true"])

    @property
    def cg_x(self) -> float:
        return float(self.hidden_params["cg_x_true"])

    @property
    def cg_z(self) -> float:
        return float(self.hidden_params["cg_z_true"])

    @property
    def driven_wheels(self) -> List[str]:
        return list(self.fixed_vehicle_context["drive_layout"]["driven_wheels"])


def vehicle_internal_hash(hidden_params: Dict[str, Any]) -> str:
    payload = json.dumps(hidden_params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def default_vehicle_config() -> VehicleConfig:
    fixed_context = {
        "wheelbase": 2.85,
        "track_front": 1.62,
        "track_rear": 1.60,
        "wheel_radius": 0.335,
        "wheel_order": list(WHEEL_ORDER),
        "steering_layout": {
            "type": "front_wheel_steer",
            "steered_wheels": ["FL", "FR"],
            "input_signal": "sw_angle",
            "steering_ratio_nominal": 14.5,
        },
        "drive_layout": {
            "type": "AWD",
            "driven_wheels": list(WHEEL_ORDER),
        },
        "brake_layout": {
            "type": "four_wheel_observed",
        },
    }
    nominal_prior = {
        "mass_nominal": 1800.0,
        "Ix_nominal": 650.0,
        "Iy_nominal": 2450.0,
        "Iz_nominal": 2750.0,
        "cg_x_nominal": 1.35,
        "cg_z_nominal": 0.56,
        "tau_steer_nominal": 0.11,
    }
    hidden = {
        "mass_true": 1865.0,
        "Ix_true": 690.0,
        "Iy_true": 2520.0,
        "Iz_true": 2860.0,
        "cg_x_true": 1.32,
        "cg_z_true": 0.58,
        "cornering_stiffness_front": 76000.0,
        "cornering_stiffness_rear": 84000.0,
        "wheel_inertia": 1.25,
        "drag_coefficient_area": 0.76,
        "air_density": 1.2,
        "rolling_resistance": 0.012,
        "steering_tau_true": 0.14,
        "steering_backlash_rad": 0.003,
        "steering_compliance_scale": 0.96,
        "drive_tau_true": 0.08,
        "brake_tau_true": 0.05,
        "max_drive_torque_per_wheel": 1450.0,
        "max_brake_torque_per_wheel": 3400.0,
        "brake_bias_front": 0.62,
        "sensor_bias": {
            "vx": 0.02,
            "vy": -0.01,
            "r": 0.0005,
            "sw_angle": 0.001,
        },
        "sensor_noise_std": {
            "vx": 0.015,
            "vy": 0.015,
            "roll": 0.0005,
            "pitch": 0.0005,
            "yaw": 0.0008,
            "p": 0.001,
            "q": 0.001,
            "r": 0.001,
            "omega": 0.03,
            "torque": 2.0,
            "sw_angle": 0.0008,
            "cmd": 0.0005,
        },
    }
    return VehicleConfig(
        fixed_vehicle_context=fixed_context,
        nominal_physics_prior=nominal_prior,
        hidden_params=hidden,
    )
