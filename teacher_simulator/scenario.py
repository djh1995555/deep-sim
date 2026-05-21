from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from teacher_simulator.vehicle_params import VehicleConfig, default_vehicle_config


ROAD_MU = {
    "dry": 0.95,
    "wet": 0.55,
    "snow": 0.28,
    "ice": 0.10,
}


@dataclass
class RoadProfile:
    road_type: str
    mu_pattern: str
    single_surface: Optional[str] = None
    split_left: Optional[str] = None
    split_right: Optional[str] = None
    transition_from: Optional[str] = None
    transition_to: Optional[str] = None
    transition_start_s: float = 4.0
    transition_duration_s: float = 3.0
    grade_rad: float = 0.0
    bank_rad: float = 0.0
    roughness_amp_m: float = 0.003

    @property
    def factor_id(self) -> str:
        if self.road_type == "single":
            return "single_%s" % self.single_surface
        if self.road_type == "split":
            return "split_left_%s_right_%s" % (self.split_left, self.split_right)
        if self.road_type == "transition":
            return "transition_%s_to_%s" % (self.transition_from, self.transition_to)
        return self.mu_pattern


@dataclass
class ControlScript:
    longitudinal: str
    lateral: str
    initial_speed_mps: float = 14.0

    @property
    def factor_id(self) -> str:
        return "%s-%s" % (self.longitudinal, self.lateral)

    def command_at(self, t: float) -> Dict[str, float]:
        ramp = 1.0 - np.exp(-max(0.0, t - 0.4) / 0.8)
        throttle = 0.045
        brake = 0.0
        if self.longitudinal == "small_acceleration":
            throttle = 0.28 * ramp
        elif self.longitudinal == "large_acceleration":
            throttle = 0.70 * ramp
        elif self.longitudinal == "small_braking":
            throttle = 0.0
            brake = 0.25 * ramp
        elif self.longitudinal == "large_braking":
            throttle = 0.0
            brake = 0.72 * ramp

        sw_amp = 0.0
        if self.lateral == "small_left_steer":
            sw_amp = 0.45
        elif self.lateral == "small_right_steer":
            sw_amp = -0.45
        elif self.lateral == "large_left_steer":
            sw_amp = 1.10
        elif self.lateral == "large_right_steer":
            sw_amp = -1.10
        steer_wave = sw_amp * ramp
        if "large" in self.lateral:
            steer_wave *= 0.85 + 0.15 * np.sin(2.0 * np.pi * 0.25 * t)
        return {
            "sw_angle": float(steer_wave),
            "steer_cmd": float(steer_wave),
            "throttle_cmd": float(np.clip(throttle, 0.0, 1.0)),
            "brake_cmd": float(np.clip(brake, 0.0, 1.0)),
        }


@dataclass
class ActuatorProfile:
    torque_observability_mode: str = "actuator_estimate"
    steering_saturation_rad: float = 0.62
    sensor_delay_steps: int = 1


@dataclass
class SensorProfile:
    noise_scale: float = 1.0
    timestamp_jitter_std_s: float = 0.00015
    dropout_probability: float = 0.0
    quantization: float = 1e-5


@dataclass
class EnvironmentProfile:
    wind_x_mps: float = 0.0
    wind_y_mps: float = 0.5


@dataclass
class SplitMetadata:
    split_role: str = "train"
    target_window_id: Optional[str] = None
    fine_tune_data_bucket: Optional[str] = None


@dataclass
class ScenarioConfig:
    scenario_id: str
    vehicle_config: VehicleConfig
    road_profile: RoadProfile
    control_script: ControlScript
    actuator_profile: ActuatorProfile = field(default_factory=ActuatorProfile)
    sensor_profile: SensorProfile = field(default_factory=SensorProfile)
    environment_profile: EnvironmentProfile = field(default_factory=EnvironmentProfile)
    teacher_feature_flags: Dict[str, Any] = field(default_factory=dict)
    split_metadata: SplitMetadata = field(default_factory=SplitMetadata)
    validation_case: str = "general"
    seed: int = 0


def _scenario(
    road: RoadProfile,
    longitudinal: str,
    lateral: str,
    validation_case: str,
    seed: int,
) -> ScenarioConfig:
    vehicle = default_vehicle_config()
    control = ControlScript(longitudinal=longitudinal, lateral=lateral)
    scenario_id = "%s-%s-%s" % (road.factor_id, longitudinal, lateral)
    return ScenarioConfig(
        scenario_id=scenario_id,
        vehicle_config=vehicle,
        road_profile=road,
        control_script=control,
        validation_case=validation_case,
        seed=seed,
    )


def make_ds0_scenarios(seed: int = 7) -> List[ScenarioConfig]:
    cases: List[Tuple[RoadProfile, str, str, str]] = [
        (
            RoadProfile("single", "dry", single_surface="dry"),
            "const_v",
            "no_steer",
            "steady",
        ),
        (
            RoadProfile("single", "dry", single_surface="dry"),
            "small_acceleration",
            "no_steer",
            "acceleration",
        ),
        (
            RoadProfile("single", "dry", single_surface="dry"),
            "large_braking",
            "no_steer",
            "braking",
        ),
        (
            RoadProfile("single", "wet", single_surface="wet"),
            "const_v",
            "small_left_steer",
            "left_turn",
        ),
        (
            RoadProfile("single", "dry", single_surface="dry"),
            "const_v",
            "large_right_steer",
            "right_turn",
        ),
        (
            RoadProfile(
                "split",
                "split",
                split_left="dry",
                split_right="ice",
            ),
            "large_braking",
            "no_steer",
            "split_mu_brake_left_high",
        ),
        (
            RoadProfile(
                "split",
                "split",
                split_left="ice",
                split_right="dry",
            ),
            "large_braking",
            "no_steer",
            "split_mu_brake_right_high",
        ),
        (
            RoadProfile(
                "transition",
                "transition",
                transition_from="dry",
                transition_to="wet",
            ),
            "small_braking",
            "small_left_steer",
            "transition_mu",
        ),
    ]
    return [_scenario(*case, seed=seed + idx) for idx, case in enumerate(cases)]
