from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simulator.vehicle_model.vehicle_params import (
    VehicleConfig,
    default_vehicle_config,
    make_vehicle_config_variants,
)


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
    transition_start_x_m: float = 25.0
    transition_length_m: float = 35.0
    split_boundary_y_m: float = 0.0
    split_start_x_m: float = 0.0
    split_end_x_m: float = 120.0
    split_default_surface: str = "dry"
    grade_rad: float = 0.0
    bank_rad: float = 0.0
    roughness_amp_m: float = 0.003
    factor_id_override: Optional[str] = None
    scenario_group: Optional[str] = None

    @property
    def factor_id(self) -> str:
        if self.factor_id_override:
            return self.factor_id_override
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
    longitudinal_id: Optional[str] = None
    lateral_id: Optional[str] = None

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
        elif self.longitudinal == "emergency_braking":
            throttle = 0.0
            brake = 0.95 * ramp
        elif self.longitudinal == "trail_braking":
            throttle = 0.0
            brake = 0.62 * ramp * np.exp(-0.20 * max(0.0, t - 1.0))
        elif self.longitudinal == "power_on_exit":
            throttle = 0.16 * ramp if t < 1.5 else 0.82 * ramp
            brake = 0.0

        sw_amp = 0.0
        if self.lateral == "small_left_steer":
            sw_amp = 0.45
        elif self.lateral == "small_right_steer":
            sw_amp = -0.45
        elif self.lateral == "large_left_steer":
            sw_amp = 1.10
        elif self.lateral == "large_right_steer":
            sw_amp = -1.10
        if self.lateral == "fishhook_left_right":
            if t < 1.55:
                steer_wave = 1.35 * ramp
            elif t < 2.65:
                steer_wave = -1.35 * (1.0 - np.exp(-(t - 1.55) / 0.35))
            else:
                steer_wave = 0.30 * np.exp(-(t - 2.65) / 0.65)
        elif self.lateral == "emergency_lane_change_left":
            phase = np.clip((t - 0.45) / 2.1, 0.0, 1.0)
            steer_wave = 1.28 * np.sin(np.pi * phase)
        elif self.lateral == "sine_sweep":
            steer_wave = 1.10 * np.sin(2.0 * np.pi * 0.65 * max(0.0, t - 0.4)) * ramp
        else:
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
class InitialState:
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    speed_mps: Optional[float] = None
    vy_mps: float = 0.0
    vz_mps: float = 0.0
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0
    roll_rate_rps: float = 0.0
    pitch_rate_rps: float = 0.0
    yaw_rate_rps: float = 0.0


@dataclass
class ScenarioConfig:
    scenario_id: str
    vehicle_config: VehicleConfig
    road_profile: RoadProfile
    control_script: ControlScript
    initial_state: InitialState = field(default_factory=InitialState)
    actuator_profile: ActuatorProfile = field(default_factory=ActuatorProfile)
    sensor_profile: SensorProfile = field(default_factory=SensorProfile)
    environment_profile: EnvironmentProfile = field(default_factory=EnvironmentProfile)
    teacher_feature_flags: Dict[str, Any] = field(default_factory=dict)
    split_metadata: SplitMetadata = field(default_factory=SplitMetadata)
    validation_case: str = "general"
    seed: int = 0
    scenario_group: str = "DS0"
    road_factor_id: str = ""
    longitudinal_factor_id: str = ""
    lateral_factor_id: str = ""
    full_matrix_index: Optional[int] = None
    perturbation_profile_id: Optional[str] = None
    target_window_role: Optional[str] = None


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


LONGITUDINAL_FACTORS = [
    ("L0", "const_v"),
    ("L1", "small_acceleration"),
    ("L2", "large_acceleration"),
    ("L3", "small_braking"),
    ("L4", "large_braking"),
]

LATERAL_FACTORS = [
    ("Y0", "no_steer"),
    ("Y1", "small_left_steer"),
    ("Y2", "small_right_steer"),
    ("Y3", "large_left_steer"),
    ("Y4", "large_right_steer"),
]

SINGLE_ROAD_FACTORS = [
    ("R00", "dry"),
    ("R01", "wet"),
    ("R02", "snow"),
    ("R03", "ice"),
]

SPLIT_ROAD_FACTORS = [
    ("SP00", "dry", "wet"),
    ("SP01", "dry", "snow"),
    ("SP02", "dry", "ice"),
    ("SP03", "wet", "snow"),
    ("SP04", "wet", "ice"),
    ("SP05", "snow", "ice"),
    ("SP06", "wet", "dry"),
    ("SP07", "snow", "dry"),
    ("SP08", "ice", "dry"),
    ("SP09", "snow", "wet"),
    ("SP10", "ice", "wet"),
    ("SP11", "ice", "snow"),
]

TRANSITION_ROAD_FACTORS = [
    ("TR00", "dry", "wet"),
    ("TR01", "dry", "snow"),
    ("TR02", "dry", "ice"),
    ("TR03", "wet", "dry"),
    ("TR04", "wet", "snow"),
    ("TR05", "wet", "ice"),
    ("TR06", "snow", "dry"),
    ("TR07", "snow", "wet"),
    ("TR08", "snow", "ice"),
    ("TR09", "ice", "dry"),
    ("TR10", "ice", "wet"),
    ("TR11", "ice", "snow"),
]


def make_ds1_scenario_matrix() -> List[Dict[str, str]]:
    matrix: List[Dict[str, str]] = []
    for group, road_factors in [
        ("CG-SINGLE", SINGLE_ROAD_FACTORS),
        ("CG-SPLIT", SPLIT_ROAD_FACTORS),
        ("CG-TRANSITION", TRANSITION_ROAD_FACTORS),
    ]:
        for road_factor in road_factors:
            road_id = road_factor[0]
            for long_id, long_name in LONGITUDINAL_FACTORS:
                for lat_id, lat_name in LATERAL_FACTORS:
                    scenario_id = "%s-%s-%s" % (road_id, long_id, lat_id)
                    matrix.append(
                        {
                            "scenario_id": scenario_id,
                            "scenario_group": group,
                            "road_factor_id": road_id,
                            "longitudinal_factor_id": long_id,
                            "lateral_factor_id": lat_id,
                            "longitudinal": long_name,
                            "lateral": lat_name,
                        }
                    )
    return matrix


DS2_EXTREME_FACTORS = [
    ("EX00", "single", "dry", "", "emergency_braking", "fishhook_left_right"),
    ("EX01", "single", "wet", "", "trail_braking", "emergency_lane_change_left"),
    ("EX02", "single", "snow", "", "power_on_exit", "sine_sweep"),
    ("EX03", "split", "dry", "ice", "emergency_braking", "fishhook_left_right"),
    ("EX04", "split", "wet", "snow", "trail_braking", "emergency_lane_change_left"),
    ("EX05", "transition", "wet", "ice", "power_on_exit", "sine_sweep"),
]


def make_ds2_extreme_matrix() -> List[Dict[str, str]]:
    matrix: List[Dict[str, str]] = []
    for idx, (factor_id, road_type, first, second, longitudinal, lateral) in enumerate(DS2_EXTREME_FACTORS):
        matrix.append(
            {
                "scenario_id": factor_id,
                "scenario_group": "DS2-EXTREME",
                "road_factor_id": factor_id,
                "longitudinal_factor_id": "EL%d" % idx,
                "lateral_factor_id": "EY%d" % idx,
                "longitudinal": longitudinal,
                "lateral": lateral,
                "road_type": road_type,
                "surface_a": first,
                "surface_b": second,
            }
        )
    return matrix


def make_ds2_extreme_scenarios(
    seed: int = 47,
    vehicle_count: int = 3,
    samples_per_vehicle: int = 6,
) -> List[ScenarioConfig]:
    matrix = make_ds2_extreme_matrix()
    vehicles = make_vehicle_config_variants(vehicle_count)
    scenarios: List[ScenarioConfig] = []
    for vehicle_idx, vehicle in enumerate(vehicles):
        for local_idx, item in enumerate(matrix[:samples_per_vehicle]):
            split_role = "train"
            if vehicle_idx == vehicle_count - 1:
                split_role = "held-out" if local_idx % 2 == 0 else "test"
            elif local_idx % 5 == 0:
                split_role = "validation"
            scenarios.append(
                _ds2_extreme_scenario_from_item(
                    item,
                    vehicle,
                    split_role,
                    seed + vehicle_idx * 1000 + local_idx,
                    local_idx,
                )
            )
    return scenarios


def make_ds1_scenarios(
    seed: int = 7,
    vehicle_count: int = 4,
    samples_per_group_per_vehicle: int = 6,
) -> List[ScenarioConfig]:
    matrix = make_ds1_scenario_matrix()
    vehicles = make_vehicle_config_variants(vehicle_count)
    scenarios: List[ScenarioConfig] = []
    by_group: Dict[str, List[Tuple[int, Dict[str, str]]]] = {}
    for idx, item in enumerate(matrix):
        by_group.setdefault(item["scenario_group"], []).append((idx, item))

    for vehicle_idx, vehicle in enumerate(vehicles):
        for group, group_items in sorted(by_group.items()):
            selected = _balanced_group_sample(
                group_items,
                samples_per_group_per_vehicle,
                seed + vehicle_idx * 100 + len(scenarios),
            )
            for local_idx, (matrix_idx, item) in enumerate(selected):
                split_role = _split_role(vehicle_idx, vehicle_count, group, local_idx)
                target_window_id = None
                fine_tune_bucket = None
                if split_role in ["fine-tune", "test-window"]:
                    target_window_id = "tw_%s_%02d" % (vehicle.vehicle_config_id, local_idx)
                    fine_tune_bucket = _fine_tune_bucket(local_idx)
                scenarios.append(
                    _ds1_scenario_from_matrix_item(
                        item,
                        vehicle,
                        split_role,
                        target_window_id,
                        fine_tune_bucket,
                        seed + matrix_idx + vehicle_idx * 1000,
                        matrix_idx,
                    )
                )
    return scenarios


def make_proxy_perturbation_profiles(
    seed: int = 23,
    profile_count: int = 3,
    min_magnitude: float = 0.05,
    max_magnitude: float = 0.15,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed + 7919)
    knobs = [
        "mass",
        "inertia",
        "cg",
        "tire",
        "suspension",
        "sensor",
        "actuator",
    ]
    profiles: List[Dict[str, Any]] = []
    for profile_idx in range(profile_count):
        magnitudes: Dict[str, float] = {}
        scales: Dict[str, float] = {}
        signs: Dict[str, int] = {}
        for knob in knobs:
            magnitude = float(rng.uniform(min_magnitude, max_magnitude))
            sign = -1 if float(rng.random()) < 0.5 else 1
            magnitudes[knob] = magnitude
            signs[knob] = sign
            scales[knob] = 1.0 + sign * magnitude
        profiles.append(
            {
                "profile_id": "proxy_profile_%02d" % profile_idx,
                "seed": int(seed + profile_idx),
                "min_requested_magnitude": float(min_magnitude),
                "max_requested_magnitude": float(max_magnitude),
                "magnitudes": magnitudes,
                "signs": signs,
                "scales": scales,
                "max_abs_magnitude": max(magnitudes.values()),
                "min_abs_magnitude": min(magnitudes.values()),
                "description": "teacher-only sim-to-real proxy offsets",
            }
        )
    return profiles


def make_ds1_proxy_scenarios(
    seed: int = 23,
    vehicle_count: int = 4,
    profile_count: int = 3,
    samples_per_group_per_role: int = 3,
    min_magnitude: float = 0.05,
    max_magnitude: float = 0.15,
    perturb: bool = True,
) -> List[ScenarioConfig]:
    profiles = make_proxy_perturbation_profiles(
        seed=seed,
        profile_count=profile_count,
        min_magnitude=min_magnitude,
        max_magnitude=max_magnitude,
    )
    return make_ds1_proxy_scenarios_from_profiles(
        profiles=profiles,
        seed=seed,
        vehicle_count=vehicle_count,
        samples_per_group_per_role=samples_per_group_per_role,
        perturb=perturb,
    )


def make_ds1_proxy_scenarios_from_profiles(
    profiles: List[Dict[str, Any]],
    seed: int = 23,
    vehicle_count: int = 4,
    samples_per_group_per_role: int = 3,
    perturb: bool = True,
) -> List[ScenarioConfig]:
    matrix = make_ds1_scenario_matrix()
    target_vehicle = make_vehicle_config_variants(vehicle_count)[-1]
    by_group: Dict[str, List[Tuple[int, Dict[str, str]]]] = {}
    for idx, item in enumerate(matrix):
        by_group.setdefault(item["scenario_group"], []).append((idx, item))

    roles = [
        ("target_train", "fine-tune"),
        ("target_validation", "validation"),
        ("target_test", "test-window"),
    ]
    scenarios: List[ScenarioConfig] = []
    for profile_idx, profile in enumerate(profiles):
        vehicle = (
            _apply_proxy_perturbation(target_vehicle, profile)
            if perturb
            else deepcopy(target_vehicle)
        )
        for role_idx, (target_role, split_role) in enumerate(roles):
            for group_idx, (group, group_items) in enumerate(sorted(by_group.items())):
                selected = _balanced_group_sample(
                    group_items,
                    samples_per_group_per_role,
                    seed
                    + profile_idx * 10000
                    + role_idx * 1000
                    + group_idx * 100,
                )
                for local_idx, (matrix_idx, item) in enumerate(selected):
                    target_window_id = "tw_%s_%s_%s_%s_%02d" % (
                        vehicle.vehicle_config_id,
                        profile["profile_id"],
                        target_role,
                        group.lower().replace("-", "_"),
                        local_idx,
                    )
                    fine_tune_bucket = (
                        _fine_tune_bucket(local_idx)
                        if target_role == "target_train"
                        else None
                    )
                    scenario = _ds1_scenario_from_matrix_item(
                        item,
                        vehicle,
                        split_role,
                        target_window_id,
                        fine_tune_bucket,
                        seed + matrix_idx + profile_idx * 10000 + role_idx * 1000,
                        matrix_idx,
                        perturbation_profile_id=profile["profile_id"],
                        target_window_role=target_role,
                    )
                    scenario.teacher_feature_flags.update(
                        {
                            "sim_to_real_proxy": True,
                            "perturbation_profile_id": profile["profile_id"],
                            "target_window_role": target_role,
                            "proxy_perturbed": perturb,
                        }
                    )
                    scenarios.append(scenario)
    return scenarios


def _balanced_group_sample(
    group_items: List[Tuple[int, Dict[str, str]]],
    count: int,
    seed: int,
) -> List[Tuple[int, Dict[str, str]]]:
    if count >= len(group_items):
        return group_items
    rng = np.random.default_rng(seed)
    chosen: List[Tuple[int, Dict[str, str]]] = []
    seen = set()

    def add_candidate(candidate: Tuple[int, Dict[str, str]]) -> None:
        scenario_id = candidate[1]["scenario_id"]
        if scenario_id not in seen:
            chosen.append(candidate)
            seen.add(scenario_id)

    long_ids = [x[0] for x in LONGITUDINAL_FACTORS]
    lat_ids = [x[0] for x in LATERAL_FACTORS]
    for long_id in long_ids:
        candidates = [
            item for item in group_items if item[1]["longitudinal_factor_id"] == long_id
        ]
        add_candidate(candidates[int(rng.integers(0, len(candidates)))])
    for lat_id in lat_ids:
        candidates = [
            item for item in group_items if item[1]["lateral_factor_id"] == lat_id
        ]
        add_candidate(candidates[int(rng.integers(0, len(candidates)))])
    remaining = [item for item in group_items if item[1]["scenario_id"] not in seen]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, count - len(chosen))])
    return chosen[:count]


def _ds1_scenario_from_matrix_item(
    item: Dict[str, str],
    vehicle: VehicleConfig,
    split_role: str,
    target_window_id: Optional[str],
    fine_tune_bucket: Optional[str],
    seed: int,
    matrix_idx: int,
    perturbation_profile_id: Optional[str] = None,
    target_window_role: Optional[str] = None,
) -> ScenarioConfig:
    road_id = item["road_factor_id"]
    if item["scenario_group"] == "CG-SINGLE":
        surface = dict(SINGLE_ROAD_FACTORS)[road_id]
        road = RoadProfile(
            "single",
            surface,
            single_surface=surface,
            factor_id_override=road_id,
            scenario_group=item["scenario_group"],
        )
    elif item["scenario_group"] == "CG-SPLIT":
        left, right = {
            road_id: (left, right)
            for road_id, left, right in SPLIT_ROAD_FACTORS
        }[road_id]
        road = RoadProfile(
            "split",
            "split",
            split_left=left,
            split_right=right,
            factor_id_override=road_id,
            scenario_group=item["scenario_group"],
        )
    else:
        start, end = {
            road_id: (start, end)
            for road_id, start, end in TRANSITION_ROAD_FACTORS
        }[road_id]
        road = RoadProfile(
            "transition",
            "transition",
            transition_from=start,
            transition_to=end,
            transition_start_s=1.5,
            transition_duration_s=1.5,
            factor_id_override=road_id,
            scenario_group=item["scenario_group"],
        )
    control = ControlScript(
        longitudinal=item["longitudinal"],
        lateral=item["lateral"],
        initial_speed_mps=12.0 + (seed % 5),
        longitudinal_id=item["longitudinal_factor_id"],
        lateral_id=item["lateral_factor_id"],
    )
    return ScenarioConfig(
        scenario_id=item["scenario_id"],
        vehicle_config=vehicle,
        road_profile=road,
        control_script=control,
        split_metadata=SplitMetadata(
            split_role=split_role,
            target_window_id=target_window_id,
            fine_tune_data_bucket=fine_tune_bucket,
        ),
        validation_case=_validation_case(item),
        seed=seed,
        scenario_group=item["scenario_group"],
        road_factor_id=item["road_factor_id"],
        longitudinal_factor_id=item["longitudinal_factor_id"],
        lateral_factor_id=item["lateral_factor_id"],
        full_matrix_index=matrix_idx,
        perturbation_profile_id=perturbation_profile_id,
        target_window_role=target_window_role,
    )


def _ds2_extreme_scenario_from_item(
    item: Dict[str, str],
    vehicle: VehicleConfig,
    split_role: str,
    seed: int,
    matrix_idx: int,
) -> ScenarioConfig:
    road_type = item["road_type"]
    if road_type == "single":
        road = RoadProfile(
            "single",
            item["surface_a"],
            single_surface=item["surface_a"],
            roughness_amp_m=0.006,
            factor_id_override=item["road_factor_id"],
            scenario_group=item["scenario_group"],
        )
    elif road_type == "split":
        road = RoadProfile(
            "split",
            "split",
            split_left=item["surface_a"],
            split_right=item["surface_b"],
            roughness_amp_m=0.007,
            factor_id_override=item["road_factor_id"],
            scenario_group=item["scenario_group"],
        )
    else:
        road = RoadProfile(
            "transition",
            "transition",
            transition_from=item["surface_a"],
            transition_to=item["surface_b"],
            transition_start_s=0.9,
            transition_duration_s=1.0,
            roughness_amp_m=0.007,
            factor_id_override=item["road_factor_id"],
            scenario_group=item["scenario_group"],
        )
    control = ControlScript(
        longitudinal=item["longitudinal"],
        lateral=item["lateral"],
        initial_speed_mps=18.0 + float(seed % 5),
        longitudinal_id=item["longitudinal_factor_id"],
        lateral_id=item["lateral_factor_id"],
    )
    scenario_id = "%s-%s-%s" % (
        item["scenario_id"],
        vehicle.vehicle_config_id,
        split_role,
    )
    return ScenarioConfig(
        scenario_id=scenario_id,
        vehicle_config=vehicle,
        road_profile=road,
        control_script=control,
        actuator_profile=ActuatorProfile(steering_saturation_rad=0.72, sensor_delay_steps=1),
        sensor_profile=SensorProfile(noise_scale=1.15, timestamp_jitter_std_s=0.0002),
        split_metadata=SplitMetadata(split_role=split_role),
        validation_case="extreme_handling",
        seed=seed,
        scenario_group=item["scenario_group"],
        road_factor_id=item["road_factor_id"],
        longitudinal_factor_id=item["longitudinal_factor_id"],
        lateral_factor_id=item["lateral_factor_id"],
        full_matrix_index=matrix_idx,
    )


def _apply_proxy_perturbation(
    vehicle: VehicleConfig,
    profile: Dict[str, Any],
) -> VehicleConfig:
    perturbed = deepcopy(vehicle)
    hidden = perturbed.hidden_params
    scales = profile["scales"]
    signs = profile["signs"]
    magnitudes = profile["magnitudes"]

    hidden["mass_true"] *= scales["mass"]
    for key in ["Ix_true", "Iy_true", "Iz_true"]:
        hidden[key] *= scales["inertia"]
    hidden["cg_x_true"] *= scales["cg"]
    hidden["cg_z_true"] *= 1.0 + signs["cg"] * min(magnitudes["cg"], 0.10)

    hidden["cornering_stiffness_front"] *= scales["tire"]
    hidden["cornering_stiffness_rear"] *= scales["tire"]
    hidden["suspension_longitudinal_transfer_scale"] = scales["suspension"]
    hidden["suspension_lateral_transfer_scale"] = scales["suspension"]

    hidden["steering_tau_true"] *= scales["actuator"]
    hidden["drive_tau_true"] *= scales["actuator"]
    hidden["brake_tau_true"] *= scales["actuator"]

    hidden["sensor_bias"] = {
        key: float(value) * scales["sensor"]
        for key, value in hidden["sensor_bias"].items()
    }
    hidden["sensor_noise_std"] = {
        key: float(value) * (1.0 + 0.5 * signs["sensor"] * magnitudes["sensor"])
        for key, value in hidden["sensor_noise_std"].items()
    }
    hidden["proxy_perturbation_profile"] = profile
    hidden["proxy_base_vehicle_config_id"] = vehicle.vehicle_config_id
    hidden["proxy_base_vehicle_id"] = vehicle.vehicle_id
    perturbed.vehicle_id = "%s_%s" % (vehicle.vehicle_id, profile["profile_id"])
    perturbed.vehicle_config_id = vehicle.vehicle_config_id
    return perturbed


def _validation_case(item: Dict[str, str]) -> str:
    if item["longitudinal"] == "large_braking" and item["lateral"] == "no_steer":
        if item["road_factor_id"] == "SP02":
            return "split_mu_brake_left_high"
        if item["road_factor_id"] == "SP08":
            return "split_mu_brake_right_high"
        if item["road_factor_id"] in ["R00", "R01"]:
            return "braking"
    if (
        item["road_factor_id"] in ["R00", "R01"]
        and item["longitudinal"] == "const_v"
        and item["lateral"] in ["small_left_steer", "large_left_steer"]
    ):
        return "left_turn"
    if (
        item["road_factor_id"] in ["R00", "R01"]
        and item["longitudinal"] == "const_v"
        and item["lateral"] in ["small_right_steer", "large_right_steer"]
    ):
        return "right_turn"
    if item["scenario_group"] == "CG-TRANSITION":
        return "transition_mu"
    return "general"


def _split_role(
    vehicle_idx: int,
    vehicle_count: int,
    group: str,
    local_idx: int,
) -> str:
    if vehicle_idx == vehicle_count - 1:
        return "held-out" if local_idx % 2 == 0 else "test-window"
    if group == "CG-TRANSITION" and local_idx % 5 == 0:
        return "test"
    if group == "CG-SPLIT" and local_idx % 4 == 0:
        return "validation"
    if local_idx % 7 == 0:
        return "fine-tune"
    return "train"


def _fine_tune_bucket(local_idx: int) -> str:
    buckets = ["FTD1", "FTD2", "FTD3", "FTD4", "FTD5"]
    return buckets[local_idx % len(buckets)]
