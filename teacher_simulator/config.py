from dataclasses import dataclass, field
from typing import Any, Dict

import yaml


@dataclass
class NumericalLimits:
    min_speed_for_slip: float = 0.5
    min_fz: float = 50.0
    max_abs_state: Dict[str, float] = field(
        default_factory=lambda: {
            "vx": 80.0,
            "vy": 30.0,
            "roll": 1.0,
            "pitch": 1.0,
            "r": 3.0,
        }
    )


@dataclass
class ExportConfig:
    include_teacher_aux_labels: bool = True
    include_metadata: bool = True
    resample_policy: str = "hold_last"


@dataclass
class TeacherSimConfig:
    dt_internal: float = 0.01
    dt_export: float = 0.02
    duration_s: float = 12.0
    integrator: str = "semi_implicit_euler"
    gravity: float = 9.81
    coordinate_convention: str = "body_x_forward_y_left_z_up"
    dataset_id: str = "DS0_DEBUG"
    schema_version: str = "v0"
    teacher_model_version: str = "teacher_simulator_v0"
    seed: int = 7
    default_feature_flags: Dict[str, Any] = field(
        default_factory=lambda: {
            "tire_relaxation": True,
            "tire_thermal_wear_pressure": True,
            "suspension_unsprung": True,
            "steering_lag_compliance_backlash_hysteresis": True,
            "drive_brake_lag_saturation_abs": True,
            "aero_wind": True,
            "sensor_noise_delay_filter_quantization": True,
            "generator_version": "teacher_simulator_v0",
            "disabled_features": [],
            "downgrade_reason": "",
        }
    )
    numerical_limits: NumericalLimits = field(default_factory=NumericalLimits)
    export: ExportConfig = field(default_factory=ExportConfig)

    def validate(self) -> None:
        if self.dt_internal <= 0:
            raise ValueError("dt_internal must be > 0")
        if self.dt_export < self.dt_internal:
            raise ValueError("dt_export must be >= dt_internal")
        ratio = self.dt_export / self.dt_internal
        if abs(round(ratio) - ratio) > 1e-9:
            raise ValueError("dt_export / dt_internal must be integer for v0")
        if self.gravity <= 0:
            raise ValueError("gravity must be > 0")
        if self.coordinate_convention != "body_x_forward_y_left_z_up":
            raise ValueError("unsupported coordinate convention")


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")
    return data


def write_yaml(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def config_from_dict(data: Dict[str, Any]) -> TeacherSimConfig:
    cfg = TeacherSimConfig()
    teacher = data.get("teacher", data)

    simple_fields = [
        "dt_internal",
        "dt_export",
        "duration_s",
        "integrator",
        "gravity",
        "coordinate_convention",
        "dataset_id",
        "schema_version",
        "teacher_model_version",
        "seed",
    ]
    for field_name in simple_fields:
        if field_name in teacher:
            setattr(cfg, field_name, teacher[field_name])

    if "default_feature_flags" in teacher:
        cfg.default_feature_flags = _merge_dict(
            cfg.default_feature_flags, teacher["default_feature_flags"]
        )

    if "numerical_limits" in teacher:
        limits = teacher["numerical_limits"]
        for field_name in ["min_speed_for_slip", "min_fz"]:
            if field_name in limits:
                setattr(cfg.numerical_limits, field_name, limits[field_name])
        if "max_abs_state" in limits:
            cfg.numerical_limits.max_abs_state = _merge_dict(
                cfg.numerical_limits.max_abs_state, limits["max_abs_state"]
            )

    if "export" in teacher:
        export = teacher["export"]
        for field_name in [
            "include_teacher_aux_labels",
            "include_metadata",
            "resample_policy",
        ]:
            if field_name in export:
                setattr(cfg.export, field_name, export[field_name])

    cfg.validate()
    return cfg


def load_teacher_config(path: str) -> TeacherSimConfig:
    return config_from_dict(load_yaml(path))
