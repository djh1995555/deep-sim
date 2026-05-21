import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from teacher_simulator.export import load_dataset, make_model_input
from teacher_simulator.vehicle_params import WHEEL_ORDER


OBS_REQUIRED = [
    "timestamp",
    "vx",
    "vy",
    "roll",
    "pitch",
    "yaw",
    "p",
    "q",
    "r",
    "omega_fl",
    "omega_fr",
    "omega_rl",
    "omega_rr",
    "sw_angle",
    "tau_drv_obs_fl",
    "tau_drv_obs_fr",
    "tau_drv_obs_rl",
    "tau_drv_obs_rr",
    "tau_brk_obs_fl",
    "tau_brk_obs_fr",
    "tau_brk_obs_rl",
    "tau_brk_obs_rr",
    "steer_cmd",
    "throttle_cmd",
    "brake_cmd",
]

CTX_REQUIRED = [
    "wheelbase",
    "track_front",
    "track_rear",
    "wheel_radius",
    "wheel_order",
    "steering_layout",
    "drive_layout",
    "brake_layout",
]

PRIOR_REQUIRED = [
    "mass_nominal",
    "Ix_nominal",
    "Iy_nominal",
    "Iz_nominal",
    "cg_x_nominal",
    "cg_z_nominal",
    "tau_steer_nominal",
]

AUX_REQUIRED = [
    "Fz_true_i",
    "Fx_true_i",
    "Fy_true_i",
    "mu_true_i",
    "slip_ratio_true_i",
    "slip_angle_true_i",
    "friction_usage_i",
    "delta_eff_i",
    "suspension_states",
    "unsprung_states",
    "camber_true_i",
    "toe_true_i",
    "actuator_delay_states",
    "tau_drv_true_i",
    "tau_brk_true_i",
    "road_surface_labels",
    "teacher_hidden_params",
]

META_REQUIRED = [
    "episode_id",
    "vehicle_id",
    "vehicle_family",
    "vehicle_config_id",
    "scenario_id",
    "road_type",
    "mu_pattern",
    "control_script",
    "seed",
    "duration_s",
    "dt",
    "teacher_model_version",
    "sensor_noise_profile",
    "actuator_profile",
    "torque_observability_mode",
    "teacher_feature_flags",
    "vehicle_internal_params_hash_algo",
    "vehicle_internal_params_hash",
    "observable_fields",
    "teacher_only_fields",
    "split_role",
    "target_window_id",
    "fine_tune_data_bucket",
]


@dataclass
class ValidationReport:
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.passed = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


class TeacherEpisodeValidator:
    def validate(self, episode: Dict[str, Any]) -> ValidationReport:
        report = ValidationReport(passed=True)
        self._check_required(episode, report)
        if report.passed:
            self._check_shapes_and_finiteness(episode, report)
            self._check_context_and_metadata(episode, report)
            self._check_leakage(episode, report)
        return report

    def validate_dataset(self, dataset_dir: str) -> ValidationReport:
        report = ValidationReport(passed=True)
        episodes = load_dataset(dataset_dir)
        if not episodes:
            report.add_error("dataset has no episodes")
            return report
        episode_reports = [self.validate(ep) for ep in episodes]
        for idx, ep_report in enumerate(episode_reports):
            if not ep_report.passed:
                for err in ep_report.errors:
                    report.add_error("episode %d: %s" % (idx, err))
        sign_report = self._check_dataset_signs(episodes)
        for err in sign_report.errors:
            report.add_error(err)
        report.warnings.extend(sign_report.warnings)
        coverage = self._coverage_metrics(episodes)
        report.metrics.update(coverage)
        report.metrics["episode_count"] = len(episodes)
        report.metrics["schema_checks_passed"] = int(report.passed)
        report.metrics["leakage_checks_passed"] = int(
            all(self._leakage_ok(ep) for ep in episodes)
        )
        report.metrics["sign_checks_passed"] = int(len(sign_report.errors) == 0)
        return report

    def _check_required(self, episode: Dict[str, Any], report: ValidationReport) -> None:
        for key in [
            "metadata",
            "fixed_vehicle_context",
            "nominal_physics_prior",
            "time_series_observable",
            "teacher_aux_labels",
        ]:
            if key not in episode:
                report.add_error("missing top-level key %s" % key)
                return
        for key in OBS_REQUIRED:
            if key not in episode["time_series_observable"]:
                report.add_error("missing observable %s" % key)
        for key in CTX_REQUIRED:
            if key not in episode["fixed_vehicle_context"]:
                report.add_error("missing fixed_vehicle_context %s" % key)
        for key in PRIOR_REQUIRED:
            if key not in episode["nominal_physics_prior"]:
                report.add_error("missing nominal_physics_prior %s" % key)
        for key in AUX_REQUIRED:
            if key not in episode["teacher_aux_labels"]:
                report.add_error("missing teacher_aux_label %s" % key)
        for key in META_REQUIRED:
            if key not in episode["metadata"]:
                report.add_error("missing metadata %s" % key)

    def _check_shapes_and_finiteness(
        self, episode: Dict[str, Any], report: ValidationReport
    ) -> None:
        obs = episode["time_series_observable"]
        timestamp = np.asarray(obs["timestamp"])
        if timestamp.ndim != 1 or len(timestamp) < 2:
            report.add_error("timestamp must be 1D with length >= 2")
            return
        if np.any(np.diff(timestamp) <= 0):
            report.add_error("timestamp must be strictly increasing")
        length = len(timestamp)
        for key in OBS_REQUIRED:
            arr = np.asarray(obs[key])
            if arr.shape != (length,):
                report.add_error("observable %s shape %s != [%d]" % (key, arr.shape, length))
            if not np.all(np.isfinite(arr)):
                report.add_error("observable %s has NaN/Inf" % key)

        aux_shapes = {
            "Fz_true_i": (length, 4),
            "Fx_true_i": (length, 4),
            "Fy_true_i": (length, 4),
            "mu_true_i": (length, 4),
            "slip_ratio_true_i": (length, 4),
            "slip_angle_true_i": (length, 4),
            "friction_usage_i": (length, 4),
            "delta_eff_i": (length, 2),
            "tau_drv_true_i": (length, 4),
            "tau_brk_true_i": (length, 4),
        }
        aux = episode["teacher_aux_labels"]
        for key, expected in aux_shapes.items():
            arr = np.asarray(aux[key])
            if arr.shape != expected:
                report.add_error("aux %s shape %s != %s" % (key, arr.shape, expected))
            if not np.all(np.isfinite(arr)):
                report.add_error("aux %s has NaN/Inf" % key)
        if np.any(np.asarray(aux["Fz_true_i"]) < -1e-3):
            report.add_error("Fz_true_i contains negative values")
        usage = np.asarray(aux["friction_usage_i"])
        if not np.all(np.isfinite(usage)):
            report.add_error("friction_usage_i has NaN/Inf")

    def _check_context_and_metadata(
        self, episode: Dict[str, Any], report: ValidationReport
    ) -> None:
        ctx = episode["fixed_vehicle_context"]
        if list(ctx["wheel_order"]) != WHEEL_ORDER:
            report.add_error("wheel_order must be %s" % WHEEL_ORDER)
        for key in ["wheelbase", "track_front", "track_rear", "wheel_radius"]:
            if float(ctx[key]) <= 0:
                report.add_error("%s must be > 0" % key)
        if float(ctx["steering_layout"]["steering_ratio_nominal"]) <= 0:
            report.add_error("steering_ratio_nominal must be > 0")
        prior = episode["nominal_physics_prior"]
        for key in PRIOR_REQUIRED:
            if key not in ["cg_x_nominal"] and float(prior[key]) <= 0:
                report.add_error("%s must be > 0" % key)
        meta = episode["metadata"]
        if meta["torque_observability_mode"] not in [
            "true_per_wheel_sensor",
            "actuator_estimate",
            "command_only_projection",
        ]:
            report.add_error("invalid torque_observability_mode")
        if meta["vehicle_internal_params_hash_algo"] != "sha256":
            report.add_error("vehicle_internal_params_hash_algo must be sha256")
        if not meta["vehicle_internal_params_hash"]:
            report.add_error("vehicle_internal_params_hash is empty")

    def _check_leakage(self, episode: Dict[str, Any], report: ValidationReport) -> None:
        if not self._leakage_ok(episode):
            report.add_error("teacher_aux_labels or metadata leaked into model_input")

    def _leakage_ok(self, episode: Dict[str, Any]) -> bool:
        model_input = make_model_input(episode)
        forbidden = set(episode["teacher_aux_labels"].keys()) | set(
            episode["metadata"].keys()
        )
        return forbidden.isdisjoint(set(model_input.keys()))

    def _check_dataset_signs(self, episodes: List[Dict[str, Any]]) -> ValidationReport:
        report = ValidationReport(passed=True)
        for episode in episodes:
            case = episode["metadata"].get("validation_case", "general")
            aux = episode["teacher_aux_labels"]
            obs = episode["time_series_observable"]
            fz = np.asarray(aux["Fz_true_i"])
            if case == "braking":
                if not self._front_load_exceeds_rear_during_event(fz):
                    report.add_error("%s braking load transfer sign failed" % episode["metadata"]["episode_id"])
            elif case == "left_turn":
                if not self._right_load_exceeds_left_during_event(fz):
                    report.add_error("%s left-turn lateral load transfer sign failed" % episode["metadata"]["episode_id"])
                if np.nanmax(obs["r"]) <= 0.0:
                    report.add_error("%s left-turn yaw sign failed" % episode["metadata"]["episode_id"])
            elif case == "right_turn":
                if np.nanmin(obs["r"]) >= 0.0:
                    report.add_error("%s right-turn yaw sign failed" % episode["metadata"]["episode_id"])
            elif case == "split_mu_brake_left_high":
                if np.nanmax(obs["r"]) <= 0.0:
                    report.add_error("%s split-mu yaw sign failed" % episode["metadata"]["episode_id"])
            elif case == "split_mu_brake_right_high":
                if np.nanmin(obs["r"]) >= 0.0:
                    report.add_error("%s split-mu yaw sign failed" % episode["metadata"]["episode_id"])
        return report

    @staticmethod
    def _front_load_exceeds_rear_during_event(fz: np.ndarray) -> bool:
        start = max(1, int(0.35 * len(fz)))
        front = np.mean(fz[start:, 0] + fz[start:, 1])
        rear = np.mean(fz[start:, 2] + fz[start:, 3])
        return front > rear

    @staticmethod
    def _right_load_exceeds_left_during_event(fz: np.ndarray) -> bool:
        start = max(1, int(0.35 * len(fz)))
        left = np.mean(fz[start:, 0] + fz[start:, 2])
        right = np.mean(fz[start:, 1] + fz[start:, 3])
        return right > left

    @staticmethod
    def _coverage_metrics(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        road_types = sorted(set(ep["metadata"]["road_type"] for ep in episodes))
        cases = sorted(set(ep["metadata"].get("validation_case", "") for ep in episodes))
        scenario_groups = sorted(
            set(ep["metadata"].get("scenario_group", "") for ep in episodes)
        )
        split_roles = sorted(set(ep["metadata"].get("split_role", "") for ep in episodes))
        vehicle_configs = sorted(
            set(ep["metadata"].get("vehicle_config_id", "") for ep in episodes)
        )
        vehicle_families = sorted(
            set(ep["metadata"].get("vehicle_family", "") for ep in episodes)
        )
        road_factors = sorted(
            set(ep["metadata"].get("road_factor_id", "") for ep in episodes)
        )
        target_windows = sorted(
            set(
                ep["metadata"].get("target_window_id")
                for ep in episodes
                if ep["metadata"].get("target_window_id")
            )
        )
        fine_tune_buckets = sorted(
            set(
                ep["metadata"].get("fine_tune_data_bucket")
                for ep in episodes
                if ep["metadata"].get("fine_tune_data_bucket")
            )
        )
        return {
            "road_types": road_types,
            "validation_cases": cases,
            "scenario_groups": scenario_groups,
            "split_roles": split_roles,
            "vehicle_configs": vehicle_configs,
            "vehicle_families": vehicle_families,
            "road_factor_count": len(road_factors),
            "target_window_count": len(target_windows),
            "fine_tune_buckets": fine_tune_buckets,
            "has_single_mu": int("single" in road_types),
            "has_split_mu": int("split" in road_types),
            "has_transition_mu": int("transition" in road_types),
            "has_cg_single": int("CG-SINGLE" in scenario_groups),
            "has_cg_split": int("CG-SPLIT" in scenario_groups),
            "has_cg_transition": int("CG-TRANSITION" in scenario_groups),
            "scenario_groups_covered": len([x for x in scenario_groups if x]),
            "split_roles_covered": len([x for x in split_roles if x]),
            "vehicle_config_count": len([x for x in vehicle_configs if x]),
            "vehicle_family_count": len([x for x in vehicle_families if x]),
            "has_held_out_vehicle_config": int("held-out" in split_roles),
            "has_test_window": int("test-window" in split_roles),
            "has_fine_tune_windows": int(len(target_windows) > 0),
            "fine_tune_bucket_count": len(fine_tune_buckets),
        }


def write_validation_report(report: ValidationReport, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
