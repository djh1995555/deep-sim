import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np

from simulator.vehicle_model.config import load_teacher_config, load_yaml
from simulator.vehicle_model.export import export_dataset
from simulator.vehicle_model.scenario import (
    make_ds0_scenarios,
    make_ds2_extreme_matrix,
    make_ds2_extreme_scenarios,
    make_ds1_proxy_scenarios_from_profiles,
    make_ds1_scenario_matrix,
    make_ds1_scenarios,
    make_proxy_perturbation_profiles,
)
from simulator.vehicle_model.model import TeacherSimulator


def generate_dataset(config_path: str, out_dir: str) -> Dict[str, Any]:
    raw = load_yaml(config_path)
    cfg = load_teacher_config(config_path)
    sim = TeacherSimulator(cfg)
    scenario_set = raw.get("scenario_set", "ds0")
    ds1_cfg = raw.get("ds1", {})
    matrix: List[Dict[str, str]] = []
    if scenario_set == "ds0":
        scenarios = make_ds0_scenarios(cfg.seed)
    elif scenario_set == "ds1":
        matrix = make_ds1_scenario_matrix()
        scenarios = make_ds1_scenarios(
            seed=cfg.seed,
            vehicle_count=int(ds1_cfg.get("vehicle_count", 4)),
            samples_per_group_per_vehicle=int(
                ds1_cfg.get("samples_per_group_per_vehicle", 10)
            ),
        )
    elif scenario_set == "ds1_proxy":
        matrix = make_ds1_scenario_matrix()
        proxy_cfg = raw.get("proxy", {})
        profiles = make_proxy_perturbation_profiles(
            seed=cfg.seed,
            profile_count=int(proxy_cfg.get("profile_count", 3)),
            min_magnitude=float(proxy_cfg.get("min_magnitude", 0.05)),
            max_magnitude=float(proxy_cfg.get("max_magnitude", 0.15)),
        )
        scenarios = make_ds1_proxy_scenarios_from_profiles(
            profiles=profiles,
            seed=cfg.seed,
            vehicle_count=int(ds1_cfg.get("vehicle_count", 4)),
            samples_per_group_per_role=int(
                proxy_cfg.get("samples_per_group_per_role", 3)
            ),
            perturb=True,
        )
        reference_scenarios = make_ds1_proxy_scenarios_from_profiles(
            profiles=profiles,
            seed=cfg.seed,
            vehicle_count=int(ds1_cfg.get("vehicle_count", 4)),
            samples_per_group_per_role=int(
                proxy_cfg.get("samples_per_group_per_role", 3)
            ),
            perturb=False,
        )
    elif scenario_set == "ds2_extreme":
        ds2_cfg = raw.get("ds2", {})
        matrix = make_ds2_extreme_matrix()
        scenarios = make_ds2_extreme_scenarios(
            seed=cfg.seed,
            vehicle_count=int(ds2_cfg.get("vehicle_count", 3)),
            samples_per_vehicle=int(ds2_cfg.get("samples_per_vehicle", 6)),
        )
    else:
        raise ValueError("unsupported scenario_set=%s" % scenario_set)
    episodes = [sim.run_episode(scenario) for scenario in scenarios]
    reference_episodes = []
    if scenario_set == "ds1_proxy":
        reference_episodes = [sim.run_episode(scenario) for scenario in reference_scenarios]
    manifest = export_dataset(
        episodes,
        out_dir,
        dataset_id=cfg.dataset_id,
        schema_version=cfg.schema_version,
        teacher_model_version=cfg.teacher_model_version,
    )
    if scenario_set in ["ds1", "ds1_proxy", "ds2_extreme"]:
        _write_json(os.path.join(out_dir, "scenario_matrix.json"), matrix)
        _write_json(os.path.join(out_dir, "split_manifest.json"), _split_manifest(manifest))
        _write_json(
            os.path.join(out_dir, "scenario_coverage_report.json"),
            _scenario_coverage_report(matrix, manifest),
        )
        _write_json(
            os.path.join(out_dir, "dataset_qa_report.json"),
            _dataset_qa_report(matrix, manifest),
        )
    if scenario_set == "ds1_proxy":
        _write_json(os.path.join(out_dir, "perturbation_profiles.json"), profiles)
        _write_json(
            os.path.join(out_dir, "proxy_target_windows.json"),
            _proxy_target_window_report(manifest),
        )
        _write_json(
            os.path.join(out_dir, "proxy_distribution_report.json"),
            _proxy_distribution_report(profiles, episodes, reference_episodes),
        )
    with open(os.path.join(out_dir, "generation_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": config_path,
                "out_dir": out_dir,
                "episode_count": len(episodes),
                "full_matrix_count": len(matrix),
                "scenario_set": scenario_set,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return manifest


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _split_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    by_split: Dict[str, List[str]] = {}
    by_target_window: Dict[str, List[str]] = {}
    for item in manifest["episodes"]:
        by_split.setdefault(item["split_role"], []).append(item["episode_id"])
        target_window = item.get("target_window_id")
        if target_window:
            by_target_window.setdefault(target_window, []).append(item["episode_id"])
    return {
        "dataset_id": manifest["dataset_id"],
        "split_roles": by_split,
        "target_windows": by_target_window,
    }


def _scenario_coverage_report(
    matrix: List[Dict[str, str]],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    full_groups: Dict[str, int] = {}
    for item in matrix:
        full_groups[item["scenario_group"]] = full_groups.get(item["scenario_group"], 0) + 1
    sampled_groups: Dict[str, int] = {}
    sampled_road_factors: Dict[str, int] = {}
    sampled_longitudinal: Dict[str, int] = {}
    sampled_lateral: Dict[str, int] = {}
    for item in manifest["episodes"]:
        sampled_groups[item["scenario_group"]] = sampled_groups.get(item["scenario_group"], 0) + 1
        sampled_road_factors[item["road_factor_id"]] = sampled_road_factors.get(item["road_factor_id"], 0) + 1
        sampled_longitudinal[item["longitudinal_factor_id"]] = sampled_longitudinal.get(item["longitudinal_factor_id"], 0) + 1
        sampled_lateral[item["lateral_factor_id"]] = sampled_lateral.get(item["lateral_factor_id"], 0) + 1
    return {
        "full_matrix_count": len(matrix),
        "full_group_counts": full_groups,
        "sampled_episode_count": len(manifest["episodes"]),
        "sampled_group_counts": sampled_groups,
        "sampled_road_factor_count": len(sampled_road_factors),
        "sampled_longitudinal_factor_counts": sampled_longitudinal,
        "sampled_lateral_factor_counts": sampled_lateral,
        "covers_cg_single": sampled_groups.get("CG-SINGLE", 0) > 0,
        "covers_cg_split": sampled_groups.get("CG-SPLIT", 0) > 0,
        "covers_cg_transition": sampled_groups.get("CG-TRANSITION", 0) > 0,
    }


def _dataset_qa_report(
    matrix: List[Dict[str, str]],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    vehicle_configs = sorted(
        set(item.get("vehicle_config_id") for item in manifest["episodes"])
    )
    split_roles = sorted(set(item.get("split_role") for item in manifest["episodes"]))
    target_windows = sorted(
        set(
            item.get("target_window_id")
            for item in manifest["episodes"]
            if item.get("target_window_id")
        )
    )
    fine_tune_buckets = sorted(
        set(
            item.get("fine_tune_data_bucket")
            for item in manifest["episodes"]
            if item.get("fine_tune_data_bucket")
        )
    )
    hashes = [item["vehicle_internal_params_hash"] for item in manifest["episodes"]]
    checks = {
        "full_matrix_has_700_scenarios": len(matrix) == 700,
        "multi_vehicle_config_present": len(vehicle_configs) >= 3,
        "held_out_split_present": "held-out" in split_roles,
        "test_window_split_present": "test-window" in split_roles,
        "fine_tune_split_present": "fine-tune" in split_roles,
        "target_windows_present": len(target_windows) > 0,
        "fine_tune_buckets_present": len(fine_tune_buckets) >= 3,
        "vehicle_hash_present": all(bool(x) for x in hashes),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "vehicle_config_count": len(vehicle_configs),
        "split_roles": split_roles,
        "target_window_count": len(target_windows),
        "fine_tune_buckets": fine_tune_buckets,
    }


def _proxy_target_window_report(manifest: Dict[str, Any]) -> Dict[str, Any]:
    by_window: Dict[str, Dict[str, Any]] = {}
    by_profile: Dict[str, int] = {}
    by_role: Dict[str, int] = {}
    held_out_configs = set()
    for item in manifest["episodes"]:
        profile_id = item.get("perturbation_profile_id")
        target_role = item.get("target_window_role")
        target_window = item.get("target_window_id")
        if profile_id:
            by_profile[profile_id] = by_profile.get(profile_id, 0) + 1
        if target_role:
            by_role[target_role] = by_role.get(target_role, 0) + 1
        if item.get("split_role") in ["fine-tune", "validation", "test-window"]:
            held_out_configs.add(item.get("vehicle_config_id"))
        if target_window:
            by_window.setdefault(
                target_window,
                {
                    "episode_ids": [],
                    "roles": set(),
                    "split_roles": set(),
                    "profile_ids": set(),
                },
            )
            by_window[target_window]["episode_ids"].append(item["episode_id"])
            by_window[target_window]["roles"].add(target_role)
            by_window[target_window]["split_roles"].add(item.get("split_role"))
            by_window[target_window]["profile_ids"].add(profile_id)

    overlapping_windows = {
        window_id: data
        for window_id, data in by_window.items()
        if len(data["roles"]) > 1 or len(data["split_roles"]) > 1
    }
    required_roles = {"target_train", "target_validation", "target_test"}
    checks = {
        "target_roles_present": required_roles.issubset(set(by_role.keys())),
        "target_windows_unique": len(overlapping_windows) == 0,
        "perturbation_profiles_present": len(by_profile) >= 1,
        "held_out_vehicle_config_present": len(held_out_configs) >= 1,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "target_window_count": len(by_window),
        "target_window_role_counts": by_role,
        "perturbation_profile_counts": by_profile,
        "held_out_vehicle_config_count": len(held_out_configs),
        "target_window_overlap_count": len(overlapping_windows),
    }


def _proxy_distribution_report(
    profiles: List[Dict[str, Any]],
    episodes: List[Dict[str, Any]],
    reference_episodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    channels = ["vx", "vy", "r", "roll", "pitch", "omega_fl", "omega_fr", "omega_rl", "omega_rr"]
    shift_scores = []
    by_profile: Dict[str, List[float]] = {}
    for episode, reference in zip(episodes, reference_episodes):
        score_parts = []
        for channel in channels:
            obs = np.asarray(episode["time_series_observable"][channel], dtype=np.float64)
            ref = np.asarray(reference["time_series_observable"][channel], dtype=np.float64)
            denom = float(np.std(ref) + 0.05 * abs(np.mean(ref)) + 1e-3)
            score_parts.append(float(np.mean(np.abs(obs - ref)) / denom))
        score = float(np.mean(score_parts))
        shift_scores.append(score)
        profile_id = episode["metadata"].get("perturbation_profile_id") or "none"
        by_profile.setdefault(profile_id, []).append(score)

    profile_mins = [float(profile["min_abs_magnitude"]) for profile in profiles]
    profile_maxs = [float(profile["max_abs_magnitude"]) for profile in profiles]
    mean_shift = float(np.mean(shift_scores)) if shift_scores else 0.0
    by_profile_mean = {
        profile_id: float(np.mean(values))
        for profile_id, values in sorted(by_profile.items())
    }
    checks = {
        "profile_count_positive": len(profiles) >= 1,
        "profile_magnitudes_within_5_to_15_percent": (
            bool(profile_mins)
            and min(profile_mins) >= 0.05 - 1e-9
            and max(profile_maxs) <= 0.15 + 1e-9
        ),
        "paired_reference_count_matches": len(episodes) == len(reference_episodes),
        "observable_shift_measurable": mean_shift >= 0.015,
        "observable_shift_not_explosive": mean_shift <= 5.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "profile_count": len(profiles),
        "profile_min_abs_magnitude": min(profile_mins) if profile_mins else 0.0,
        "profile_max_abs_magnitude": max(profile_maxs) if profile_maxs else 0.0,
        "proxy_distribution_shift_score": mean_shift,
        "proxy_distribution_shift_by_profile": by_profile_mean,
        "paired_episode_count": min(len(episodes), len(reference_episodes)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        manifest = generate_dataset(args.config, args.out)
    except ValueError as exc:
        print("configuration error: %s" % exc)
        return 2
    except FloatingPointError as exc:
        print("numerical integration failure: %s" % exc)
        return 3
    print(json.dumps({"status": "success", "episodes": len(manifest["episodes"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
