import argparse
import json
import os
from typing import Any, Dict, List

from teacher_simulator.config import load_teacher_config, load_yaml
from teacher_simulator.export import export_dataset
from teacher_simulator.scenario import (
    make_ds0_scenarios,
    make_ds1_scenario_matrix,
    make_ds1_scenarios,
)
from teacher_simulator.simulator import TeacherSimulator


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
    else:
        raise ValueError("unsupported scenario_set=%s" % scenario_set)
    episodes = [sim.run_episode(scenario) for scenario in scenarios]
    manifest = export_dataset(
        episodes,
        out_dir,
        dataset_id=cfg.dataset_id,
        schema_version=cfg.schema_version,
        teacher_model_version=cfg.teacher_model_version,
    )
    if scenario_set == "ds1":
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
