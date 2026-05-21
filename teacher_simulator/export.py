import json
import os
from typing import Any, Dict, Iterable, List

import numpy as np


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def export_dataset(
    episodes: Iterable[Dict[str, Any]],
    out_dir: str,
    dataset_id: str,
    schema_version: str,
    teacher_model_version: str,
) -> Dict[str, Any]:
    episodes = list(episodes)
    _ensure_dir(out_dir)
    ep_dir = os.path.join(out_dir, "episodes")
    _ensure_dir(ep_dir)
    manifest: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "schema_version": schema_version,
        "teacher_model_version": teacher_model_version,
        "episodes": [],
    }
    for episode in episodes:
        metadata = episode["metadata"]
        episode_id = metadata["episode_id"]
        stem = _safe_stem(episode_id)
        npz_rel = os.path.join("episodes", "%s.npz" % stem)
        json_rel = os.path.join("episodes", "%s.json" % stem)
        arrays = {}
        for key, value in episode["time_series_observable"].items():
            arrays["obs__%s" % key] = value
        for key, value in episode["teacher_aux_labels"].items():
            if isinstance(value, np.ndarray):
                arrays["aux__%s" % key] = value
        np.savez_compressed(os.path.join(out_dir, npz_rel), **arrays)
        json_payload = {
            "metadata": metadata,
            "fixed_vehicle_context": episode["fixed_vehicle_context"],
            "nominal_physics_prior": episode["nominal_physics_prior"],
            "teacher_aux_labels_json": {
                key: value
                for key, value in episode["teacher_aux_labels"].items()
                if not isinstance(value, np.ndarray)
            },
            "array_file": npz_rel,
        }
        with open(os.path.join(out_dir, json_rel), "w", encoding="utf-8") as handle:
            json.dump(json_payload, handle, indent=2, sort_keys=True)
        manifest["episodes"].append(
            {
                "episode_id": episode_id,
                "path": npz_rel,
                "sidecar": json_rel,
                "split_role": metadata["split_role"],
                "scenario_id": metadata["scenario_id"],
                "scenario_group": metadata.get("scenario_group"),
                "road_factor_id": metadata.get("road_factor_id"),
                "longitudinal_factor_id": metadata.get("longitudinal_factor_id"),
                "lateral_factor_id": metadata.get("lateral_factor_id"),
                "vehicle_config_id": metadata["vehicle_config_id"],
                "vehicle_family": metadata.get("vehicle_family"),
                "vehicle_internal_params_hash": metadata[
                    "vehicle_internal_params_hash"
                ],
                "target_window_id": metadata.get("target_window_id"),
                "fine_tune_data_bucket": metadata.get("fine_tune_data_bucket"),
                "perturbation_profile_id": metadata.get("perturbation_profile_id"),
                "target_window_role": metadata.get("target_window_role"),
                "validation_case": metadata.get("validation_case"),
                "road_type": metadata.get("road_type"),
            }
        )

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def load_dataset(dataset_dir: str) -> List[Dict[str, Any]]:
    with open(os.path.join(dataset_dir, "manifest.json"), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    episodes = []
    for item in manifest["episodes"]:
        with open(os.path.join(dataset_dir, item["sidecar"]), "r", encoding="utf-8") as handle:
            sidecar = json.load(handle)
        npz_path = os.path.join(dataset_dir, sidecar["array_file"])
        npz = np.load(npz_path, allow_pickle=False)
        observable = {
            key[len("obs__") :]: npz[key]
            for key in npz.files
            if key.startswith("obs__")
        }
        aux = {
            key[len("aux__") :]: npz[key]
            for key in npz.files
            if key.startswith("aux__")
        }
        aux.update(sidecar.get("teacher_aux_labels_json", {}))
        episodes.append(
            {
                "metadata": sidecar["metadata"],
                "fixed_vehicle_context": sidecar["fixed_vehicle_context"],
                "nominal_physics_prior": sidecar["nominal_physics_prior"],
                "time_series_observable": observable,
                "teacher_aux_labels": aux,
            }
        )
    return episodes


def make_model_input(episode: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "time_series_observable": episode["time_series_observable"],
        "fixed_vehicle_context": episode["fixed_vehicle_context"],
        "nominal_physics_prior": episode["nominal_physics_prior"],
    }


def _safe_stem(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in ["_", "-", "."]:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep)
