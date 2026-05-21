import argparse
import csv
import json
import os
from typing import Any, Dict, List

import numpy as np

from student_model.constants import CONTROL_KEYS, STATE_KEYS
from student_model.data import validate_canonical_dataset


DEFAULT_FIXED_CONTEXT = {
    "wheelbase": 2.85,
    "track_front": 1.62,
    "track_rear": 1.60,
    "wheel_radius": 0.335,
    "wheel_order": ["FL", "FR", "RL", "RR"],
    "drive_layout": {"type": "AWD", "driven_wheels": ["FL", "FR", "RL", "RR"]},
    "brake_layout": {"type": "four_wheel_observed"},
    "steering_layout": {
        "type": "front_wheel_steer",
        "input_signal": "sw_angle",
        "steered_wheels": ["FL", "FR"],
        "steering_ratio_nominal": 14.5,
    },
}

DEFAULT_NOMINAL_PRIOR = {
    "mass_nominal": 1800.0,
    "Ix_nominal": 650.0,
    "Iy_nominal": 2450.0,
    "Iz_nominal": 2750.0,
    "cg_x_nominal": 1.35,
    "cg_z_nominal": 0.56,
    "tau_steer_nominal": 0.11,
}


def _read_json_or_default(path: str, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path:
        return dict(default)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    merged = dict(default)
    merged.update(data)
    return merged


def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _read_csv_columns(path: str) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError("input CSV has no rows: %s" % path)
    columns: Dict[str, List[float]] = {key: [] for key in rows[0].keys()}
    for row in rows:
        for key in columns:
            value = row.get(key, "")
            columns[key].append(float(value) if value not in {"", "nan", "None"} else float("nan"))
    return columns


def _column(columns: Dict[str, List[float]], field_map: Dict[str, str], key: str, default: float) -> np.ndarray:
    candidates = [field_map.get(key, ""), key, "obs__%s" % key]
    for candidate in candidates:
        if candidate and candidate in columns:
            return np.asarray(columns[candidate], dtype=np.float32)
    return np.full((len(next(iter(columns.values()))),), default, dtype=np.float32)


def convert_csv_to_canonical(
    input_csv: str,
    output_dir: str,
    dataset_id: str,
    episode_id: str,
    fixed_context: Dict[str, Any] | None = None,
    nominal_prior: Dict[str, Any] | None = None,
    metadata: Dict[str, Any] | None = None,
    field_map: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    columns = _read_csv_columns(input_csv)
    field_map = field_map or {}
    fixed_context = fixed_context or dict(DEFAULT_FIXED_CONTEXT)
    nominal_prior = nominal_prior or dict(DEFAULT_NOMINAL_PRIOR)
    metadata = metadata or {}
    os.makedirs(os.path.join(output_dir, "episodes"), exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    timestamp = _column(columns, field_map, "timestamp", 0.0).astype(np.float64)
    if np.all(timestamp == 0.0):
        dt = float(metadata.get("dt", 0.04))
        timestamp = np.arange(len(timestamp), dtype=np.float64) * dt
    arrays["obs__timestamp"] = timestamp
    for key in STATE_KEYS:
        default = 10.0 if key == "vx" else 0.0
        arrays["obs__%s" % key] = _column(columns, field_map, key, default)
    for key in CONTROL_KEYS:
        arrays["obs__%s" % key] = _column(columns, field_map, key, 0.0)
    array_rel = "episodes/%s.npz" % episode_id
    array_path = os.path.join(output_dir, array_rel)
    np.savez(array_path, **arrays)
    sample_count = len(timestamp)
    if sample_count < 2:
        raise ValueError("episode must contain at least two samples")
    dt_values = np.diff(timestamp)
    dt = float(np.nanmedian(dt_values)) if len(dt_values) else float(metadata.get("dt", 0.04))
    sidecar_rel = "episodes/%s.json" % episode_id
    sidecar = {
        "array_file": array_rel,
        "fixed_vehicle_context": fixed_context,
        "nominal_physics_prior": nominal_prior,
        "metadata": {
            "episode_id": episode_id,
            "dataset_id": dataset_id,
            "schema_version": "real_adapter_v0",
            "teacher_model_version": "real_data_adapter_v0",
            "dt": dt,
            "duration_s": float(timestamp[-1] - timestamp[0]),
            "sample_count": int(sample_count),
            "split_role": metadata.get("split_role", "train"),
            "scenario_group": metadata.get("scenario_group", "REAL"),
            "vehicle_config_id": metadata.get("vehicle_config_id", "real_vehicle"),
            "vehicle_family": metadata.get("vehicle_family", "real_vehicle"),
            "observable_fields": ["timestamp"] + STATE_KEYS + CONTROL_KEYS,
            "teacher_only_fields": [],
            **metadata,
        },
        "teacher_aux_labels_json": {},
    }
    _write_json(os.path.join(output_dir, sidecar_rel), sidecar)
    manifest = {
        "dataset_id": dataset_id,
        "schema_version": "real_adapter_v0",
        "episodes": [
            {
                "episode_id": episode_id,
                "path": array_rel,
                "sidecar": sidecar_rel,
                "split_role": sidecar["metadata"].get("split_role", "train"),
                "scenario_group": sidecar["metadata"].get("scenario_group", "REAL"),
                "vehicle_config_id": sidecar["metadata"].get("vehicle_config_id", "real_vehicle"),
                "vehicle_family": sidecar["metadata"].get("vehicle_family", "real_vehicle"),
                "fine_tune_data_bucket": sidecar["metadata"].get("fine_tune_data_bucket"),
                "target_window_role": sidecar["metadata"].get("target_window_role"),
            }
        ],
    }
    _write_json(os.path.join(output_dir, "manifest.json"), manifest)
    summary = validate_canonical_dataset(output_dir)
    summary.update({"dataset_id": dataset_id, "episode_id": episode_id, "output_dir": output_dir})
    _write_json(os.path.join(output_dir, "adapter_summary.json"), summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-id", default="REAL_V0")
    parser.add_argument("--episode-id", default="real_episode_000")
    parser.add_argument("--fixed-context-json", default="")
    parser.add_argument("--nominal-prior-json", default="")
    parser.add_argument("--metadata-json", default="")
    parser.add_argument("--field-map-json", default="")
    args = parser.parse_args()
    fixed_context = _read_json_or_default(args.fixed_context_json, DEFAULT_FIXED_CONTEXT)
    nominal_prior = _read_json_or_default(args.nominal_prior_json, DEFAULT_NOMINAL_PRIOR)
    metadata = _read_json_or_default(args.metadata_json, {}) if args.metadata_json else {}
    field_map = _read_json_or_default(args.field_map_json, {}) if args.field_map_json else {}
    summary = convert_csv_to_canonical(
        args.input_csv,
        args.output_dir,
        args.dataset_id,
        args.episode_id,
        fixed_context=fixed_context,
        nominal_prior=nominal_prior,
        metadata=metadata,
        field_map=field_map,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
