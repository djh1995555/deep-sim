import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from student_model.constants import CONTEXT_KEYS, CONTROL_KEYS, STATE_KEYS


@dataclass(frozen=True)
class EpisodeRecord:
    dataset_dir: str
    episode_id: str
    array_path: str
    sidecar_path: str
    metadata: Dict[str, Any]
    fixed_vehicle_context: Dict[str, Any]
    nominal_physics_prior: Dict[str, Any]


def load_manifest(dataset_dir: str) -> Dict[str, Any]:
    with open(os.path.join(dataset_dir, "manifest.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_episode_record(dataset_dir: str, index: int = 0) -> EpisodeRecord:
    manifest = load_manifest(dataset_dir)
    item = manifest["episodes"][index]
    sidecar_path = os.path.join(dataset_dir, item["sidecar"])
    with open(sidecar_path, "r", encoding="utf-8") as handle:
        sidecar = json.load(handle)
    return EpisodeRecord(
        dataset_dir=dataset_dir,
        episode_id=item["episode_id"],
        array_path=os.path.join(dataset_dir, sidecar["array_file"]),
        sidecar_path=sidecar_path,
        metadata=sidecar["metadata"],
        fixed_vehicle_context=sidecar["fixed_vehicle_context"],
        nominal_physics_prior=sidecar["nominal_physics_prior"],
    )


def episode_arrays(record: EpisodeRecord) -> Tuple[np.ndarray, np.ndarray]:
    npz = np.load(record.array_path, allow_pickle=False)
    states = np.column_stack([npz["obs__%s" % key] for key in STATE_KEYS]).astype(
        np.float32
    )
    controls = np.column_stack([npz["obs__%s" % key] for key in CONTROL_KEYS]).astype(
        np.float32
    )
    return states, controls


def context_vector(record: EpisodeRecord) -> np.ndarray:
    ctx = record.fixed_vehicle_context
    prior = record.nominal_physics_prior
    drive_type = ctx["drive_layout"]["type"]
    brake_type = ctx["brake_layout"]["type"]
    values = [
        float(ctx["wheelbase"]),
        float(ctx["track_front"]),
        float(ctx["track_rear"]),
        float(ctx["wheel_radius"]),
        float(ctx["steering_layout"]["steering_ratio_nominal"]),
        float(prior["mass_nominal"]),
        float(prior["Ix_nominal"]),
        float(prior["Iy_nominal"]),
        float(prior["Iz_nominal"]),
        float(prior["cg_x_nominal"]),
        float(prior["cg_z_nominal"]),
        float(prior["tau_steer_nominal"]),
        float(drive_type == "FWD"),
        float(drive_type == "RWD"),
        float(drive_type == "AWD"),
        float(brake_type == "hydraulic_split"),
        float(brake_type == "brake_by_wire"),
    ]
    if len(values) != len(CONTEXT_KEYS):
        raise ValueError("context vector length mismatch")
    return np.asarray(values, dtype=np.float32)


def observable_history(states: np.ndarray, controls: np.ndarray) -> np.ndarray:
    return np.concatenate([states, controls], axis=-1).astype(np.float32)


def _optional_npz_row(npz: Any, key: str, index: int, default: np.ndarray) -> np.ndarray:
    if key not in npz.files:
        return default.astype(np.float32)
    return np.asarray(npz[key][index], dtype=np.float32)


def _manifest_item_matches(
    item: Dict[str, Any],
    split_role: str,
    fine_tune_buckets: List[str] | None,
    target_window_role: str | None,
    scenario_groups: List[str] | None,
    vehicle_config_ids: List[str] | None,
) -> bool:
    if split_role and item.get("split_role") != split_role:
        return False
    if fine_tune_buckets and item.get("fine_tune_data_bucket") not in fine_tune_buckets:
        return False
    if target_window_role and item.get("target_window_role") != target_window_role:
        return False
    if scenario_groups and item.get("scenario_group") not in scenario_groups:
        return False
    if vehicle_config_ids and item.get("vehicle_config_id") not in vehicle_config_ids:
        return False
    return True


def _filter_description(
    split_role: str,
    fine_tune_buckets: List[str] | None,
    target_window_role: str | None,
    scenario_groups: List[str] | None,
    vehicle_config_ids: List[str] | None,
) -> Dict[str, Any]:
    return {
        "split_role": split_role,
        "fine_tune_buckets": fine_tune_buckets,
        "target_window_role": target_window_role,
        "scenario_groups": scenario_groups,
        "vehicle_config_ids": vehicle_config_ids,
    }


def filter_manifest_items(
    manifest: Dict[str, Any],
    split_role: str = "train",
    fine_tune_buckets: List[str] | None = None,
    target_window_role: str | None = None,
    scenario_groups: List[str] | None = None,
    vehicle_config_ids: List[str] | None = None,
    allow_empty_filter_fallback: bool = False,
) -> List[Tuple[int, Dict[str, Any]]]:
    episodes = list(manifest.get("episodes", []))
    items = [
        (idx, item)
        for idx, item in enumerate(episodes)
        if _manifest_item_matches(
            item,
            split_role,
            fine_tune_buckets,
            target_window_role,
            scenario_groups,
            vehicle_config_ids,
        )
    ]
    if items:
        return items
    if allow_empty_filter_fallback:
        fallback_items = [
            (idx, item)
            for idx, item in enumerate(episodes)
            if not split_role or item.get("split_role") == split_role
        ]
        return fallback_items or list(enumerate(episodes))
    raise ValueError(
        "no episodes match dataset filter: %s"
        % json.dumps(
            _filter_description(
                split_role,
                fine_tune_buckets,
                target_window_role,
                scenario_groups,
                vehicle_config_ids,
            ),
            sort_keys=True,
        )
    )


def validate_canonical_dataset(dataset_dir: str) -> Dict[str, int]:
    manifest = load_manifest(dataset_dir)
    episode_count = len(manifest.get("episodes", []))
    split_roles = {item.get("split_role") for item in manifest.get("episodes", [])}
    first = load_episode_record(dataset_dir, 0)
    states, controls = episode_arrays(first)
    ctx = context_vector(first)
    return {
        "episode_count": episode_count,
        "split_role_count": len(split_roles),
        "state_dim": int(states.shape[1]),
        "control_dim": int(controls.shape[1]),
        "context_dim": int(ctx.shape[0]),
        "sequence_len": int(states.shape[0]),
    }


class TorchEpisodeDataset:
    def __init__(
        self,
        dataset_dir: str,
        history_len: int = 8,
        split_role: str = "train",
        allow_empty_filter_fallback: bool = False,
    ):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError("TorchEpisodeDataset requires PyTorch") from exc
        self.dataset_dir = dataset_dir
        self.history_len = int(history_len)
        manifest = load_manifest(dataset_dir)
        self.items = [
            item
            for _, item in filter_manifest_items(
                manifest,
                split_role=split_role,
                allow_empty_filter_fallback=allow_empty_filter_fallback,
            )
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        import torch

        item = self.items[index]
        manifest = load_manifest(self.dataset_dir)
        global_index = next(
            idx
            for idx, candidate in enumerate(manifest["episodes"])
            if candidate["episode_id"] == item["episode_id"]
        )
        record = load_episode_record(self.dataset_dir, global_index)
        states, controls = episode_arrays(record)
        hist = observable_history(states, controls)
        length = min(self.history_len, len(hist) - 1)
        start = max(0, length - self.history_len)
        window = hist[start:length]
        if len(window) < self.history_len:
            pad = np.repeat(window[:1], self.history_len - len(window), axis=0)
            window = np.vstack([pad, window])
        return {
            "observable_history": torch.from_numpy(window),
            "current_state": torch.from_numpy(states[length]),
            "current_control": torch.from_numpy(controls[length]),
            "target_next_state": torch.from_numpy(states[length + 1]),
            "context": torch.from_numpy(context_vector(record)),
            "dt": torch.tensor(float(record.metadata["dt"]), dtype=torch.float32),
            "episode_id": record.episode_id,
        }


class TorchTransitionDataset:
    def __init__(
        self,
        dataset_dir: str,
        history_len: int = 8,
        split_role: str = "train",
        stride: int = 1,
        max_episodes: int = 0,
        max_samples: int = 0,
        fine_tune_buckets: List[str] | None = None,
        target_window_role: str | None = None,
        scenario_groups: List[str] | None = None,
        vehicle_config_ids: List[str] | None = None,
        allow_empty_filter_fallback: bool = False,
    ):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError("TorchTransitionDataset requires PyTorch") from exc
        self.dataset_dir = dataset_dir
        self.history_len = int(history_len)
        self.stride = max(1, int(stride))
        self.fine_tune_buckets = fine_tune_buckets
        self.target_window_role = target_window_role
        manifest = load_manifest(dataset_dir)
        items = filter_manifest_items(
            manifest,
            split_role=split_role,
            fine_tune_buckets=fine_tune_buckets,
            target_window_role=target_window_role,
            scenario_groups=scenario_groups,
            vehicle_config_ids=vehicle_config_ids,
            allow_empty_filter_fallback=allow_empty_filter_fallback,
        )
        max_episode_count = int(max_episodes or 0)
        if max_episode_count > 0:
            items = items[:max_episode_count]
        self.indices = []
        for global_index, _ in items:
            record = load_episode_record(dataset_dir, global_index)
            states, _ = episode_arrays(record)
            for state_index in range(self.history_len, len(states) - 1, self.stride):
                self.indices.append((global_index, state_index))
                if max_samples and len(self.indices) >= int(max_samples):
                    return

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        import torch

        global_index, state_index = self.indices[index]
        record = load_episode_record(self.dataset_dir, global_index)
        states, controls = episode_arrays(record)
        npz = np.load(record.array_path, allow_pickle=False)
        hist = observable_history(states, controls)
        start = state_index - self.history_len
        window = hist[start:state_index]
        fz_default = np.zeros((4,), dtype=np.float32)
        force_default = np.zeros((4,), dtype=np.float32)
        mu_default = np.full((4,), 0.8, dtype=np.float32)
        delta_default = np.zeros((2,), dtype=np.float32)
        fx = _optional_npz_row(npz, "aux__Fx_true_i", state_index, force_default)
        fy = _optional_npz_row(npz, "aux__Fy_true_i", state_index, force_default)
        return {
            "observable_history": torch.from_numpy(window),
            "current_state": torch.from_numpy(states[state_index]),
            "current_control": torch.from_numpy(controls[state_index]),
            "target_next_state": torch.from_numpy(states[state_index + 1]),
            "context": torch.from_numpy(context_vector(record)),
            "dt": torch.tensor(float(record.metadata["dt"]), dtype=torch.float32),
            "fz_true": torch.from_numpy(
                _optional_npz_row(npz, "aux__Fz_true_i", state_index, fz_default)
            ),
            "tire_force_true": torch.from_numpy(
                np.stack([fx, fy], axis=-1).reshape(-1).astype(np.float32)
            ),
            "mu_true": torch.from_numpy(
                _optional_npz_row(npz, "aux__mu_true_i", state_index, mu_default)
            ),
            "steering_true": torch.from_numpy(
                _optional_npz_row(npz, "aux__delta_eff_i", state_index, delta_default)
            ),
            "episode_id": record.episode_id,
            "state_index": torch.tensor(state_index, dtype=torch.int64),
        }
