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
    def __init__(self, dataset_dir: str, history_len: int = 8, split_role: str = "train"):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError("TorchEpisodeDataset requires PyTorch") from exc
        self.dataset_dir = dataset_dir
        self.history_len = int(history_len)
        manifest = load_manifest(dataset_dir)
        self.items = [
            item
            for item in manifest.get("episodes", [])
            if item.get("split_role") == split_role
        ]
        if not self.items:
            self.items = manifest.get("episodes", [])

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
