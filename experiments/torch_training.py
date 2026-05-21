import importlib.util
import json
import os
import random
from dataclasses import asdict, fields
from typing import Any, Dict, Tuple

import numpy as np

from student_model.data import (
    TorchEpisodeDataset,
    context_vector,
    episode_arrays,
    load_episode_record,
    validate_canonical_dataset,
)


def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _torch_import_status() -> Tuple[Any, str]:
    if importlib.util.find_spec("torch") is None:
        return None, "PyTorch is not installed in the current environment"
    try:
        import torch

        return torch, ""
    except Exception as exc:  # pragma: no cover - depends on local torch install state.
        return None, "PyTorch import failed: %r" % exc


def _set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(torch: Any, cfg: Dict[str, Any]) -> Any:
    requested = cfg.get("device", "cpu")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def _model_config(torch_cfg: Dict[str, Any]) -> Any:
    from student_model.torch_model import HybridStudentConfig

    overrides = torch_cfg.get("model_config", {})
    allowed = {item.name for item in fields(HybridStudentConfig)}
    clean = {key: value for key, value in overrides.items() if key in allowed}
    return HybridStudentConfig(**clean)


def _build_model(torch_cfg: Dict[str, Any], device: Any) -> Any:
    from student_model.torch_model import HybridStudentModel

    model = HybridStudentModel(_model_config(torch_cfg))
    return model.to(device)


def _build_loader(torch: Any, dataset_dir: str, torch_cfg: Dict[str, Any]) -> Any:
    from torch.utils.data import DataLoader, Subset

    dataset = TorchEpisodeDataset(
        dataset_dir,
        history_len=int(torch_cfg.get("history_len", 8)),
        split_role=torch_cfg.get("split_role", "train"),
    )
    max_episodes = int(torch_cfg.get("max_episodes", 0) or 0)
    if max_episodes > 0 and len(dataset) > max_episodes:
        dataset = Subset(dataset, list(range(max_episodes)))
    return DataLoader(
        dataset,
        batch_size=int(torch_cfg.get("batch_size", 4)),
        shuffle=bool(torch_cfg.get("shuffle", False)),
        drop_last=False,
    )


def _batch_to_device(batch: Dict[str, Any], device: Any) -> Dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _forward(model: Any, batch: Dict[str, Any]) -> Dict[str, Any]:
    return model(
        observable_history=batch["observable_history"],
        current_state=batch["current_state"],
        current_control=batch["current_control"],
        context=batch["context"],
        dt=batch["dt"],
    )


def _loss_metrics(torch: Any, out: Dict[str, Any], batch: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    err = out["x_next"] - batch["target_next_state"]
    mse = err.pow(2).mean()
    logvar = out["logvar"].clamp(-8.0, 5.0)
    nll = 0.5 * (err.pow(2) * torch.exp(-logvar) + logvar).mean()
    residual_l2 = out["delta_x"].pow(2).mean()
    loss = mse + 1.0e-4 * residual_l2
    return loss, {
        "one_step_mse": float(mse.detach().cpu()),
        "one_step_nll": float(nll.detach().cpu()),
        "residual_l2": float(residual_l2.detach().cpu()),
    }


def _data_loader_smoke(torch: Any, dataset_dir: str, torch_cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    summary = validate_canonical_dataset(dataset_dir)
    loader = _build_loader(torch, dataset_dir, torch_cfg)
    batch = next(iter(loader))
    finite = bool(torch.isfinite(batch["observable_history"]).all())
    finite = finite and bool(torch.isfinite(batch["target_next_state"]).all())
    metrics = {
        "torch_data_loader_smoke_passed": int(
            len(loader.dataset) > 0
            and finite
            and tuple(batch["observable_history"].shape[-2:])
            == (int(torch_cfg.get("history_len", 8)), summary["state_dim"] + summary["control_dim"])
        ),
        "torch_dataset_sample_count": len(loader.dataset),
        "torch_batch_size": int(batch["observable_history"].shape[0]),
        "torch_observable_dim": int(batch["observable_history"].shape[-1]),
        "torch_state_dim": int(batch["current_state"].shape[-1]),
        "torch_control_dim": int(batch["current_control"].shape[-1]),
        "torch_context_dim": int(batch["context"].shape[-1]),
    }
    return bool(metrics["torch_data_loader_smoke_passed"]), metrics


def _forward_loss_smoke(torch: Any, dataset_dir: str, torch_cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    device = _device(torch, torch_cfg)
    loader = _build_loader(torch, dataset_dir, torch_cfg)
    batch = _batch_to_device(next(iter(loader)), device)
    model = _build_model(torch_cfg, device)
    model.eval()
    with torch.no_grad():
        out = _forward(model, batch)
        _, metrics = _loss_metrics(torch, out, batch)
    passed = (
        bool(torch.isfinite(out["x_next"]).all())
        and out["x_next"].shape == batch["target_next_state"].shape
        and np.isfinite(metrics["one_step_mse"])
    )
    metrics.update(
        {
            "torch_forward_loss_smoke_passed": int(passed),
            "torch_device": str(device),
            "torch_forward_batch_size": int(batch["observable_history"].shape[0]),
        }
    )
    return passed, metrics


def _tiny_overfit(torch: Any, dataset_dir: str, torch_cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    device = _device(torch, torch_cfg)
    loader = _build_loader(torch, dataset_dir, {**torch_cfg, "shuffle": False})
    fixed_batch = _batch_to_device(next(iter(loader)), device)
    model = _build_model(torch_cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(torch_cfg.get("learning_rate", 1.0e-3)),
        weight_decay=float(torch_cfg.get("weight_decay", 0.0)),
    )
    max_steps = int(torch_cfg.get("max_steps", 25))
    grad_clip = float(torch_cfg.get("grad_clip", 1.0))
    model.eval()
    with torch.no_grad():
        out = _forward(model, fixed_batch)
        initial_loss, _ = _loss_metrics(torch, out, fixed_batch)
        initial_loss_value = float(initial_loss.detach().cpu())
    model.train()
    for _ in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        out = _forward(model, fixed_batch)
        loss, _ = _loss_metrics(torch, out, fixed_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        out = _forward(model, fixed_batch)
        final_loss, _ = _loss_metrics(torch, out, fixed_batch)
        final_loss_value = float(final_loss.detach().cpu())
    ratio = final_loss_value / max(initial_loss_value, 1.0e-12)
    threshold = float(torch_cfg.get("success_loss_ratio", 0.995))
    passed = bool(np.isfinite(final_loss_value) and ratio <= threshold)
    metrics = {
        "torch_tiny_overfit_passed": int(passed),
        "torch_tiny_overfit_initial_loss": float(initial_loss_value),
        "torch_tiny_overfit_final_loss": float(final_loss_value),
        "torch_tiny_overfit_loss_ratio": float(ratio),
        "torch_tiny_overfit_steps": max_steps,
        "torch_device": str(device),
    }
    return passed, metrics


def _rollout_smoke(torch: Any, dataset_dir: str, torch_cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    device = _device(torch, torch_cfg)
    model = _build_model(torch_cfg, device)
    model.eval()
    record = load_episode_record(dataset_dir, int(torch_cfg.get("episode_index", 0)))
    states, controls = episode_arrays(record)
    history_len = int(torch_cfg.get("history_len", 8))
    rollout_steps = min(int(torch_cfg.get("rollout_steps", 8)), len(states) - history_len)
    context = torch.as_tensor(context_vector(record)[None, :], dtype=torch.float32, device=device)
    dt = torch.tensor([float(record.metadata["dt"])], dtype=torch.float32, device=device)
    history = np.concatenate([states[:history_len], controls[:history_len]], axis=-1)
    current = torch.as_tensor(states[history_len - 1 : history_len], dtype=torch.float32, device=device)
    preds = []
    targets = []
    with torch.no_grad():
        for step in range(rollout_steps):
            t = history_len - 1 + step
            obs = torch.as_tensor(history[None, :, :], dtype=torch.float32, device=device)
            control = torch.as_tensor(controls[t : t + 1], dtype=torch.float32, device=device)
            out = model(obs, current, control, context, dt)
            current = out["x_next"]
            preds.append(current.detach().cpu().numpy()[0])
            targets.append(states[t + 1])
            if step + 1 < rollout_steps:
                next_obs = np.concatenate(
                    [preds[-1], controls[t + 1]],
                    axis=-1,
                ).astype(np.float32)
                history = np.vstack([history[1:], next_obs])
    pred_arr = np.asarray(preds, dtype=np.float32)
    target_arr = np.asarray(targets, dtype=np.float32)
    rmse = float(np.sqrt(np.mean((pred_arr - target_arr) ** 2)))
    finite_fraction = float(np.isfinite(pred_arr).mean()) if pred_arr.size else 0.0
    passed = bool(pred_arr.size and finite_fraction == 1.0 and np.isfinite(rmse))
    metrics = {
        "torch_rollout_smoke_passed": int(passed),
        "torch_rollout_steps": int(rollout_steps),
        "torch_rollout_rmse": rmse,
        "torch_rollout_finite_fraction": finite_fraction,
        "torch_device": str(device),
    }
    return passed, metrics


def _checkpoint_smoke(torch: Any, dataset_dir: str, torch_cfg: Dict[str, Any], out_dir: str) -> Tuple[bool, Dict[str, Any]]:
    from student_model.torch_model import HybridStudentModel

    device = _device(torch, torch_cfg)
    loader = _build_loader(torch, dataset_dir, torch_cfg)
    batch = _batch_to_device(next(iter(loader)), device)
    config = _model_config(torch_cfg)
    model = HybridStudentModel(config).to(device)
    model.eval()
    with torch.no_grad():
        before = _forward(model, batch)["x_next"].detach().cpu()
    checkpoint_path = os.path.join(
        out_dir,
        "checkpoints",
        torch_cfg.get("checkpoint_name", "student_model_smoke.pt"),
    )
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(config),
            "format_version": "pytorch_smoke_v0",
        },
        checkpoint_path,
    )
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
    restored = HybridStudentModel(type(config)(**loaded["model_config"])).to(device)
    restored.load_state_dict(loaded["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        after = _forward(restored, batch)["x_next"].detach().cpu()
    max_abs_diff = float((before - after).abs().max())
    passed = bool(os.path.exists(checkpoint_path) and max_abs_diff <= 1.0e-6)
    metrics = {
        "torch_checkpoint_smoke_passed": int(passed),
        "torch_checkpoint_exists": int(os.path.exists(checkpoint_path)),
        "torch_checkpoint_save_load_max_abs_diff": max_abs_diff,
        "torch_device": str(device),
    }
    return passed, metrics


def run_torch_training_suite(
    dataset_dir: str,
    torch_cfg: Dict[str, Any],
    out_dir: str,
) -> Dict[str, Any]:
    report = {
        "name": torch_cfg.get("name", torch_cfg.get("mode", "torch_training")),
        "mode": torch_cfg.get("mode", "data_loader_smoke"),
        "passed": False,
        "blocked": False,
        "blocked_reason": "",
        "metrics": {},
        "warnings": [],
        "errors": [],
    }
    torch, import_error = _torch_import_status()
    if import_error:
        report["blocked"] = True
        report["blocked_reason"] = import_error
        report["errors"].append(import_error)
        _write_json(os.path.join(out_dir, "artifacts", "torch_training_report.json"), report)
        return report
    if torch_cfg.get("require_cuda") and not torch.cuda.is_available():
        report["blocked"] = True
        report["blocked_reason"] = "CUDA is required for this run but torch.cuda.is_available() is False"
        report["errors"].append(report["blocked_reason"])
        _write_json(os.path.join(out_dir, "artifacts", "torch_training_report.json"), report)
        return report

    _set_seed(torch, int(torch_cfg.get("seed", 23)))
    try:
        mode = report["mode"]
        if mode == "data_loader_smoke":
            passed, metrics = _data_loader_smoke(torch, dataset_dir, torch_cfg)
        elif mode == "forward_loss_smoke":
            passed, metrics = _forward_loss_smoke(torch, dataset_dir, torch_cfg)
        elif mode == "tiny_overfit":
            passed, metrics = _tiny_overfit(torch, dataset_dir, torch_cfg)
        elif mode == "rollout_smoke":
            passed, metrics = _rollout_smoke(torch, dataset_dir, torch_cfg)
        elif mode == "checkpoint_smoke":
            passed, metrics = _checkpoint_smoke(torch, dataset_dir, torch_cfg, out_dir)
        else:
            raise ValueError("unknown torch training mode: %s" % mode)
        report["metrics"].update(metrics)
        report["passed"] = bool(passed)
        if not passed:
            report["errors"].append("%s failed success gate" % mode)
    except Exception as exc:  # pragma: no cover - covered by runner smoke at integration level.
        report["errors"].append(repr(exc))
        report["passed"] = False

    _write_json(os.path.join(out_dir, "artifacts", "torch_training_report.json"), report)
    return report
