import importlib.util
import json
import os
import random
from dataclasses import asdict, fields
from typing import Any, Dict, Tuple

import numpy as np

from student_model.constants import STATE_KEYS
from student_model.data import (
    TorchEpisodeDataset,
    TorchTransitionDataset,
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


def _build_direct_model(torch: Any, torch_cfg: Dict[str, Any], device: Any) -> Any:
    from torch import nn

    from student_model.constants import CONTROL_KEYS, STATE_KEYS
    from student_model.torch_model import MLPHead, make_encoder, sanitize_state

    class DirectTCNPredictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden_dim = int(torch_cfg.get("hidden_dim", 64))
            state_dim = len(STATE_KEYS)
            control_dim = len(CONTROL_KEYS)
            context_dim = int(torch_cfg.get("context_dim", 17))
            residual_bound = float(torch_cfg.get("direct_residual_bound", 0.6))
            obs_dim = state_dim + control_dim
            feature_dim = hidden_dim + state_dim + control_dim + context_dim
            self.state_dim = state_dim
            self.context_dim = context_dim
            self.residual_bound = residual_bound
            self.encoder = make_encoder(
                torch_cfg.get("encoder_type", "tcn"),
                obs_dim,
                hidden_dim,
                int(torch_cfg.get("history_len", 8)),
            )
            self.head = MLPHead(feature_dim, hidden_dim, state_dim)
            self.logvar_head = MLPHead(feature_dim, hidden_dim, state_dim)

        def forward(
            self,
            observable_history: Any,
            current_state: Any = None,
            current_control: Any = None,
            context: Any = None,
            dt: Any = None,
        ) -> Dict[str, Any]:
            if current_state is None:
                current_state = observable_history[:, -1, : self.state_dim]
            if current_control is None:
                current_control = observable_history[:, -1, self.state_dim :]
            if context is None:
                context = observable_history.new_zeros(
                    observable_history.shape[0],
                    self.context_dim,
                )
            z = self.encoder(observable_history)
            features = torch.cat([z, current_state, current_control, context], dim=-1)
            delta_x = torch.tanh(self.head(features)) * self.residual_bound
            x_next = sanitize_state(current_state + delta_x)
            logvar = torch.clamp(self.logvar_head(features), min=-8.0, max=5.0)
            return {
                "x_next": x_next,
                "delta_x": delta_x,
                "logvar": logvar,
                "z_shared": z,
            }

    class DirectNBeatsPredictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden_dim = int(torch_cfg.get("hidden_dim", 64))
            state_dim = len(STATE_KEYS)
            control_dim = len(CONTROL_KEYS)
            context_dim = int(torch_cfg.get("context_dim", 17))
            history_len = int(torch_cfg.get("history_len", 8))
            residual_bound = float(torch_cfg.get("direct_residual_bound", 0.6))
            input_dim = history_len * (state_dim + control_dim) + state_dim + control_dim + context_dim
            self.state_dim = state_dim
            self.context_dim = context_dim
            self.history_len = history_len
            self.residual_bound = residual_bound
            self.backcast = MLPHead(input_dim, hidden_dim, input_dim)
            self.forecast = MLPHead(input_dim, hidden_dim, state_dim)
            self.refine = MLPHead(input_dim, hidden_dim, state_dim)
            self.logvar_head = MLPHead(input_dim, hidden_dim, state_dim)

        def forward(
            self,
            observable_history: Any,
            current_state: Any = None,
            current_control: Any = None,
            context: Any = None,
            dt: Any = None,
        ) -> Dict[str, Any]:
            if current_state is None:
                current_state = observable_history[:, -1, : self.state_dim]
            if current_control is None:
                current_control = observable_history[:, -1, self.state_dim :]
            if context is None:
                context = observable_history.new_zeros(
                    observable_history.shape[0],
                    self.context_dim,
                )
            flat = observable_history.reshape(observable_history.shape[0], -1)
            features = torch.cat([flat, current_state, current_control, context], dim=-1)
            residual = features - self.backcast(features)
            delta_x = torch.tanh(self.forecast(features) + self.refine(residual)) * self.residual_bound
            x_next = sanitize_state(current_state + delta_x)
            logvar = torch.clamp(self.logvar_head(features), min=-8.0, max=5.0)
            return {
                "x_next": x_next,
                "delta_x": delta_x,
                "logvar": logvar,
                "z_shared": residual[:, : min(residual.shape[-1], 64)],
            }

    if torch_cfg.get("direct_arch", torch_cfg.get("encoder_type", "tcn")) == "nbeats":
        return DirectNBeatsPredictor().to(device)
    return DirectTCNPredictor().to(device)


def _build_model_for_cfg(torch: Any, torch_cfg: Dict[str, Any], device: Any) -> Any:
    model_type = torch_cfg.get("model_type", "hybrid")
    if str(model_type).startswith("direct_"):
        direct_cfg = dict(torch_cfg.get("baseline_config", {}))
        if model_type == "direct_gru":
            direct_cfg.setdefault("encoder_type", "gru")
        elif model_type == "direct_transformer":
            direct_cfg.setdefault("encoder_type", "transformer")
        elif model_type == "direct_nbeats":
            direct_cfg.setdefault("direct_arch", "nbeats")
        else:
            direct_cfg.setdefault("encoder_type", "tcn")
        direct_cfg.setdefault("history_len", int(torch_cfg.get("history_len", 8)))
        return _build_direct_model(torch, direct_cfg, device)
    return _build_model(torch_cfg, device)


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


def _build_transition_loader(
    torch: Any,
    dataset_dir: str,
    torch_cfg: Dict[str, Any],
    split_role: str,
    shuffle: bool,
) -> Any:
    from torch.utils.data import DataLoader

    dataset = TorchTransitionDataset(
        dataset_dir,
        history_len=int(torch_cfg.get("history_len", 8)),
        split_role=split_role,
        stride=int(torch_cfg.get("sample_stride", 1)),
        max_episodes=int(torch_cfg.get("max_episodes", 0) or 0),
        max_samples=int(torch_cfg.get("max_samples", 0) or 0),
        fine_tune_buckets=torch_cfg.get("fine_tune_buckets"),
        target_window_role=torch_cfg.get("target_window_role"),
        scenario_groups=torch_cfg.get("scenario_groups"),
        vehicle_config_ids=torch_cfg.get("vehicle_config_ids"),
    )
    return DataLoader(
        dataset,
        batch_size=int(torch_cfg.get("batch_size", 8)),
        shuffle=shuffle,
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


def _compute_state_scale(torch: Any, loader: Any, device: Any) -> Any:
    chunks = []
    for batch in loader:
        chunks.append(batch["target_next_state"].to(device))
    if not chunks:
        return None
    values = torch.cat(chunks, dim=0)
    scale = values.std(dim=0).clamp_min(1.0e-3)
    return scale


def _evaluate_one_step(
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
    state_scale: Any = None,
    loss_mode: str = "mse",
    aux_weights: Dict[str, float] | None = None,
) -> Dict[str, float]:
    losses = []
    mses = []
    normalized_mses = []
    nlls = []
    channel_sqerr = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            out = _forward(model, batch)
            loss, metrics = _loss_metrics(
                torch,
                out,
                batch,
                state_scale=state_scale,
                loss_mode=loss_mode,
                aux_weights=aux_weights,
            )
            losses.append(float(loss.detach().cpu()))
            mses.append(metrics["one_step_mse"])
            normalized_mses.append(metrics["one_step_normalized_mse"])
            nlls.append(metrics["one_step_nll"])
            err = (out["x_next"] - batch["target_next_state"]).detach()
            channel_sqerr.append(err.pow(2).mean(dim=0).cpu().numpy())
    channel_rmse = np.sqrt(np.mean(np.stack(channel_sqerr), axis=0)) if channel_sqerr else []
    result = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "mse": float(np.mean(mses)) if mses else float("nan"),
        "normalized_mse": float(np.mean(normalized_mses)) if normalized_mses else float("nan"),
        "nll": float(np.mean(nlls)) if nlls else float("nan"),
    }
    for idx, value in enumerate(channel_rmse):
        result["rmse_%s" % STATE_KEYS[idx]] = float(value)
    return result


def _cycle_batches(loader: Any) -> Any:
    while True:
        for batch in loader:
            yield batch


def _checkpoint_payload(
    model: Any,
    optimizer: Any,
    torch_cfg: Dict[str, Any],
    step: int,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "format_version": "pytorch_training_v0",
        "model_type": torch_cfg.get("model_type", "hybrid"),
        "model_config": asdict(_model_config(torch_cfg))
        if torch_cfg.get("model_type", "hybrid") == "hybrid"
        else dict(torch_cfg.get("baseline_config", {})),
        "torch_training_config": dict(torch_cfg),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
        "metrics": metrics,
    }


def _save_training_checkpoint(
    torch: Any,
    path: str,
    model: Any,
    optimizer: Any,
    torch_cfg: Dict[str, Any],
    step: int,
    metrics: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(_checkpoint_payload(model, optimizer, torch_cfg, step, metrics), path)


def _load_training_checkpoint(torch: Any, path: str, device: Any) -> Dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=True)


def _restore_model_from_checkpoint(torch: Any, checkpoint: Dict[str, Any], device: Any) -> Any:
    model_type = checkpoint.get("model_type", "hybrid")
    if str(model_type).startswith("direct_"):
        direct_cfg = dict(checkpoint.get("model_config", {}))
        if model_type == "direct_nbeats":
            direct_cfg.setdefault("direct_arch", "nbeats")
        elif model_type == "direct_gru":
            direct_cfg.setdefault("encoder_type", "gru")
        elif model_type == "direct_transformer":
            direct_cfg.setdefault("encoder_type", "transformer")
        model = _build_direct_model(
            torch,
            direct_cfg,
            device,
        )
    else:
        from student_model.torch_model import HybridStudentConfig, HybridStudentModel

        model = HybridStudentModel(HybridStudentConfig(**checkpoint.get("model_config", {}))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    return model


def _one_step_train(torch: Any, dataset_dir: str, torch_cfg: Dict[str, Any], out_dir: str) -> Tuple[bool, Dict[str, Any]]:
    device = _device(torch, torch_cfg)
    train_loader = _build_transition_loader(
        torch,
        dataset_dir,
        torch_cfg,
        split_role=torch_cfg.get("split_role", "train"),
        shuffle=True,
    )
    val_cfg = {
        **torch_cfg,
        "max_samples": int(torch_cfg.get("max_val_samples", torch_cfg.get("max_samples", 0) or 0)),
        "target_window_role": torch_cfg.get("val_target_window_role"),
        "fine_tune_buckets": torch_cfg.get("val_fine_tune_buckets"),
    }
    val_loader = _build_transition_loader(
        torch,
        dataset_dir,
        val_cfg,
        split_role=torch_cfg.get("val_split_role", "validation"),
        shuffle=False,
    )
    train_eval_cfg = {
        **torch_cfg,
        "max_samples": int(torch_cfg.get("max_train_eval_samples", torch_cfg.get("max_samples", 0) or 0)),
    }
    train_eval_loader = _build_transition_loader(
        torch,
        dataset_dir,
        train_eval_cfg,
        split_role=torch_cfg.get("split_role", "train"),
        shuffle=False,
    )
    loss_mode = torch_cfg.get("loss_mode", "mse")
    state_scale = None
    if loss_mode == "normalized_mse":
        state_scale = _compute_state_scale(torch, train_eval_loader, device)
    aux_weights = {
        "fz": float(torch_cfg.get("fz_aux_weight", 0.0)),
        "tire": float(torch_cfg.get("tire_aux_weight", 0.0)),
        "mu": float(torch_cfg.get("mu_aux_weight", 0.0)),
        "steering": float(torch_cfg.get("steering_aux_weight", 0.0)),
    }
    model = _build_model_for_cfg(torch, torch_cfg, device)
    resume_from = torch_cfg.get("resume_from")
    start_step = 0
    if resume_from:
        checkpoint = _load_training_checkpoint(torch, resume_from, device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        start_step = int(checkpoint.get("step", 0))
    fine_tune_mode = torch_cfg.get("fine_tune_mode")
    if fine_tune_mode and hasattr(model, "set_trainability"):
        model.set_trainability(fine_tune_mode)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = None
    if trainable_params:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(torch_cfg.get("learning_rate", 1.0e-4)),
            weight_decay=float(torch_cfg.get("weight_decay", 0.0)),
        )
        if resume_from and not fine_tune_mode:
            checkpoint = _load_training_checkpoint(torch, resume_from, device)
            if checkpoint.get("optimizer_state_dict"):
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    max_steps = int(torch_cfg.get("max_steps", 100))
    eval_interval = max(1, int(torch_cfg.get("eval_interval", max_steps)))
    grad_clip = float(torch_cfg.get("grad_clip", 1.0))
    history_path = os.path.join(out_dir, "artifacts", "training_history.jsonl")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    iterator = _cycle_batches(train_loader)
    initial_train_eval = _evaluate_one_step(
        torch,
        model,
        train_eval_loader,
        device,
        state_scale=state_scale,
        loss_mode=loss_mode,
        aux_weights=aux_weights,
    )
    final_loss = None
    last_val = {"loss": float("nan"), "mse": float("nan"), "normalized_mse": float("nan"), "nll": float("nan")}
    for local_step in range(1, max_steps + 1):
        if optimizer is None:
            break
        model.train()
        batch = _batch_to_device(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        out = _forward(model, batch)
        loss, train_metrics = _loss_metrics(
            torch,
            out,
            batch,
            state_scale=state_scale,
            loss_mode=loss_mode,
            aux_weights=aux_weights,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        global_step = start_step + local_step
        row = {
            "step": global_step,
            "train_loss": final_loss,
            "train_mse": train_metrics["one_step_mse"],
            "grad_norm": float(grad_norm.detach().cpu() if hasattr(grad_norm, "detach") else grad_norm),
        }
        if local_step == max_steps or local_step % eval_interval == 0:
            last_val = _evaluate_one_step(
                torch,
                model,
                val_loader,
                device,
                state_scale=state_scale,
                loss_mode=loss_mode,
                aux_weights=aux_weights,
            )
            row.update(
                {
                    "val_loss": last_val["loss"],
                    "val_mse": last_val["mse"],
                    "val_normalized_mse": last_val["normalized_mse"],
                    "val_nll": last_val["nll"],
                }
            )
        with open(history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    if final_loss is None:
        final_eval = _evaluate_one_step(
            torch,
            model,
            train_eval_loader,
            device,
            state_scale=state_scale,
            loss_mode=loss_mode,
            aux_weights=aux_weights,
        )
        final_loss = final_eval["loss"]
        last_val = _evaluate_one_step(
            torch,
            model,
            val_loader,
            device,
            state_scale=state_scale,
            loss_mode=loss_mode,
            aux_weights=aux_weights,
        )
    final_train_eval = _evaluate_one_step(
        torch,
        model,
        train_eval_loader,
        device,
        state_scale=state_scale,
        loss_mode=loss_mode,
        aux_weights=aux_weights,
    )
    train_loss_ratio = final_train_eval["loss"] / max(initial_train_eval["loss"], 1.0e-12)
    checkpoint_name = torch_cfg.get("checkpoint_name", "model.pt")
    checkpoint_path = os.path.join(out_dir, "checkpoints", checkpoint_name)
    metrics = {
        "torch_one_step_training_initial_loss": float(initial_train_eval["loss"]),
        "torch_one_step_training_final_loss": float(final_train_eval["loss"]),
        "torch_one_step_training_last_batch_loss": float(final_loss),
        "torch_one_step_training_loss_ratio": float(train_loss_ratio),
        "torch_one_step_training_val_mse": float(last_val["mse"]),
        "torch_one_step_training_val_normalized_mse": float(last_val["normalized_mse"]),
        "torch_one_step_training_steps": max_steps,
        "torch_one_step_training_start_step": start_step,
        "torch_one_step_training_end_step": start_step + max_steps,
        "torch_one_step_training_sample_count": len(train_loader.dataset),
        "torch_one_step_validation_sample_count": len(val_loader.dataset),
        "torch_one_step_training_trainable_parameter_count": int(
            sum(param.numel() for param in trainable_params)
        ),
        "torch_device": str(device),
    }
    for key in STATE_KEYS:
        if "rmse_%s" % key in last_val:
            metrics["torch_one_step_training_val_rmse_%s" % key] = float(last_val["rmse_%s" % key])
    if device.type == "cuda":
        metrics["torch_cuda_max_memory_allocated_mb"] = float(
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        )
    _save_training_checkpoint(
        torch,
        checkpoint_path,
        model,
        optimizer,
        torch_cfg,
        start_step + max_steps,
        metrics,
    )
    metrics["torch_one_step_training_checkpoint_exists"] = int(os.path.exists(checkpoint_path))
    metric_name = (
        "torch_black_box_training_passed"
        if torch_cfg.get("model_type") == "direct_tcn"
        else "torch_one_step_training_passed"
    )
    passed = (
        np.isfinite(final_loss)
        and np.isfinite(last_val["mse"])
        and train_loss_ratio <= float(torch_cfg.get("success_loss_ratio", 1.0))
        and os.path.exists(checkpoint_path)
    )
    metrics[metric_name] = int(bool(passed))
    return bool(passed), metrics


def _loss_metrics(
    torch: Any,
    out: Dict[str, Any],
    batch: Dict[str, Any],
    state_scale: Any = None,
    loss_mode: str = "mse",
    aux_weights: Dict[str, float] | None = None,
) -> Tuple[Any, Dict[str, float]]:
    err = out["x_next"] - batch["target_next_state"]
    mse = err.pow(2).mean()
    if state_scale is None:
        normalized_mse = mse
    else:
        normalized_mse = (err / state_scale[None, :]).pow(2).mean()
    logvar = out["logvar"].clamp(-8.0, 5.0)
    nll = 0.5 * (err.pow(2) * torch.exp(-logvar) + logvar).mean()
    residual_l2 = out["delta_x"].pow(2).mean()
    base_loss = normalized_mse if loss_mode == "normalized_mse" else mse
    aux_weights = aux_weights or {}
    aux_loss = mse.new_tensor(0.0)
    aux_metrics: Dict[str, float] = {}
    if aux_weights.get("fz", 0.0) and "fz_true" in batch and "fz" in out:
        value = (out["fz"] - batch["fz_true"]).pow(2).mean()
        aux_loss = aux_loss + float(aux_weights["fz"]) * value
        aux_metrics["fz_aux_mse"] = float(value.detach().cpu())
    if aux_weights.get("tire", 0.0) and "tire_force_true" in batch and "tire_forces" in out:
        scale = batch["tire_force_true"].abs().mean().clamp_min(1.0)
        value = ((out["tire_forces"] - batch["tire_force_true"]) / scale).pow(2).mean()
        aux_loss = aux_loss + float(aux_weights["tire"]) * value
        aux_metrics["tire_aux_normalized_mse"] = float(value.detach().cpu())
    if aux_weights.get("mu", 0.0) and "mu_true" in batch and "mu" in out:
        value = (out["mu"] - batch["mu_true"]).pow(2).mean()
        aux_loss = aux_loss + float(aux_weights["mu"]) * value
        aux_metrics["mu_aux_mse"] = float(value.detach().cpu())
    if aux_weights.get("steering", 0.0) and "steering_true" in batch and "steering_delta" in out:
        steering_pred = out["steering_delta"].expand(-1, batch["steering_true"].shape[-1])
        value = (steering_pred - batch["steering_true"]).pow(2).mean()
        aux_loss = aux_loss + float(aux_weights["steering"]) * value
        aux_metrics["steering_aux_mse"] = float(value.detach().cpu())
    loss = base_loss + 1.0e-4 * residual_l2 + aux_loss
    metrics = {
        "one_step_mse": float(mse.detach().cpu()),
        "one_step_normalized_mse": float(normalized_mse.detach().cpu()),
        "one_step_nll": float(nll.detach().cpu()),
        "residual_l2": float(residual_l2.detach().cpu()),
    }
    metrics.update(aux_metrics)
    return loss, metrics


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


def _constraint_violation_rate(pred_arr: np.ndarray) -> float:
    if pred_arr.size == 0:
        return 1.0
    violations = np.zeros(pred_arr.shape[0], dtype=bool)
    for idx, key in enumerate(STATE_KEYS):
        value = pred_arr[:, idx]
        if key == "vx":
            violations |= value < 0.03
        elif key.startswith("omega_"):
            violations |= value < 0.0
        elif key in {"roll", "pitch"}:
            violations |= np.abs(value) > 2.0
        elif key in {"p", "q", "r"}:
            violations |= np.abs(value) > 20.0
    return float(np.mean(violations))


def _rollout_eval_from_checkpoint(
    torch: Any,
    dataset_dir: str,
    torch_cfg: Dict[str, Any],
    out_dir: str,
) -> Tuple[bool, Dict[str, Any]]:
    device = _device(torch, torch_cfg)
    checkpoint = _load_training_checkpoint(torch, torch_cfg["checkpoint_path"], device)
    model = _restore_model_from_checkpoint(torch, checkpoint, device)
    model.eval()
    manifest = validate_canonical_dataset(dataset_dir)
    history_len = int(torch_cfg.get("history_len", 8))
    rollout_steps = int(torch_cfg.get("rollout_steps", 16))
    max_episodes = int(torch_cfg.get("max_episodes", 4))
    split_role = torch_cfg.get("split_role", "validation")
    from student_model.data import load_manifest

    items = [
        (idx, item)
        for idx, item in enumerate(load_manifest(dataset_dir).get("episodes", []))
        if item.get("split_role") == split_role
    ]
    if not items:
        items = [(idx, item) for idx, item in enumerate(load_manifest(dataset_dir).get("episodes", []))]
    rows = []
    all_sqerr = []
    all_channel_sqerr = []
    finite_values = []
    violation_values = []
    with torch.no_grad():
        for global_index, item in items[:max_episodes]:
            record = load_episode_record(dataset_dir, global_index)
            states, controls = episode_arrays(record)
            steps = min(rollout_steps, len(states) - history_len - 1)
            if steps <= 0:
                continue
            context = torch.as_tensor(context_vector(record)[None, :], dtype=torch.float32, device=device)
            dt = torch.tensor([float(record.metadata["dt"])], dtype=torch.float32, device=device)
            history = np.concatenate([states[:history_len], controls[:history_len]], axis=-1)
            current = torch.as_tensor(
                states[history_len - 1 : history_len],
                dtype=torch.float32,
                device=device,
            )
            preds = []
            targets = []
            for step in range(steps):
                t = history_len - 1 + step
                obs = torch.as_tensor(history[None, :, :], dtype=torch.float32, device=device)
                control = torch.as_tensor(controls[t : t + 1], dtype=torch.float32, device=device)
                out = model(obs, current, control, context, dt)
                current = out["x_next"]
                pred = current.detach().cpu().numpy()[0]
                preds.append(pred)
                targets.append(states[t + 1])
                if step + 1 < steps:
                    history = np.vstack(
                        [
                            history[1:],
                            np.concatenate([pred, controls[t + 1]], axis=-1).astype(np.float32),
                        ]
                    )
            pred_arr = np.asarray(preds, dtype=np.float32)
            target_arr = np.asarray(targets, dtype=np.float32)
            sqerr = (pred_arr - target_arr) ** 2
            all_sqerr.append(sqerr.reshape(-1))
            all_channel_sqerr.append(np.mean(sqerr, axis=0))
            finite_fraction = float(np.isfinite(pred_arr).mean()) if pred_arr.size else 0.0
            violation_rate = _constraint_violation_rate(pred_arr)
            finite_values.append(finite_fraction)
            violation_values.append(violation_rate)
            rows.append(
                {
                    "episode_id": item.get("episode_id", record.episode_id),
                    "split_role": split_role,
                    "steps": int(steps),
                    "rmse": float(np.sqrt(np.mean(sqerr))),
                    "finite_fraction": finite_fraction,
                    "constraint_violation_rate": violation_rate,
                }
            )
    report_path = os.path.join(out_dir, "artifacts", "rollout_eval.json")
    _write_json(
        report_path,
        {
            "checkpoint_path": torch_cfg["checkpoint_path"],
            "dataset_summary": manifest,
            "episodes": rows,
        },
    )
    rmse = float(np.sqrt(np.mean(np.concatenate(all_sqerr)))) if all_sqerr else float("nan")
    channel_rmse = (
        np.sqrt(np.mean(np.stack(all_channel_sqerr), axis=0))
        if all_channel_sqerr
        else []
    )
    finite_fraction = float(np.mean(finite_values)) if finite_values else 0.0
    violation_rate = float(np.mean(violation_values)) if violation_values else 1.0
    passed = (
        bool(rows)
        and np.isfinite(rmse)
        and finite_fraction == 1.0
        and violation_rate <= float(torch_cfg.get("max_constraint_violation_rate", 0.0))
        and rmse <= float(torch_cfg.get("max_rollout_rmse", 1.0e6))
    )
    metrics = {
        "torch_rollout_eval_passed": int(passed),
        "torch_rollout_eval_rmse": rmse,
        "torch_rollout_eval_episode_count": len(rows),
        "torch_rollout_eval_steps": rollout_steps,
        "torch_rollout_eval_finite_fraction": finite_fraction,
        "torch_rollout_eval_constraint_violation_rate": violation_rate,
        "torch_device": str(device),
    }
    for idx, value in enumerate(channel_rmse):
        metrics["torch_rollout_eval_rmse_%s" % STATE_KEYS[idx]] = float(value)
    return bool(passed), metrics


def _resume_eval_smoke(
    torch: Any,
    dataset_dir: str,
    torch_cfg: Dict[str, Any],
    out_dir: str,
) -> Tuple[bool, Dict[str, Any]]:
    device = _device(torch, torch_cfg)
    checkpoint = _load_training_checkpoint(torch, torch_cfg["checkpoint_path"], device)
    model = _restore_model_from_checkpoint(torch, checkpoint, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(torch_cfg.get("learning_rate", 1.0e-4)),
        weight_decay=float(torch_cfg.get("weight_decay", 0.0)),
    )
    if checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    val_cfg = {**torch_cfg, "max_samples": int(torch_cfg.get("max_val_samples", 64))}
    val_loader = _build_transition_loader(
        torch,
        dataset_dir,
        val_cfg,
        split_role=torch_cfg.get("val_split_role", "validation"),
        shuffle=False,
    )
    before = _evaluate_one_step(torch, model, val_loader, device)
    train_cfg = {
        **torch_cfg,
        "max_steps": int(torch_cfg.get("resume_steps", 10)),
        "checkpoint_name": torch_cfg.get("checkpoint_name", "resume_model.pt"),
        "resume_from": torch_cfg["checkpoint_path"],
    }
    passed_train, train_metrics = _one_step_train(torch, dataset_dir, train_cfg, out_dir)
    resumed_path = os.path.join(out_dir, "checkpoints", train_cfg["checkpoint_name"])
    resumed = _load_training_checkpoint(torch, resumed_path, device)
    restored = _restore_model_from_checkpoint(torch, resumed, device)
    after = _evaluate_one_step(torch, restored, val_loader, device)
    start_step = int(checkpoint.get("step", 0))
    end_step = int(resumed.get("step", 0))
    passed = (
        passed_train
        and os.path.exists(resumed_path)
        and end_step > start_step
        and np.isfinite(before["mse"])
        and np.isfinite(after["mse"])
    )
    metrics = {
        "torch_resume_eval_passed": int(bool(passed)),
        "torch_resume_start_step": start_step,
        "torch_resume_end_step": end_step,
        "torch_resume_val_mse_before": float(before["mse"]),
        "torch_resume_val_mse_after": float(after["mse"]),
        "torch_resume_checkpoint_exists": int(os.path.exists(resumed_path)),
        "torch_device": str(device),
    }
    metrics.update(
        {
            "torch_resume_train_loss_ratio": train_metrics.get(
                "torch_one_step_training_loss_ratio",
                float("nan"),
            )
        }
    )
    return bool(passed), metrics


def _prefixed_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {"%s_%s" % (prefix, key): value for key, value in metrics.items()}


def _fair_compare(
    torch: Any,
    dataset_dir: str,
    torch_cfg: Dict[str, Any],
    out_dir: str,
) -> Tuple[bool, Dict[str, Any]]:
    variants = torch_cfg.get("variants") or [
        {"name": "hybrid", "model_type": "hybrid"},
        {"name": "direct_tcn", "model_type": "direct_tcn"},
    ]
    rows = []
    metrics: Dict[str, Any] = {"torch_fair_compare_variant_count": len(variants)}
    shared_keys = [
        "history_len",
        "batch_size",
        "max_episodes",
        "max_samples",
        "max_val_samples",
        "sample_stride",
        "max_steps",
        "eval_interval",
        "learning_rate",
        "weight_decay",
        "grad_clip",
        "success_loss_ratio",
        "loss_mode",
        "device",
        "require_cuda",
        "split_role",
        "val_split_role",
    ]
    all_passed = True
    best_black_box_mse = float("inf")
    hybrid_mse = float("nan")
    best_black_box_rollout = float("inf")
    hybrid_rollout = float("nan")
    for variant in variants:
        name = variant["name"]
        variant_dir = os.path.join(out_dir, "artifacts", "fair_compare", name)
        variant_cfg = {key: torch_cfg[key] for key in shared_keys if key in torch_cfg}
        variant_cfg.update(variant.get("torch_training", {}))
        variant_cfg["model_type"] = variant.get("model_type", "hybrid")
        variant_cfg["name"] = "fair_compare_%s" % name
        variant_cfg["checkpoint_name"] = "%s.pt" % name
        if "model_config" in variant:
            variant_cfg["model_config"] = variant["model_config"]
        if "baseline_config" in variant:
            variant_cfg["baseline_config"] = variant["baseline_config"]
        train_passed, train_metrics = _one_step_train(torch, dataset_dir, variant_cfg, variant_dir)
        checkpoint_path = os.path.join(variant_dir, "checkpoints", variant_cfg["checkpoint_name"])
        rollout_cfg = {
            **variant_cfg,
            "checkpoint_path": checkpoint_path,
            "rollout_steps": int(torch_cfg.get("rollout_steps", 16)),
            "max_episodes": int(torch_cfg.get("rollout_max_episodes", 4)),
            "max_rollout_rmse": float(torch_cfg.get("max_rollout_rmse", 1.0e6)),
            "max_constraint_violation_rate": float(
                torch_cfg.get("max_constraint_violation_rate", 0.0)
            ),
            "split_role": torch_cfg.get("rollout_split_role", torch_cfg.get("val_split_role", "validation")),
        }
        rollout_passed, rollout_metrics = _rollout_eval_from_checkpoint(
            torch,
            dataset_dir,
            rollout_cfg,
            variant_dir,
        )
        val_mse = float(train_metrics.get("torch_one_step_training_val_mse", float("nan")))
        rollout_rmse = float(rollout_metrics.get("torch_rollout_eval_rmse", float("nan")))
        row = {
            "name": name,
            "model_type": variant_cfg["model_type"],
            "train_passed": bool(train_passed),
            "rollout_passed": bool(rollout_passed),
            "val_mse": val_mse,
            "val_normalized_mse": float(
                train_metrics.get("torch_one_step_training_val_normalized_mse", float("nan"))
            ),
            "rollout_rmse": rollout_rmse,
            "checkpoint_path": checkpoint_path,
        }
        rows.append(row)
        metrics.update(_prefixed_metrics("torch_fair_%s_train" % name, train_metrics))
        metrics.update(_prefixed_metrics("torch_fair_%s_rollout" % name, rollout_metrics))
        all_passed = all_passed and bool(train_passed) and bool(rollout_passed)
        if variant_cfg["model_type"] == "hybrid":
            hybrid_mse = val_mse
            hybrid_rollout = rollout_rmse
        elif np.isfinite(val_mse):
            best_black_box_mse = min(best_black_box_mse, val_mse)
            if np.isfinite(rollout_rmse):
                best_black_box_rollout = min(best_black_box_rollout, rollout_rmse)
    if np.isfinite(hybrid_mse) and np.isfinite(best_black_box_mse):
        metrics["torch_fair_compare_hybrid_vs_best_black_box_mse_ratio"] = (
            hybrid_mse / max(best_black_box_mse, 1.0e-12)
        )
    if np.isfinite(hybrid_rollout) and np.isfinite(best_black_box_rollout):
        metrics["torch_fair_compare_hybrid_vs_best_black_box_rollout_ratio"] = (
            hybrid_rollout / max(best_black_box_rollout, 1.0e-12)
        )
    _write_json(
        os.path.join(out_dir, "artifacts", "fair_compare_report.json"),
        {"variants": rows, "metrics": metrics},
    )
    passed = all_passed and len(rows) >= 2
    metrics["torch_fair_compare_passed"] = int(bool(passed))
    return bool(passed), metrics


def _model_variant_smoke(
    torch: Any,
    dataset_dir: str,
    torch_cfg: Dict[str, Any],
    out_dir: str,
) -> Tuple[bool, Dict[str, Any]]:
    device = _device(torch, torch_cfg)
    loader = _build_transition_loader(
        torch,
        dataset_dir,
        {**torch_cfg, "max_samples": int(torch_cfg.get("max_samples", 8))},
        split_role=torch_cfg.get("split_role", "train"),
        shuffle=False,
    )
    batch = _batch_to_device(next(iter(loader)), device)
    variants = torch_cfg.get("variants") or []
    rows = []
    passed = True
    for variant in variants:
        cfg = {
            **torch_cfg,
            "model_type": variant.get("model_type", "hybrid"),
            "model_config": variant.get("model_config", torch_cfg.get("model_config", {})),
            "baseline_config": variant.get("baseline_config", torch_cfg.get("baseline_config", {})),
        }
        model = _build_model_for_cfg(torch, cfg, device)
        model.eval()
        with torch.no_grad():
            out = _forward(model, batch)
        finite = bool(torch.isfinite(out["x_next"]).all())
        shape_ok = tuple(out["x_next"].shape) == tuple(batch["target_next_state"].shape)
        rows.append(
            {
                "name": variant.get("name", variant.get("model_type", "hybrid")),
                "model_type": cfg["model_type"],
                "finite": finite,
                "shape_ok": shape_ok,
                "output_keys": sorted(out.keys()),
                "parameter_count": int(sum(param.numel() for param in model.parameters())),
            }
        )
        passed = passed and finite and shape_ok
    _write_json(os.path.join(out_dir, "artifacts", "model_variant_smoke.json"), {"variants": rows})
    metrics = {
        "torch_model_variant_smoke_passed": int(bool(passed and rows)),
        "torch_model_variant_count": len(rows),
        "torch_device": str(device),
    }
    return bool(metrics["torch_model_variant_smoke_passed"]), metrics


def _fine_tune_smoke(
    torch: Any,
    dataset_dir: str,
    torch_cfg: Dict[str, Any],
    out_dir: str,
) -> Tuple[bool, Dict[str, Any]]:
    modes = torch_cfg.get("fine_tune_modes") or ["FT0", "FT1", "FT2", "FT3", "FT4", "FT5", "FT6"]
    buckets = torch_cfg.get("fine_tune_buckets_grid") or [["FTD1"], ["FTD1", "FTD2"]]
    rows = []
    metrics: Dict[str, Any] = {
        "torch_fine_tune_mode_count": len(modes),
        "torch_fine_tune_bucket_cell_count": len(buckets),
    }
    all_passed = True
    for mode in modes:
        for bucket_index, bucket_values in enumerate(buckets):
            cell_name = "%s_B%d" % (mode, bucket_index)
            cell_dir = os.path.join(out_dir, "artifacts", "fine_tune", cell_name)
            max_steps = int(torch_cfg.get("max_steps", 8))
            cell_cfg = {
                **torch_cfg,
                "mode": "one_step_train",
                "name": "fine_tune_%s" % cell_name,
                "fine_tune_mode": mode,
                "fine_tune_buckets": bucket_values,
                "target_window_role": torch_cfg.get("target_window_role", "target_train"),
                "val_target_window_role": torch_cfg.get("val_target_window_role", "target_test"),
                "split_role": torch_cfg.get("split_role", "fine-tune"),
                "val_split_role": torch_cfg.get("val_split_role", "test-window"),
                "max_steps": 0 if mode == "FT0" else max_steps,
                "success_loss_ratio": float(torch_cfg.get("success_loss_ratio", 1.25)),
                "checkpoint_name": "%s.pt" % cell_name,
            }
            train_passed, train_metrics = _one_step_train(torch, dataset_dir, cell_cfg, cell_dir)
            trainable_count = int(train_metrics.get("torch_one_step_training_trainable_parameter_count", 0))
            expected_trainable = 0 if mode == "FT0" else 1
            trainability_ok = (trainable_count == 0) if mode == "FT0" else (trainable_count > 0)
            row = {
                "mode": mode,
                "buckets": bucket_values,
                "passed": bool(train_passed),
                "trainability_ok": trainability_ok,
                "trainable_parameter_count": trainable_count,
                "val_mse": float(train_metrics.get("torch_one_step_training_val_mse", float("nan"))),
                "checkpoint": os.path.join(cell_dir, "checkpoints", cell_cfg["checkpoint_name"]),
            }
            rows.append(row)
            metrics.update(_prefixed_metrics("torch_ft_%s_B%d" % (mode, bucket_index), train_metrics))
            all_passed = all_passed and bool(train_passed) and trainability_ok and trainable_count >= expected_trainable
    _write_json(os.path.join(out_dir, "artifacts", "fine_tune_smoke_report.json"), {"rows": rows})
    metrics["torch_fine_tune_smoke_passed"] = int(bool(all_passed and rows))
    return bool(metrics["torch_fine_tune_smoke_passed"]), metrics


def _evaluate_ensemble_one_step(
    torch: Any,
    checkpoints: list[str],
    dataset_dir: str,
    torch_cfg: Dict[str, Any],
    device: Any,
) -> Dict[str, float]:
    loader = _build_transition_loader(
        torch,
        dataset_dir,
        {**torch_cfg, "max_samples": int(torch_cfg.get("max_val_samples", 64))},
        split_role=torch_cfg.get("val_split_role", "validation"),
        shuffle=False,
    )
    models = []
    for path in checkpoints:
        checkpoint = _load_training_checkpoint(torch, path, device)
        model = _restore_model_from_checkpoint(torch, checkpoint, device)
        model.eval()
        models.append(model)
    sqerr = []
    variances = []
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            preds = torch.stack([_forward(model, batch)["x_next"] for model in models], dim=0)
            mean_pred = preds.mean(dim=0)
            variance = preds.var(dim=0, unbiased=False)
            sqerr.append((mean_pred - batch["target_next_state"]).pow(2).mean().detach().cpu())
            variances.append(variance.mean().detach().cpu())
    return {
        "ensemble_mse": float(torch.stack(sqerr).mean()) if sqerr else float("nan"),
        "ensemble_predictive_variance": float(torch.stack(variances).mean()) if variances else float("nan"),
        "ensemble_member_count": len(models),
    }


def _ensemble_train(
    torch: Any,
    dataset_dir: str,
    torch_cfg: Dict[str, Any],
    out_dir: str,
) -> Tuple[bool, Dict[str, Any]]:
    device = _device(torch, torch_cfg)
    ensemble_k = int(torch_cfg.get("ensemble_k", 3))
    base_seed = int(torch_cfg.get("seed", 23))
    checkpoints: list[str] = []
    rows = []
    all_passed = True
    metrics: Dict[str, Any] = {"torch_ensemble_member_target_count": ensemble_k}
    for member in range(ensemble_k):
        _set_seed(torch, base_seed + member)
        member_dir = os.path.join(out_dir, "artifacts", "ensemble", "member_%02d" % member)
        member_cfg = {
            **torch_cfg,
            "mode": "one_step_train",
            "name": "ensemble_member_%02d" % member,
            "checkpoint_name": "member_%02d.pt" % member,
        }
        train_passed, train_metrics = _one_step_train(torch, dataset_dir, member_cfg, member_dir)
        checkpoint_path = os.path.join(member_dir, "checkpoints", member_cfg["checkpoint_name"])
        checkpoints.append(checkpoint_path)
        rows.append(
            {
                "member": member,
                "passed": bool(train_passed),
                "checkpoint_path": checkpoint_path,
                "val_mse": float(train_metrics.get("torch_one_step_training_val_mse", float("nan"))),
            }
        )
        metrics.update(_prefixed_metrics("torch_ensemble_member_%02d" % member, train_metrics))
        all_passed = all_passed and bool(train_passed) and os.path.exists(checkpoint_path)
    ensemble_eval = _evaluate_ensemble_one_step(torch, checkpoints, dataset_dir, torch_cfg, device)
    metrics.update({"torch_%s" % key: value for key, value in ensemble_eval.items()})
    passed = (
        all_passed
        and ensemble_eval["ensemble_member_count"] == ensemble_k
        and np.isfinite(ensemble_eval["ensemble_mse"])
        and np.isfinite(ensemble_eval["ensemble_predictive_variance"])
    )
    metrics["torch_ensemble_training_passed"] = int(bool(passed))
    _write_json(
        os.path.join(out_dir, "artifacts", "ensemble_report.json"),
        {"members": rows, "metrics": metrics},
    )
    return bool(passed), metrics


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
        elif mode == "one_step_train":
            passed, metrics = _one_step_train(torch, dataset_dir, torch_cfg, out_dir)
        elif mode == "rollout_eval":
            passed, metrics = _rollout_eval_from_checkpoint(
                torch,
                dataset_dir,
                torch_cfg,
                out_dir,
            )
        elif mode == "resume_eval_smoke":
            passed, metrics = _resume_eval_smoke(torch, dataset_dir, torch_cfg, out_dir)
        elif mode == "fair_compare":
            passed, metrics = _fair_compare(torch, dataset_dir, torch_cfg, out_dir)
        elif mode == "model_variant_smoke":
            passed, metrics = _model_variant_smoke(torch, dataset_dir, torch_cfg, out_dir)
        elif mode == "fine_tune_smoke":
            passed, metrics = _fine_tune_smoke(torch, dataset_dir, torch_cfg, out_dir)
        elif mode == "ensemble_train":
            passed, metrics = _ensemble_train(torch, dataset_dir, torch_cfg, out_dir)
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
