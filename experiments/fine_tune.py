import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from experiments.baselines import _fit_ridge
from experiments.hybrid import (
    BASE_VARIANT,
    EVAL_STATE_KEYS,
    _base_feature_row,
    _base_training_rows,
    _constraint_violation_rate,
    _context_vector,
    _control_matrix,
    _fit_weighted_ridge,
    _nominal_step,
    _residual_dot_bounds,
    _sanitize_state,
    _state_matrix,
    _variant_bounds,
)
from experiments.sanity import CONTROL_KEYS, STATE_KEYS
from simulator.data_generator.generate import generate_dataset
from simulator.vehicle_model.export import load_dataset


BUCKETS = ["FTD1", "FTD2", "FTD3", "FTD4", "FTD5"]
FT_RUN_DIRS = {
    "FT0": "runs/R038_finetune_FT0",
    "FT1": "runs/R039_finetune_FT1_vehicle_param_adapter",
    "FT2": "runs/R040_finetune_FT2_mu_head",
    "FT3": "runs/R041_finetune_FT3_fz_residual",
    "FT4": "runs/R042_finetune_FT4_tire_residual",
    "FT5": "runs/R043_finetune_FT5_steering_residual",
    "FT6": "runs/R044_finetune_FT6_full_model",
}


def run_fine_tune_suite(
    dataset_dir: str,
    fine_tune_cfg: Dict[str, Any],
    out_dir: str,
) -> Dict[str, Any]:
    mode = fine_tune_cfg.get("mode", "single")
    if mode == "aggregate":
        return _aggregate_existing_runs(out_dir, fine_tune_cfg)
    target_episodes = load_dataset(dataset_dir)
    source_episodes = _load_source_episodes(out_dir, fine_tune_cfg)
    ft_variant = fine_tune_cfg.get("ft_variant", "FT0")
    final_variant = _final_variant(fine_tune_cfg)
    history_len = int(fine_tune_cfg.get("history_len", 4))
    ridge_lambda = float(fine_tune_cfg.get("ridge_lambda", 10000.0))
    adapter_lambda = float(fine_tune_cfg.get("adapter_ridge_lambda", 2500.0))
    horizon = int(fine_tune_cfg.get("horizon_steps", [100])[-1])
    bounds = _variant_bounds(_residual_dot_bounds(fine_tune_cfg), final_variant)
    base_model = _train_base_model(
        source_episodes,
        final_variant,
        history_len,
        ridge_lambda,
        bounds,
    )
    target_train = _target_episodes(target_episodes, "train")
    target_validation = _target_episodes(target_episodes, "validation")
    target_test = _target_episodes(target_episodes, "test")
    ft0_validation = _evaluate_adapter_group(
        target_validation,
        base_model,
        None,
        "FT0",
        final_variant,
        history_len,
        horizon,
        bounds,
    )
    ft0_test = _evaluate_adapter_group(
        target_test,
        base_model,
        None,
        "FT0",
        final_variant,
        history_len,
        horizon,
        bounds,
    )
    rows = []
    if ft_variant == "FT0":
        rows.append(
            _result_row(
                ft_variant,
                "FTD0",
                0,
                0,
                ft0_validation,
                ft0_test,
                ft0_test["rmse"],
            )
        )
    else:
        for bucket in BUCKETS:
            selected = _episodes_up_to_bucket(target_train, bucket)
            x_adapt, y_adapt = _adapter_training_rows(
                selected,
                base_model,
                ft_variant,
                final_variant,
                history_len,
                bounds,
            )
            adapter = None
            if len(x_adapt):
                adapter = _fit_ridge(x_adapt, y_adapt, adapter_lambda)
            validation = _evaluate_adapter_group(
                target_validation,
                base_model,
                adapter,
                ft_variant,
                final_variant,
                history_len,
                horizon,
                bounds,
            )
            test = _evaluate_adapter_group(
                target_test,
                base_model,
                adapter,
                ft_variant,
                final_variant,
                history_len,
                horizon,
                bounds,
            )
            rows.append(
                _result_row(
                    ft_variant,
                    bucket,
                    len(selected),
                    len(x_adapt),
                    validation,
                    test,
                    ft0_test["rmse"],
                )
            )

    best = min(rows, key=lambda row: row["test_rmse"]) if rows else {}
    report = {
        "dataset_dir": dataset_dir,
        "source_dataset_dir": _source_dataset_dir(out_dir),
        "system": "fine_tune_scaffold",
        "ft_variant": ft_variant,
        "final_variant": final_variant,
        "history_len": history_len,
        "horizon_steps": horizon,
        "target_episode_counts": {
            "train": len(target_train),
            "validation": len(target_validation),
            "test": len(target_test),
        },
        "ft0_reference": {
            "validation_rmse": ft0_validation["rmse"],
            "test_rmse": ft0_test["rmse"],
            "test_constraint_violation_rate": ft0_test["constraint_violation_rate"],
        },
        "rows": rows,
        "summary_metrics": _summary_metrics(ft_variant, rows, ft0_test["rmse"]),
        "passed": True,
        "best_row": best,
        "warnings": [
            "M9 uses a ridge adapter scaffold over the frozen scaffold-selected hybrid model.",
            "This validates FT matrix plumbing and target-window split handling before PyTorch fine-tuning.",
        ],
    }
    report["passed"] = bool(report["summary_metrics"].get("fine_tune_run_passed", 0))
    _write_json(os.path.join(out_dir, "artifacts", "fine_tune_report.json"), report)
    return report


def fine_tune_metrics_for_runner(report: Dict[str, Any]) -> Dict[str, Any]:
    metrics = dict(report.get("summary_metrics", {}))
    metrics["fine_tune_suite_passed"] = int(bool(report.get("passed")))
    return metrics


def _load_source_episodes(out_dir: str, fine_tune_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    source_cfg = fine_tune_cfg.get("source_teacher_config")
    if not source_cfg:
        return []
    source_dir = _source_dataset_dir(out_dir)
    generate_dataset(source_cfg, source_dir)
    return load_dataset(source_dir)


def _source_dataset_dir(out_dir: str) -> str:
    return os.path.join(out_dir, "artifacts", "ds1_source")


def _final_variant(fine_tune_cfg: Dict[str, Any]) -> Dict[str, str]:
    variant = dict(BASE_VARIANT)
    variant.update(fine_tune_cfg.get("final_variant", {}))
    return variant


def _train_base_model(
    episodes: List[Dict[str, Any]],
    final_variant: Dict[str, str],
    history_len: int,
    ridge_lambda: float,
    bounds: np.ndarray,
):
    train = [ep for ep in episodes if ep["metadata"].get("split_role") == "train"]
    if not train:
        train = episodes
    x_base, y_residual, sample_weight = _base_training_rows(
        train,
        history_len,
        bounds,
        final_variant,
    )
    return _fit_weighted_ridge(x_base, y_residual, ridge_lambda, sample_weight)


def _target_episodes(episodes: List[Dict[str, Any]], role: str) -> List[Dict[str, Any]]:
    if role == "train":
        return [
            ep
            for ep in episodes
            if ep["metadata"].get("split_role") == "fine-tune"
            or ep["metadata"].get("target_window_role") == "target_train"
        ]
    if role == "validation":
        return [
            ep
            for ep in episodes
            if ep["metadata"].get("split_role") == "validation"
            or ep["metadata"].get("target_window_role") == "target_validation"
        ]
    return [
        ep
        for ep in episodes
        if ep["metadata"].get("split_role") == "test-window"
        or ep["metadata"].get("target_window_role") == "target_test"
    ]


def _episodes_up_to_bucket(
    episodes: List[Dict[str, Any]],
    bucket: str,
) -> List[Dict[str, Any]]:
    rank = BUCKETS.index(bucket)
    allowed = set(BUCKETS[: rank + 1])
    return [
        ep
        for ep in episodes
        if ep["metadata"].get("fine_tune_data_bucket") in allowed
    ]


def _adapter_training_rows(
    episodes: List[Dict[str, Any]],
    base_model: Any,
    ft_variant: str,
    final_variant: Dict[str, str],
    history_len: int,
    bounds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    adapter_bounds = bounds * _adapter_bound_scale(ft_variant)
    for episode in episodes:
        states = _state_matrix(episode)
        controls = _control_matrix(episode)
        dt = float(episode["metadata"]["dt"])
        for t in range(len(states) - 1):
            base_next = _base_single_step(
                episode,
                states,
                controls,
                t,
                base_model,
                final_variant,
                history_len,
                bounds,
            )
            correction_dot = (states[t + 1] - base_next) / max(dt, 1e-6)
            xs.append(_adapter_feature_row(episode, states, controls, t, history_len, ft_variant, final_variant))
            ys.append(np.clip(correction_dot, -adapter_bounds, adapter_bounds))
    if not xs:
        return np.zeros((0, 1), dtype=np.float64), np.zeros((0, len(STATE_KEYS)), dtype=np.float64)
    return np.vstack(xs), np.vstack(ys)


def _evaluate_adapter_group(
    episodes: List[Dict[str, Any]],
    base_model: Any,
    adapter_model: Any,
    ft_variant: str,
    final_variant: Dict[str, str],
    history_len: int,
    horizon: int,
    bounds: np.ndarray,
) -> Dict[str, float]:
    if not episodes:
        return {
            "rmse": 0.0,
            "episode_count": 0,
            "constraint_violation_rate": 1.0,
            "adapter_to_base_ratio": 0.0,
        }
    rmses = []
    violations = []
    adapter_norms = []
    base_norms = []
    eval_idx = [STATE_KEYS.index(key) for key in EVAL_STATE_KEYS]
    for episode in episodes:
        target = _state_matrix(episode)
        pred, diag = _rollout_with_adapter(
            episode,
            base_model,
            adapter_model,
            ft_variant,
            final_variant,
            history_len,
            min(horizon, len(target)),
            bounds,
        )
        n = min(len(pred), len(target))
        if n < 2:
            continue
        rmses.append(_rmse(pred[1:n, eval_idx], target[1:n, eval_idx]))
        violations.append(_constraint_violation_rate(pred[:n]))
        adapter_norms.extend(diag["adapter_norms"])
        base_norms.extend(diag["base_norms"])
    return {
        "rmse": _mean(rmses),
        "episode_count": len(episodes),
        "constraint_violation_rate": _mean(violations),
        "adapter_to_base_ratio": _mean(adapter_norms) / max(_mean(base_norms), 1e-9),
    }


def _rollout_with_adapter(
    episode: Dict[str, Any],
    base_model: Any,
    adapter_model: Any,
    ft_variant: str,
    final_variant: Dict[str, str],
    history_len: int,
    horizon_steps: int,
    bounds: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    states = _state_matrix(episode)
    controls = _control_matrix(episode)
    n = min(horizon_steps, len(states))
    pred = np.zeros((n, len(STATE_KEYS)), dtype=np.float64)
    pred[0] = states[0]
    dt = float(episode["metadata"]["dt"])
    adapter_bounds = bounds * _adapter_bound_scale(ft_variant)
    adapter_norms = []
    base_norms = []
    for t in range(n - 1):
        base_next = _base_single_step(
            episode,
            pred,
            controls,
            t,
            base_model,
            final_variant,
            history_len,
            bounds,
        )
        base_dot = (base_next - pred[t]) / max(dt, 1e-6)
        adapter_dot = np.zeros(len(STATE_KEYS), dtype=np.float64)
        if adapter_model is not None:
            row = _adapter_feature_row(
                episode,
                pred,
                controls,
                t,
                history_len,
                ft_variant,
                final_variant,
            )
            adapter_dot = np.clip(
                adapter_model.predict(row[None, :])[0],
                -adapter_bounds,
                adapter_bounds,
            )
        pred[t + 1] = _sanitize_state(base_next + adapter_dot * dt)
        adapter_norms.append(float(np.linalg.norm(adapter_dot)))
        base_norms.append(float(np.linalg.norm(base_dot)))
        if not np.all(np.isfinite(pred[t + 1])):
            pred[t + 1 :] = np.nan
            break
    return pred, {"adapter_norms": adapter_norms, "base_norms": base_norms}


def _base_single_step(
    episode: Dict[str, Any],
    states: np.ndarray,
    controls: np.ndarray,
    t: int,
    base_model: Any,
    final_variant: Dict[str, str],
    history_len: int,
    bounds: np.ndarray,
) -> np.ndarray:
    dt = float(episode["metadata"]["dt"])
    phys_next = _nominal_step(states[t], controls[t], episode, dt)
    row = _base_feature_row(episode, states, controls, t, history_len, final_variant)
    if final_variant["vehicle"] == "V2-small":
        residual_delta = np.clip(
            base_model.predict(row[None, :])[0],
            -bounds * dt,
            bounds * dt,
        )
        return _sanitize_state(phys_next + residual_delta)
    residual_dot = np.clip(base_model.predict(row[None, :])[0], -bounds, bounds)
    return _sanitize_state(phys_next + residual_dot * dt)


def _adapter_feature_row(
    episode: Dict[str, Any],
    states: np.ndarray,
    controls: np.ndarray,
    t: int,
    history_len: int,
    ft_variant: str,
    final_variant: Dict[str, str],
) -> np.ndarray:
    current = states[t]
    control = controls[t]
    context = _context_vector(episode)
    vx = max(abs(float(current[STATE_KEYS.index("vx")])), 0.5)
    omega = np.asarray(
        [current[STATE_KEYS.index(key)] for key in ["omega_fl", "omega_fr", "omega_rl", "omega_rr"]],
        dtype=np.float64,
    )
    radius = float(episode["fixed_vehicle_context"]["wheel_radius"])
    slip = (radius * omega - vx) / max(vx, 0.5)
    torques = np.asarray(
        [
            control[CONTROL_KEYS.index("tau_drv_obs_fl")] - control[CONTROL_KEYS.index("tau_brk_obs_fl")],
            control[CONTROL_KEYS.index("tau_drv_obs_fr")] - control[CONTROL_KEYS.index("tau_brk_obs_fr")],
            control[CONTROL_KEYS.index("tau_drv_obs_rl")] - control[CONTROL_KEYS.index("tau_brk_obs_rl")],
            control[CONTROL_KEYS.index("tau_drv_obs_rr")] - control[CONTROL_KEYS.index("tau_brk_obs_rr")],
        ],
        dtype=np.float64,
    )
    steer = np.asarray(
        [
            control[CONTROL_KEYS.index("sw_angle")],
            control[CONTROL_KEYS.index("steer_cmd")],
            current[STATE_KEYS.index("r")],
            current[STATE_KEYS.index("vy")],
            vx,
        ],
        dtype=np.float64,
    )
    attitude = np.asarray(
        [
            current[STATE_KEYS.index("roll")],
            current[STATE_KEYS.index("pitch")],
            current[STATE_KEYS.index("p")],
            current[STATE_KEYS.index("q")],
            current[STATE_KEYS.index("r")],
            vx,
        ],
        dtype=np.float64,
    )
    if ft_variant == "FT1":
        return np.concatenate([context, attitude[:3], np.tanh(context)])
    if ft_variant == "FT2":
        return np.concatenate([slip, np.abs(slip), torques / 4000.0, steer, context[:6]])
    if ft_variant == "FT3":
        return np.concatenate([attitude, steer, torques[:2] / 4000.0, context[:8]])
    if ft_variant == "FT4":
        return np.concatenate([slip, np.abs(slip), torques / 4000.0, steer, current[:8]])
    if ft_variant == "FT5":
        hist_controls = _history_rows(controls, t, history_len)
        steer_hist = hist_controls[:, [CONTROL_KEYS.index("sw_angle"), CONTROL_KEYS.index("steer_cmd")]]
        return np.concatenate([steer, steer_hist.reshape(-1), context[:6]])
    return _base_feature_row(episode, states, controls, t, history_len, final_variant)


def _history_rows(arr: np.ndarray, end_idx: int, history_len: int) -> np.ndarray:
    rows = []
    max_idx = len(arr) - 1
    for offset in range(history_len - 1, -1, -1):
        rows.append(arr[min(max(0, end_idx - offset), max_idx)])
    return np.asarray(rows, dtype=np.float64)


def _adapter_bound_scale(ft_variant: str) -> float:
    return {
        "FT0": 0.0,
        "FT1": 0.18,
        "FT2": 0.16,
        "FT3": 0.20,
        "FT4": 0.28,
        "FT5": 0.14,
        "FT6": 0.45,
    }.get(ft_variant, 0.20)


def _result_row(
    ft_variant: str,
    bucket: str,
    train_episode_count: int,
    train_transition_count: int,
    validation: Dict[str, float],
    test: Dict[str, float],
    ft0_test_rmse: float,
) -> Dict[str, Any]:
    return {
        "ft_variant": ft_variant,
        "bucket": bucket,
        "train_episode_count": train_episode_count,
        "train_transition_count": train_transition_count,
        "validation_rmse": validation["rmse"],
        "test_rmse": test["rmse"],
        "test_relative_to_ft0": test["rmse"] / max(ft0_test_rmse, 1e-9),
        "test_constraint_violation_rate": test["constraint_violation_rate"],
        "adapter_to_base_ratio": test["adapter_to_base_ratio"],
    }


def _summary_metrics(
    ft_variant: str,
    rows: List[Dict[str, Any]],
    ft0_test_rmse: float,
) -> Dict[str, Any]:
    if not rows:
        return {"fine_tune_run_passed": 0}
    best = min(rows, key=lambda row: row["test_rmse"])
    metrics = {
        "fine_tune_run_passed": 0,
        "fine_tune_improved_over_ft0": int(best["test_relative_to_ft0"] < 1.0),
        "fine_tune_variant_count": 1,
        "fine_tune_bucket_count": len(rows),
        "fine_tune_ft0_test_rmse": ft0_test_rmse,
        "fine_tune_best_test_rmse": best["test_rmse"],
        "fine_tune_best_relative_to_ft0": best["test_relative_to_ft0"],
        "fine_tune_best_constraint_violation_rate": best[
            "test_constraint_violation_rate"
        ],
        "fine_tune_best_adapter_to_base_ratio": best["adapter_to_base_ratio"],
    }
    if ft_variant == "FT0":
        metrics["fine_tune_run_passed"] = int(ft0_test_rmse > 0.0)
    else:
        metrics["fine_tune_run_passed"] = int(
            np.isfinite(best["test_relative_to_ft0"])
            and best["test_relative_to_ft0"] <= 3.0
            and best["test_constraint_violation_rate"] <= 0.05
            and best["train_transition_count"] > 0
        )
    return metrics


def _aggregate_existing_runs(
    out_dir: str,
    fine_tune_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    rows = []
    reports = {}
    for ft_variant, run_dir in FT_RUN_DIRS.items():
        path = os.path.join(run_dir, "artifacts", "fine_tune_report.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        reports[ft_variant] = report
        rows.extend(report.get("rows", []))
    ft0_rmse = reports.get("FT0", {}).get("summary_metrics", {}).get(
        "fine_tune_ft0_test_rmse", 0.0
    )
    small_rows = [row for row in rows if row.get("ft_variant") in ["FT1", "FT2", "FT3", "FT4", "FT5"]]
    full_rows = [row for row in rows if row.get("ft_variant") == "FT6"]
    best_small = min(small_rows, key=lambda row: row["test_rmse"]) if small_rows else {}
    best_full = min(full_rows, key=lambda row: row["test_rmse"]) if full_rows else {}
    summary_metrics = {
        "fine_tune_summary_passed": int(
            bool(best_small)
            and ft0_rmse > 0
            and best_small["test_relative_to_ft0"]
            <= float(fine_tune_cfg.get("small_ft_success_ratio", 0.98))
            and best_small["test_constraint_violation_rate"] <= 0.05
        ),
        "fine_tune_completed_variant_count": len(reports),
        "fine_tune_completed_cell_count": len(rows),
        "fine_tune_ft0_test_rmse": ft0_rmse,
        "fine_tune_best_small_relative_to_ft0": best_small.get(
            "test_relative_to_ft0", 0.0
        ),
        "fine_tune_best_full_relative_to_ft0": best_full.get(
            "test_relative_to_ft0", 0.0
        ),
        "fine_tune_best_small_vs_full_gap": best_small.get("test_rmse", 0.0)
        / max(best_full.get("test_rmse", 0.0), 1e-9)
        if best_full
        else 0.0,
    }
    report = {
        "system": "fine_tune_aggregate_scaffold",
        "rows": rows,
        "reports": sorted(reports),
        "best_small_module": best_small,
        "best_full_model": best_full,
        "summary_metrics": summary_metrics,
        "passed": bool(summary_metrics["fine_tune_summary_passed"]),
        "warnings": [
            "B6 is still a scaffold adapter experiment over simulator proxy target windows.",
            "Real vehicle data is required before making deployment claims.",
        ],
    }
    _write_json(os.path.join(out_dir, "artifacts", "fine_tune_report.json"), report)
    _write_json("reports/B6_fine_tune_summary.json", report)
    _write_fine_tune_markdown("reports/B6_fine_tune.md", report)
    return report


def _write_fine_tune_markdown(path: str, report: Dict[str, Any]) -> None:
    lines = [
        "# B6 Fine-Tune Data Efficiency Scaffold Report",
        "",
        "This report aggregates M9 scaffold runs R038-R045. Rows are target-window 100-step RMSE values; lower is better.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in sorted(report.get("summary_metrics", {}).items()):
        lines.append("| %s | %s |" % (key, value))
    lines.extend(
        [
            "",
            "## Best Small Module",
            "",
            "```json",
            json.dumps(report.get("best_small_module", {}), indent=2, sort_keys=True),
            "```",
            "",
            "## Cells",
            "",
            "| FT | Bucket | Train Episodes | Test RMSE | Relative to FT0 | Constraint Rate | Adapter/Base |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(report.get("rows", []), key=lambda item: (item["ft_variant"], item["bucket"])):
        lines.append(
            "| {ft_variant} | {bucket} | {train_episode_count} | {test_rmse:.6f} | {test_relative_to_ft0:.6f} | {test_constraint_violation_rate:.6f} | {adapter_to_base_ratio:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `FT0` is no fine-tune and only has `FTD0`.",
            "- `FT1-FT6` use cumulative target train episodes from `FTD1` to the reported bucket.",
            "- This is not a PyTorch fine-tuning result; it validates the experiment matrix, target-window split, metrics, and reporting path.",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))
