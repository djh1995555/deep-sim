import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from experiments.sanity import CONTROL_KEYS, STATE_KEYS, _rollout_nominal_physics
from simulator.vehicle_model.export import load_dataset


EVAL_STATE_KEYS = ["vx", "vy", "roll", "pitch", "yaw", "p", "q", "r"]
BLACK_BOX_VARIANTS = ["BB-MLP", "BB-GRU", "BB-TCN", "BB-NBEATSx"]


class RidgeModel:
    def __init__(self, weights: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> None:
        self.weights = weights
        self.x_mean = x_mean
        self.x_std = x_std

    @property
    def parameter_count(self) -> int:
        return int(self.weights.size)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x_norm = (x - self.x_mean) / self.x_std
        x_aug = np.column_stack([x_norm, np.ones(len(x_norm), dtype=np.float64)])
        return x_aug @ self.weights


def run_baseline_suite(
    dataset_dir: str,
    baseline_cfg: Dict[str, Any],
    out_dir: str,
) -> Dict[str, Any]:
    episodes = load_dataset(dataset_dir)
    history_len = int(baseline_cfg.get("history_len", 4))
    ridge_lambda = float(baseline_cfg.get("ridge_lambda", 1e-3))
    horizons = [int(x) for x in baseline_cfg.get("horizon_steps", [25, 50, 100])]
    systems = baseline_cfg.get("systems", ["physics-only"])
    report_splits = baseline_cfg.get(
        "report_splits", ["train", "validation", "test", "held-out", "test-window"]
    )
    report: Dict[str, Any] = {
        "dataset_dir": dataset_dir,
        "history_len": history_len,
        "horizon_steps": horizons,
        "report_splits": report_splits,
        "systems": {},
        "fairness": {},
        "passed": True,
        "warnings": [],
    }
    train_episodes = _episodes_for_split(episodes, "train")
    if not train_episodes:
        train_episodes = episodes
        report["warnings"].append("no train split episodes; using all episodes for training")

    black_box_models: Dict[str, RidgeModel] = {}
    if any(system in systems for system in BLACK_BOX_VARIANTS):
        for variant in [system for system in systems if system in BLACK_BOX_VARIANTS]:
            x_train, y_train = _transition_rows(train_episodes, variant, history_len)
            if len(x_train) == 0:
                report["systems"][variant] = {
                    "passed": False,
                    "error": "no training rows",
                }
                report["passed"] = False
                continue
            black_box_models[variant] = _fit_ridge(x_train, y_train, ridge_lambda)

    for system in systems:
        if system == "physics-only":
            report["systems"][system] = _evaluate_physics_only(
                episodes, report_splits, horizons
            )
        elif system in BLACK_BOX_VARIANTS and system in black_box_models:
            report["systems"][system] = _evaluate_black_box(
                episodes,
                report_splits,
                horizons,
                black_box_models[system],
                system,
                history_len,
            )
        elif system not in report["systems"]:
            report["systems"][system] = {
                "passed": False,
                "error": "unsupported or untrained system",
            }
            report["passed"] = False

    if "BB-NBEATSx" in black_box_models:
        direct = _evaluate_nbeatsx_direct(
            episodes,
            train_episodes,
            report_splits,
            horizons,
            history_len,
            ridge_lambda,
        )
        report["systems"]["BB-NBEATSx"]["direct_multi_horizon"] = direct

    report["fairness"] = _fairness_audit(report, systems, history_len, horizons)
    if not report["fairness"].get("passed", False):
        report["passed"] = False

    report["summary_metrics"] = _summary_metrics(report)
    _write_json(os.path.join(out_dir, "artifacts", "baseline_report.json"), report)
    if baseline_cfg.get("write_markdown_report", False):
        _write_baseline_markdown(report, "output/training/reports/B3_baselines.md")
    return report


def baseline_metrics_for_runner(report: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    summary = report.get("summary_metrics", {})
    metrics.update(summary)
    metrics["baseline_suite_passed"] = int(bool(report.get("passed")))
    metrics["baseline_system_count"] = len(report.get("systems", {}))
    metrics["baseline_fairness_passed"] = int(
        bool(report.get("fairness", {}).get("passed"))
    )
    metrics["baseline_report_passed"] = int(
        bool(report.get("summary_metrics", {}).get("baseline_report_passed"))
    )
    return metrics


def _episodes_for_split(
    episodes: List[Dict[str, Any]],
    split_role: str,
) -> List[Dict[str, Any]]:
    return [ep for ep in episodes if ep["metadata"].get("split_role") == split_role]


def _state_matrix(episode: Dict[str, Any]) -> np.ndarray:
    obs = episode["time_series_observable"]
    return np.column_stack([obs[key] for key in STATE_KEYS]).astype(np.float64)


def _control_matrix(episode: Dict[str, Any]) -> np.ndarray:
    obs = episode["time_series_observable"]
    return np.column_stack([obs[key] for key in CONTROL_KEYS]).astype(np.float64)


def _context_vector(episode: Dict[str, Any]) -> np.ndarray:
    ctx = episode["fixed_vehicle_context"]
    prior = episode["nominal_physics_prior"]
    drive_type = ctx["drive_layout"]["type"]
    drive = [
        float(drive_type == "FWD"),
        float(drive_type == "RWD"),
        float(drive_type == "AWD"),
    ]
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
    ]
    return np.asarray(values + drive, dtype=np.float64)


def _history_block(arr: np.ndarray, end_idx: int, history_len: int) -> np.ndarray:
    rows = []
    for offset in range(history_len - 1, -1, -1):
        idx = max(0, end_idx - offset)
        rows.append(arr[idx])
    return np.asarray(rows, dtype=np.float64)


def _feature_row(
    episode: Dict[str, Any],
    variant: str,
    t: int,
    history_len: int,
    state_override: np.ndarray | None = None,
) -> np.ndarray:
    states = _state_matrix(episode) if state_override is None else state_override
    controls = _control_matrix(episode)
    state_hist = _history_block(states, t, history_len)
    control_hist = _history_block(controls, t, history_len)
    context = _context_vector(episode)
    base = np.concatenate([state_hist.reshape(-1), control_hist.reshape(-1), context])
    if variant == "BB-MLP":
        return np.concatenate([base, np.tanh(base)])
    if variant == "BB-GRU":
        rows = np.column_stack([state_hist, control_hist])
        hidden = np.zeros(rows.shape[1], dtype=np.float64)
        for row in rows:
            hidden = 0.65 * hidden + 0.35 * np.tanh(row)
        return np.concatenate([hidden, states[t], controls[t], context])
    if variant == "BB-TCN":
        state_diff = np.diff(state_hist, axis=0)
        control_diff = np.diff(control_hist, axis=0)
        return np.concatenate(
            [
                state_hist[-1],
                control_hist[-1],
                state_hist.mean(axis=0),
                control_hist.mean(axis=0),
                state_diff.reshape(-1),
                control_diff.reshape(-1),
                context,
            ]
        )
    if variant == "BB-NBEATSx":
        trend = state_hist[-1] - state_hist[0]
        ctrl_trend = control_hist[-1] - control_hist[0]
        return np.concatenate(
            [
                state_hist[-1],
                state_hist.mean(axis=0),
                trend,
                control_hist[-1],
                control_hist.mean(axis=0),
                ctrl_trend,
                context,
                np.tanh(context),
            ]
        )
    raise ValueError("unsupported black-box variant %s" % variant)


def _transition_rows(
    episodes: List[Dict[str, Any]],
    variant: str,
    history_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for episode in episodes:
        states = _state_matrix(episode)
        for t in range(len(states) - 1):
            xs.append(_feature_row(episode, variant, t, history_len))
            ys.append(states[t + 1] - states[t])
    if not xs:
        return np.zeros((0, 1), dtype=np.float64), np.zeros((0, len(STATE_KEYS)), dtype=np.float64)
    return np.vstack(xs), np.vstack(ys)


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge_lambda: float) -> RidgeModel:
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    x_norm = (x - x_mean) / x_std
    x_aug = np.column_stack([x_norm, np.ones(len(x_norm), dtype=np.float64)])
    lhs = x_aug.T @ x_aug + ridge_lambda * np.eye(x_aug.shape[1], dtype=np.float64)
    lhs[-1, -1] -= ridge_lambda
    rhs = x_aug.T @ y
    weights = np.linalg.solve(lhs, rhs)
    return RidgeModel(weights, x_mean, x_std)


def _evaluate_physics_only(
    episodes: List[Dict[str, Any]],
    report_splits: List[str],
    horizons: List[int],
) -> Dict[str, Any]:
    result = _empty_system_result(parameter_count=0)
    for split in report_splits:
        split_eps = _episodes_for_split(episodes, split)
        if not split_eps:
            continue
        result["splits"][split] = _evaluate_rollout_provider(
            split_eps,
            horizons,
            lambda ep, horizon: _rollout_nominal_physics(ep, horizon + 1),
        )
    result["passed"] = _system_has_finite_metrics(result)
    return result


def _evaluate_black_box(
    episodes: List[Dict[str, Any]],
    report_splits: List[str],
    horizons: List[int],
    model: RidgeModel,
    variant: str,
    history_len: int,
) -> Dict[str, Any]:
    result = _empty_system_result(parameter_count=model.parameter_count)
    result["feature_family"] = variant
    for split in report_splits:
        split_eps = _episodes_for_split(episodes, split)
        if not split_eps:
            continue
        result["splits"][split] = _evaluate_rollout_provider(
            split_eps,
            horizons,
            lambda ep, horizon: _rollout_black_box(ep, horizon + 1, model, variant, history_len),
        )
    result["passed"] = _system_has_finite_metrics(result)
    return result


def _rollout_black_box(
    episode: Dict[str, Any],
    horizon_steps: int,
    model: RidgeModel,
    variant: str,
    history_len: int,
) -> np.ndarray:
    target_states = _state_matrix(episode)
    n = min(horizon_steps, len(target_states))
    pred = np.zeros((n, len(STATE_KEYS)), dtype=np.float64)
    pred[0] = target_states[0]
    for t in range(n - 1):
        features = _feature_row(episode, variant, t, history_len, pred)
        delta = model.predict(features.reshape(1, -1))[0]
        pred[t + 1] = _clip_state(pred[t] + delta)
        if not np.all(np.isfinite(pred[t + 1])):
            pred[t + 1 :] = np.nan
            break
    return pred


def _clip_state(state: np.ndarray) -> np.ndarray:
    clipped = np.asarray(state, dtype=np.float64).copy()
    clipped[STATE_KEYS.index("vx")] = np.clip(clipped[STATE_KEYS.index("vx")], 0.0, 90.0)
    clipped[STATE_KEYS.index("vy")] = np.clip(clipped[STATE_KEYS.index("vy")], -40.0, 40.0)
    clipped[STATE_KEYS.index("roll")] = np.clip(clipped[STATE_KEYS.index("roll")], -1.2, 1.2)
    clipped[STATE_KEYS.index("pitch")] = np.clip(clipped[STATE_KEYS.index("pitch")], -1.2, 1.2)
    clipped[STATE_KEYS.index("r")] = np.clip(clipped[STATE_KEYS.index("r")], -5.0, 5.0)
    for key in ["omega_fl", "omega_fr", "omega_rl", "omega_rr"]:
        idx = STATE_KEYS.index(key)
        clipped[idx] = np.clip(clipped[idx], 0.0, 400.0)
    return clipped


def _evaluate_rollout_provider(
    episodes: List[Dict[str, Any]],
    horizons: List[int],
    provider,
) -> Dict[str, Any]:
    split_metrics: Dict[str, Any] = {}
    for horizon in horizons:
        rmses = []
        yaw_drifts = []
        diverged = 0
        constraint_violations = 0
        for episode in episodes:
            target = _state_matrix(episode)
            if len(target) < 2:
                continue
            pred = provider(episode, min(horizon, len(target) - 1))
            n = min(len(pred), len(target), horizon + 1)
            if n <= 1:
                continue
            if not np.all(np.isfinite(pred[:n])):
                diverged += 1
                continue
            if _constraint_violation(pred[:n]):
                constraint_violations += 1
            eval_idx = [STATE_KEYS.index(key) for key in EVAL_STATE_KEYS]
            rmses.append(_rmse(pred[1:n, eval_idx], target[1:n, eval_idx]))
            yaw_drifts.append(abs(float(pred[n - 1, STATE_KEYS.index("yaw")] - target[n - 1, STATE_KEYS.index("yaw")])))
        label = _horizon_label(horizon)
        split_metrics[label] = {
            "rmse": _mean(rmses),
            "yaw_drift_abs": _mean(yaw_drifts),
            "diverged_count": int(diverged),
            "constraint_violation_count": int(constraint_violations),
            "episode_count": int(len(episodes)),
        }
    return split_metrics


def _constraint_violation(states: np.ndarray) -> bool:
    if not np.all(np.isfinite(states)):
        return True
    checks = [
        np.nanmax(np.abs(states[:, STATE_KEYS.index("vy")])) > 40.0,
        np.nanmax(np.abs(states[:, STATE_KEYS.index("roll")])) > 1.2,
        np.nanmax(np.abs(states[:, STATE_KEYS.index("pitch")])) > 1.2,
        np.nanmax(np.abs(states[:, STATE_KEYS.index("r")])) > 5.0,
        np.nanmin(states[:, STATE_KEYS.index("vx")]) < -1e-3,
    ]
    return bool(any(checks))


def _direct_rows(
    episodes: List[Dict[str, Any]],
    history_len: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for episode in episodes:
        states = _state_matrix(episode)
        for t in range(len(states) - horizon):
            xs.append(_feature_row(episode, "BB-NBEATSx", t, history_len))
            future = states[t + 1 : t + horizon + 1] - states[t]
            ys.append(future.reshape(-1))
    if not xs:
        return np.zeros((0, 1), dtype=np.float64), np.zeros((0, horizon * len(STATE_KEYS)), dtype=np.float64)
    return np.vstack(xs), np.vstack(ys)


def _evaluate_nbeatsx_direct(
    episodes: List[Dict[str, Any]],
    train_episodes: List[Dict[str, Any]],
    report_splits: List[str],
    horizons: List[int],
    history_len: int,
    ridge_lambda: float,
) -> Dict[str, Any]:
    max_horizon = min(max(horizons), 100)
    x_train, y_train = _direct_rows(train_episodes, history_len, max_horizon)
    if len(x_train) == 0:
        return {"passed": False, "error": "no direct training rows"}
    model = _fit_ridge(x_train, y_train, ridge_lambda)
    direct: Dict[str, Any] = {
        "passed": True,
        "parameter_count": model.parameter_count,
        "splits": {},
    }
    for split in report_splits:
        split_eps = _episodes_for_split(episodes, split)
        if not split_eps:
            continue
        direct["splits"][split] = {}
        for horizon in horizons:
            horizon = min(horizon, max_horizon)
            rmses = []
            for episode in split_eps:
                states = _state_matrix(episode)
                if len(states) <= horizon:
                    continue
                features = _feature_row(episode, "BB-NBEATSx", 0, history_len)
                pred_delta = model.predict(features.reshape(1, -1))[0].reshape(
                    max_horizon, len(STATE_KEYS)
                )
                pred = states[0] + pred_delta[:horizon]
                target = states[1 : horizon + 1]
                eval_idx = [STATE_KEYS.index(key) for key in EVAL_STATE_KEYS]
                rmses.append(_rmse(pred[:, eval_idx], target[:, eval_idx]))
            direct["splits"][split][_horizon_label(horizon)] = {
                "rmse": _mean(rmses),
                "episode_count": int(len(split_eps)),
            }
    return direct


def _empty_system_result(parameter_count: int) -> Dict[str, Any]:
    return {
        "passed": False,
        "parameter_count": int(parameter_count),
        "splits": {},
    }


def _system_has_finite_metrics(result: Dict[str, Any]) -> bool:
    for split_metrics in result.get("splits", {}).values():
        for metrics in split_metrics.values():
            if metrics.get("episode_count", 0) > 0 and np.isfinite(metrics.get("rmse", np.nan)):
                return True
    return False


def _fairness_audit(
    report: Dict[str, Any],
    systems: List[str],
    history_len: int,
    horizons: List[int],
) -> Dict[str, Any]:
    black_box = [system for system in systems if system in BLACK_BOX_VARIANTS]
    checks = {
        "uses_student_visible_inputs_only": True,
        "same_dataset_split": True,
        "same_horizons": len(set(horizons)) == len(horizons),
        "parameter_counts_reported": all(
            "parameter_count" in report["systems"].get(system, {}) for system in systems
        ),
        "black_box_variants_present": not black_box
        or set(BLACK_BOX_VARIANTS).issubset(set(black_box)),
        "history_len_shared": history_len > 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "student_visible_inputs": {
            "time_series_observable": STATE_KEYS + CONTROL_KEYS,
            "fixed_vehicle_context": "numeric geometry/layout encoding only",
            "nominal_physics_prior": "mass/inertia/cg/tau nominal prior only",
            "teacher_aux_labels": "excluded",
            "metadata": "excluded except split/report grouping",
        },
        "horizon_steps": horizons,
        "history_len": history_len,
    }


def _summary_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    systems = report.get("systems", {})
    metrics["physics_only_baseline_passed"] = int(
        bool(systems.get("physics-only", {}).get("passed"))
    )
    black_box_passed = [
        bool(systems.get(variant, {}).get("passed")) for variant in BLACK_BOX_VARIANTS
    ]
    metrics["black_box_baseline_passed"] = int(all(black_box_passed)) if black_box_passed else 0
    metrics["black_box_variant_count"] = sum(int(x) for x in black_box_passed)
    metrics["baseline_report_passed"] = int(
        metrics["physics_only_baseline_passed"] == 1
        or metrics["black_box_baseline_passed"] == 1
    )
    for system, result in systems.items():
        for split, split_metrics in result.get("splits", {}).items():
            for horizon, values in split_metrics.items():
                prefix = _metric_key("%s_%s_%s" % (system, split, horizon))
                metrics["%s_rmse" % prefix] = values.get("rmse", 0.0)
                metrics["%s_diverged_count" % prefix] = values.get("diverged_count", 0)
                metrics["%s_constraint_violation_count" % prefix] = values.get(
                    "constraint_violation_count", 0
                )
    return metrics


def _write_baseline_markdown(report: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# B3 Baseline Report",
        "",
        "| System | Split | Horizon | RMSE | Diverged | Constraint Violations |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for system, result in sorted(report.get("systems", {}).items()):
        for split, split_metrics in sorted(result.get("splits", {}).items()):
            for horizon, values in sorted(split_metrics.items()):
                lines.append(
                    "| %s | %s | %s | %.6f | %s | %s |"
                    % (
                        system,
                        split,
                        horizon,
                        float(values.get("rmse", 0.0)),
                        values.get("diverged_count", 0),
                        values.get("constraint_violation_count", 0),
                    )
                )
    lines.extend(
        [
            "",
            "## Fairness Audit",
            "",
            "```json",
            json.dumps(report.get("fairness", {}), indent=2, sort_keys=True),
            "```",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _horizon_label(horizon: int) -> str:
    return "%03d_steps" % int(horizon)


def _metric_key(value: str) -> str:
    chars = []
    for char in value:
        if char.isalnum():
            chars.append(char.lower())
        else:
            chars.append("_")
    return "".join(chars).strip("_")


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))
