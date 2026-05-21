import json
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from experiments.baselines import _fit_ridge
from experiments.sanity import CONTROL_KEYS, STATE_KEYS, _rollout_nominal_physics
from teacher_simulator.export import load_dataset


EVAL_STATE_KEYS = ["vx", "vy", "roll", "pitch", "yaw", "p", "q", "r"]
DEFAULT_REPORT_GROUPS = [
    "train",
    "validation",
    "test",
    "seen-config",
    "held-out-road",
    "held-out-vehicle",
    "test-window",
]


def run_base_hybrid_suite(
    dataset_dir: str,
    hybrid_cfg: Dict[str, Any],
    out_dir: str,
) -> Dict[str, Any]:
    episodes = load_dataset(dataset_dir)
    seeds = [int(x) for x in hybrid_cfg.get("seeds", [hybrid_cfg.get("seed", 23)])]
    history_len = int(hybrid_cfg.get("history_len", 4))
    ridge_lambda = float(hybrid_cfg.get("ridge_lambda", 0.01))
    horizons = [int(x) for x in hybrid_cfg.get("horizon_steps", [25, 50, 100])]
    report_groups = hybrid_cfg.get("report_groups", DEFAULT_REPORT_GROUPS)
    bounds = _residual_dot_bounds(hybrid_cfg)
    reports = []
    for seed in seeds:
        reports.append(
            _run_single_seed(
                episodes,
                seed,
                history_len,
                ridge_lambda,
                horizons,
                report_groups,
                bounds,
                hybrid_cfg,
            )
        )

    report: Dict[str, Any] = {
        "dataset_dir": dataset_dir,
        "system": "Base = E1 + T1 + F1 + S1 + M1a + V1 + U0 scaffold",
        "history_len": history_len,
        "horizon_steps": horizons,
        "report_groups": report_groups,
        "seeds": seeds,
        "seed_reports": reports,
        "summary_metrics": _summary_metrics(reports, hybrid_cfg),
        "passed": True,
        "warnings": [
            "M5 uses a numpy ridge residual-dot scaffold; it validates experiment plumbing before full PyTorch module implementation.",
            "Teacher-only labels are used only for diagnostics, not as model inputs.",
        ],
    }
    report["passed"] = all(
        bool(report["summary_metrics"].get(key, 0))
        for key in _required_success_keys(hybrid_cfg)
    )
    _write_json(os.path.join(out_dir, "artifacts", "base_hybrid_report.json"), report)
    if hybrid_cfg.get("write_markdown_report", False):
        _write_hybrid_markdown(report, "reports/B3_base_hybrid.md")
    return report


def base_hybrid_metrics_for_runner(report: Dict[str, Any]) -> Dict[str, Any]:
    metrics = dict(report.get("summary_metrics", {}))
    metrics["base_hybrid_suite_passed"] = int(bool(report.get("passed")))
    metrics["base_hybrid_seed_count"] = len(report.get("seed_reports", []))
    return metrics


def _run_single_seed(
    episodes: List[Dict[str, Any]],
    seed: int,
    history_len: int,
    ridge_lambda: float,
    horizons: List[int],
    report_groups: List[str],
    bounds: np.ndarray,
    hybrid_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    train_episodes = _episodes_for_group(episodes, "train")
    if not train_episodes:
        train_episodes = episodes
    bootstrap_fraction = float(hybrid_cfg.get("bootstrap_fraction", 1.0))
    if bootstrap_fraction < 1.0 and len(train_episodes) > 1:
        count = max(1, int(round(len(train_episodes) * bootstrap_fraction)))
        indices = rng.choice(len(train_episodes), size=count, replace=True)
        train_episodes = [train_episodes[int(i)] for i in indices]

    x_base, y_residual_dot = _base_training_rows(train_episodes, history_len, bounds)
    base_model = _fit_ridge(x_base, y_residual_dot, ridge_lambda)
    x_bb, y_delta_dot = _black_box_training_rows(train_episodes, history_len, bounds)
    black_box_model = _fit_ridge(x_bb, y_delta_dot, ridge_lambda)

    groups: Dict[str, Any] = {}
    for group in report_groups:
        group_eps = _episodes_for_group(episodes, group)
        groups[group] = _evaluate_group(
            group_eps,
            base_model,
            black_box_model,
            history_len,
            horizons,
            bounds,
        )

    diagnostics = _component_diagnostics(episodes, base_model, history_len, bounds)
    return {
        "seed": seed,
        "train_episode_count": len(train_episodes),
        "train_transition_count": int(len(x_base)),
        "base_parameter_count": int(base_model.parameter_count),
        "black_box_parameter_count": int(black_box_model.parameter_count),
        "groups": groups,
        "diagnostics": diagnostics,
    }


def _base_training_rows(
    episodes: List[Dict[str, Any]],
    history_len: int,
    bounds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for episode in episodes:
        states = _state_matrix(episode)
        controls = _control_matrix(episode)
        dt = float(episode["metadata"]["dt"])
        for t in range(len(states) - 1):
            phys_next = _nominal_step(states[t], controls[t], episode, dt)
            residual_dot = (states[t + 1] - phys_next) / max(dt, 1e-6)
            xs.append(_base_feature_row(episode, states, controls, t, history_len))
            ys.append(np.clip(residual_dot, -bounds, bounds))
    return _stack_or_empty(xs), _stack_or_empty(ys, len(STATE_KEYS))


def _black_box_training_rows(
    episodes: List[Dict[str, Any]],
    history_len: int,
    bounds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for episode in episodes:
        states = _state_matrix(episode)
        controls = _control_matrix(episode)
        dt = float(episode["metadata"]["dt"])
        for t in range(len(states) - 1):
            delta_dot = (states[t + 1] - states[t]) / max(dt, 1e-6)
            xs.append(_black_box_feature_row(episode, states, controls, t, history_len))
            ys.append(np.clip(delta_dot, -bounds, bounds))
    return _stack_or_empty(xs), _stack_or_empty(ys, len(STATE_KEYS))


def _evaluate_group(
    episodes: List[Dict[str, Any]],
    base_model: Any,
    black_box_model: Any,
    history_len: int,
    horizons: List[int],
    bounds: np.ndarray,
) -> Dict[str, Any]:
    if not episodes:
        return {"episode_count": 0, "passed": False}
    result: Dict[str, Any] = {"episode_count": len(episodes), "horizons": {}}
    for horizon in horizons:
        base_preds = []
        physics_preds = []
        black_box_preds = []
        targets = []
        base_diag = []
        physics_diverged = 0
        base_diverged = 0
        bb_diverged = 0
        for episode in episodes:
            target = _state_matrix(episode)[: min(horizon, len(_state_matrix(episode)))]
            if len(target) < 2:
                continue
            physics = _rollout_nominal_physics(episode, len(target))
            base = _rollout_base_hybrid(
                episode, base_model, history_len, len(target), bounds
            )
            black_box = _rollout_black_box(
                episode, black_box_model, history_len, len(target), bounds
            )
            physics_diverged += int(not _is_stable(physics))
            base_diverged += int(not _is_stable(base))
            bb_diverged += int(not _is_stable(black_box))
            n = min(len(target), len(physics), len(base), len(black_box))
            if n < 2:
                continue
            base_preds.append(base[:n])
            physics_preds.append(physics[:n])
            black_box_preds.append(black_box[:n])
            targets.append(target[:n])
            base_diag.append(_rollout_diagnostics(episode, base_model, history_len, n, bounds))
        if not targets:
            result["horizons"][str(horizon)] = {"passed": False}
            continue
        target_all = np.vstack(targets)
        base_all = np.vstack(base_preds)
        physics_all = np.vstack(physics_preds)
        bb_all = np.vstack(black_box_preds)
        idx = [STATE_KEYS.index(key) for key in EVAL_STATE_KEYS]
        base_rmse = _rmse(base_all[:, idx], target_all[:, idx])
        physics_rmse = _rmse(physics_all[:, idx], target_all[:, idx])
        bb_rmse = _rmse(bb_all[:, idx], target_all[:, idx])
        yaw_idx = STATE_KEYS.index("yaw")
        diag = _merge_diagnostics(base_diag)
        result["horizons"][str(horizon)] = {
            "base_rmse": base_rmse,
            "physics_rmse": physics_rmse,
            "black_box_rmse": bb_rmse,
            "base_vs_physics_ratio": base_rmse / max(physics_rmse, 1e-9),
            "base_vs_black_box_ratio": base_rmse / max(bb_rmse, 1e-9),
            "base_yaw_drift_mean_abs": float(np.mean(np.abs(base_all[:, yaw_idx] - target_all[:, yaw_idx]))),
            "physics_yaw_drift_mean_abs": float(np.mean(np.abs(physics_all[:, yaw_idx] - target_all[:, yaw_idx]))),
            "base_constraint_violation_rate": _constraint_violation_rate(base_all),
            "physics_constraint_violation_rate": _constraint_violation_rate(physics_all),
            "black_box_constraint_violation_rate": _constraint_violation_rate(bb_all),
            "base_diverged_count": base_diverged,
            "physics_diverged_count": physics_diverged,
            "black_box_diverged_count": bb_diverged,
            **diag,
        }
    result["passed"] = True
    return result


def _rollout_base_hybrid(
    episode: Dict[str, Any],
    model: Any,
    history_len: int,
    horizon_steps: int,
    bounds: np.ndarray,
) -> np.ndarray:
    states = _state_matrix(episode)
    controls = _control_matrix(episode)
    n = min(horizon_steps, len(states))
    pred = np.zeros((n, len(STATE_KEYS)), dtype=np.float64)
    pred[0] = states[0]
    dt = float(episode["metadata"]["dt"])
    for t in range(n - 1):
        phys_next = _nominal_step(pred[t], controls[t], episode, dt)
        row = _base_feature_row(episode, pred, controls, t, history_len)
        residual_dot = np.clip(model.predict(row[None, :])[0], -bounds, bounds)
        pred[t + 1] = _sanitize_state(phys_next + residual_dot * dt)
        if not np.all(np.isfinite(pred[t + 1])):
            pred[t + 1 :] = np.nan
            break
    return pred


def _rollout_black_box(
    episode: Dict[str, Any],
    model: Any,
    history_len: int,
    horizon_steps: int,
    bounds: np.ndarray,
) -> np.ndarray:
    states = _state_matrix(episode)
    controls = _control_matrix(episode)
    n = min(horizon_steps, len(states))
    pred = np.zeros((n, len(STATE_KEYS)), dtype=np.float64)
    pred[0] = states[0]
    dt = float(episode["metadata"]["dt"])
    for t in range(n - 1):
        row = _black_box_feature_row(episode, pred, controls, t, history_len)
        delta_dot = np.clip(model.predict(row[None, :])[0], -bounds, bounds)
        pred[t + 1] = _sanitize_state(pred[t] + delta_dot * dt)
        if not np.all(np.isfinite(pred[t + 1])):
            pred[t + 1 :] = np.nan
            break
    return pred


def _rollout_diagnostics(
    episode: Dict[str, Any],
    model: Any,
    history_len: int,
    horizon_steps: int,
    bounds: np.ndarray,
) -> Dict[str, float]:
    states = _state_matrix(episode)
    controls = _control_matrix(episode)
    n = min(horizon_steps, len(states))
    pred = np.zeros((n, len(STATE_KEYS)), dtype=np.float64)
    pred[0] = states[0]
    dt = float(episode["metadata"]["dt"])
    residual_norms = []
    physics_norms = []
    residual_diffs = []
    prev_residual = None
    for t in range(n - 1):
        phys_next = _nominal_step(pred[t], controls[t], episode, dt)
        row = _base_feature_row(episode, pred, controls, t, history_len)
        residual_dot = np.clip(model.predict(row[None, :])[0], -bounds, bounds)
        physics_dot = (phys_next - pred[t]) / max(dt, 1e-6)
        residual_norms.append(float(np.linalg.norm(residual_dot)))
        physics_norms.append(float(np.linalg.norm(physics_dot)))
        if prev_residual is not None:
            residual_diffs.append(float(np.linalg.norm(residual_dot - prev_residual)))
        prev_residual = residual_dot
        pred[t + 1] = _sanitize_state(phys_next + residual_dot * dt)
    return {
        "residual_dot_norm_mean": _mean(residual_norms),
        "physics_dot_norm_mean": _mean(physics_norms),
        "residual_to_physics_ratio": _mean(residual_norms) / max(_mean(physics_norms), 1e-9),
        "residual_dot_smoothness": _mean(residual_diffs),
    }


def _component_diagnostics(
    episodes: List[Dict[str, Any]],
    model: Any,
    history_len: int,
    bounds: np.ndarray,
) -> Dict[str, float]:
    fz_errors = []
    mu_values = []
    usage_values = []
    delta_errors = []
    sampled = episodes[: min(24, len(episodes))]
    for episode in sampled:
        aux = episode["teacher_aux_labels"]
        if "Fz_true_i" in aux:
            fz_true = np.asarray(aux["Fz_true_i"], dtype=np.float64)
            fz_nominal = _nominal_fz(episode, len(fz_true))
            fz_errors.append(_rmse(fz_nominal, fz_true))
        if "mu_true_i" in aux:
            mu_values.extend(np.asarray(aux["mu_true_i"], dtype=np.float64).reshape(-1).tolist())
        if "friction_usage_i" in aux:
            usage_values.extend(
                np.asarray(aux["friction_usage_i"], dtype=np.float64).reshape(-1).tolist()
            )
        if "delta_eff_i" in aux:
            obs = episode["time_series_observable"]
            ratio = float(episode["fixed_vehicle_context"]["steering_layout"]["steering_ratio_nominal"])
            delta_nom = np.asarray(obs["sw_angle"], dtype=np.float64) / max(ratio, 1e-6)
            delta_true = np.asarray(aux["delta_eff_i"], dtype=np.float64)
            delta_errors.append(_rmse(delta_nom[:, None], delta_true))
    _ = model
    _ = history_len
    _ = bounds
    return {
        "diagnostic_fz_nominal_rmse": _mean(fz_errors),
        "diagnostic_mu_mean": _mean(mu_values),
        "diagnostic_friction_usage_p99": _percentile(usage_values, 99),
        "diagnostic_delta_nominal_rmse": _mean(delta_errors),
        "diagnostic_sample_episode_count": len(sampled),
        "diagnostic_rollout_available": int(bool(sampled)),
    }


def _base_feature_row(
    episode: Dict[str, Any],
    states: np.ndarray,
    controls: np.ndarray,
    t: int,
    history_len: int,
) -> np.ndarray:
    state_hist = _history_block(states, t, history_len)
    control_hist = _history_block(controls, t, history_len)
    current = state_hist[-1]
    control = control_hist[-1]
    dt = float(episode["metadata"]["dt"])
    phys_next = _nominal_step(current, control, episode, dt)
    phys_dot = (phys_next - current) / max(dt, 1e-6)
    context = _context_vector(episode)
    shared = _causal_gru_proxy(np.column_stack([state_hist, control_hist]))
    module_proxy = _module_proxy_features(episode, state_hist, control_hist, phys_dot)
    return np.concatenate(
        [
            shared,
            state_hist.reshape(-1),
            control_hist.reshape(-1),
            current,
            control,
            phys_dot,
            module_proxy,
            context,
        ]
    )


def _black_box_feature_row(
    episode: Dict[str, Any],
    states: np.ndarray,
    controls: np.ndarray,
    t: int,
    history_len: int,
) -> np.ndarray:
    state_hist = _history_block(states, t, history_len)
    control_hist = _history_block(controls, t, history_len)
    context = _context_vector(episode)
    trend = state_hist[-1] - state_hist[0]
    control_trend = control_hist[-1] - control_hist[0]
    return np.concatenate(
        [
            state_hist[-1],
            state_hist.mean(axis=0),
            trend,
            control_hist[-1],
            control_hist.mean(axis=0),
            control_trend,
            context,
            np.tanh(context),
        ]
    )


def _module_proxy_features(
    episode: Dict[str, Any],
    state_hist: np.ndarray,
    control_hist: np.ndarray,
    phys_dot: np.ndarray,
) -> np.ndarray:
    ctx = episode["fixed_vehicle_context"]
    prior = episode["nominal_physics_prior"]
    radius = float(ctx["wheel_radius"])
    wheelbase = float(ctx["wheelbase"])
    track = 0.5 * (float(ctx["track_front"]) + float(ctx["track_rear"]))
    ratio = float(ctx["steering_layout"]["steering_ratio_nominal"])
    mass = float(prior["mass_nominal"])
    cg_z = float(prior["cg_z_nominal"])
    vx = max(abs(float(state_hist[-1, STATE_KEYS.index("vx")])), 0.5)
    vy = float(state_hist[-1, STATE_KEYS.index("vy")])
    r = float(state_hist[-1, STATE_KEYS.index("r")])
    sw = float(control_hist[-1, CONTROL_KEYS.index("sw_angle")])
    steer_cmd = float(control_hist[-1, CONTROL_KEYS.index("steer_cmd")])
    delta_nom = np.clip(sw / max(ratio, 1e-6), -0.6, 0.6)
    tau_drv = np.array(
        [
            control_hist[-1, CONTROL_KEYS.index("tau_drv_obs_fl")],
            control_hist[-1, CONTROL_KEYS.index("tau_drv_obs_fr")],
            control_hist[-1, CONTROL_KEYS.index("tau_drv_obs_rl")],
            control_hist[-1, CONTROL_KEYS.index("tau_drv_obs_rr")],
        ],
        dtype=np.float64,
    )
    tau_brk = np.array(
        [
            control_hist[-1, CONTROL_KEYS.index("tau_brk_obs_fl")],
            control_hist[-1, CONTROL_KEYS.index("tau_brk_obs_fr")],
            control_hist[-1, CONTROL_KEYS.index("tau_brk_obs_rl")],
            control_hist[-1, CONTROL_KEYS.index("tau_brk_obs_rr")],
        ],
        dtype=np.float64,
    )
    omega = np.array(
        [
            state_hist[-1, STATE_KEYS.index("omega_fl")],
            state_hist[-1, STATE_KEYS.index("omega_fr")],
            state_hist[-1, STATE_KEYS.index("omega_rl")],
            state_hist[-1, STATE_KEYS.index("omega_rr")],
        ],
        dtype=np.float64,
    )
    slip_proxy = (radius * omega - vx) / max(vx, 0.5)
    ax_proxy = float(phys_dot[STATE_KEYS.index("vx")])
    ay_proxy = vx * r + float(phys_dot[STATE_KEYS.index("vy")])
    fz_transfer_long = mass * ax_proxy * cg_z / max(wheelbase, 1e-6)
    fz_transfer_lat = mass * ay_proxy * cg_z / max(track, 1e-6)
    force_proxy = (tau_drv - tau_brk) / max(radius, 1e-6)
    return np.concatenate(
        [
            np.asarray(
                [
                    delta_nom,
                    steer_cmd - sw,
                    vx,
                    vy,
                    r,
                    ax_proxy,
                    ay_proxy,
                    fz_transfer_long,
                    fz_transfer_lat,
                    float(np.linalg.norm(force_proxy)),
                    float(np.linalg.norm(slip_proxy)),
                    float(np.tanh(abs(delta_nom) * vx / max(wheelbase, 1e-6))),
                ],
                dtype=np.float64,
            ),
            force_proxy / max(mass, 1.0),
            slip_proxy,
        ]
    )


def _nominal_step(
    state: np.ndarray,
    control: np.ndarray,
    episode: Dict[str, Any],
    dt: float,
) -> np.ndarray:
    ctx = episode["fixed_vehicle_context"]
    prior = episode["nominal_physics_prior"]
    mass = float(prior["mass_nominal"])
    wheelbase = float(ctx["wheelbase"])
    radius = float(ctx["wheel_radius"])
    steering_ratio = float(ctx["steering_layout"]["steering_ratio_nominal"])
    tau_steer = max(float(prior.get("tau_steer_nominal", 0.25)), 0.05)
    next_state = np.asarray(state, dtype=np.float64).copy()
    tau_drv_i = np.array(
        [
            control[CONTROL_KEYS.index("tau_drv_obs_fl")],
            control[CONTROL_KEYS.index("tau_drv_obs_fr")],
            control[CONTROL_KEYS.index("tau_drv_obs_rl")],
            control[CONTROL_KEYS.index("tau_drv_obs_rr")],
        ],
        dtype=np.float64,
    )
    tau_brk_i = np.array(
        [
            control[CONTROL_KEYS.index("tau_brk_obs_fl")],
            control[CONTROL_KEYS.index("tau_brk_obs_fr")],
            control[CONTROL_KEYS.index("tau_brk_obs_rl")],
            control[CONTROL_KEYS.index("tau_brk_obs_rr")],
        ],
        dtype=np.float64,
    )
    tau_drv = float(np.sum(tau_drv_i))
    tau_brk = float(np.sum(tau_brk_i))
    fx = (tau_drv - tau_brk) / max(radius, 1e-6) - 0.012 * mass * 9.81
    ax = np.clip(fx / max(mass, 1.0), -9.0, 5.0)
    sw = float(control[CONTROL_KEYS.index("sw_angle")])
    delta = np.clip(sw / max(steering_ratio, 1e-6), -0.6, 0.6)
    vx = max(0.05, float(state[STATE_KEYS.index("vx")]))
    r_target = vx * math.tan(delta) / max(wheelbase, 1e-6)
    r_prev = float(state[STATE_KEYS.index("r")])
    r_next = r_prev + dt / tau_steer * (r_target - r_prev)
    next_state[STATE_KEYS.index("vx")] = max(0.05, vx + ax * dt)
    next_state[STATE_KEYS.index("vy")] = (
        0.92 * float(state[STATE_KEYS.index("vy")]) + 0.08 * vx * math.sin(delta)
    )
    next_state[STATE_KEYS.index("r")] = r_next
    next_state[STATE_KEYS.index("yaw")] = float(state[STATE_KEYS.index("yaw")]) + r_next * dt
    next_state[STATE_KEYS.index("roll")] = 0.96 * float(state[STATE_KEYS.index("roll")])
    next_state[STATE_KEYS.index("pitch")] = 0.96 * float(state[STATE_KEYS.index("pitch")])
    next_state[STATE_KEYS.index("p")] = (
        next_state[STATE_KEYS.index("roll")] - float(state[STATE_KEYS.index("roll")])
    ) / max(dt, 1e-6)
    next_state[STATE_KEYS.index("q")] = (
        next_state[STATE_KEYS.index("pitch")] - float(state[STATE_KEYS.index("pitch")])
    ) / max(dt, 1e-6)
    wheel_alpha = np.clip((tau_drv_i - tau_brk_i) / 1.25, -4000.0, 4000.0)
    for idx, key in enumerate(["omega_fl", "omega_fr", "omega_rl", "omega_rr"]):
        state_idx = STATE_KEYS.index(key)
        next_state[state_idx] = max(0.0, float(state[state_idx]) + wheel_alpha[idx] * dt)
    return _sanitize_state(next_state)


def _nominal_fz(episode: Dict[str, Any], length: int) -> np.ndarray:
    ctx = episode["fixed_vehicle_context"]
    prior = episode["nominal_physics_prior"]
    mass = float(prior["mass_nominal"])
    wheelbase = float(ctx["wheelbase"])
    cg_x = float(prior["cg_x_nominal"])
    front_static = mass * 9.81 * (wheelbase - cg_x) / max(wheelbase, 1e-6)
    rear_static = mass * 9.81 * cg_x / max(wheelbase, 1e-6)
    row = np.array(
        [front_static / 2.0, front_static / 2.0, rear_static / 2.0, rear_static / 2.0],
        dtype=np.float64,
    )
    return np.tile(row[None, :], (length, 1))


def _context_vector(episode: Dict[str, Any]) -> np.ndarray:
    ctx = episode["fixed_vehicle_context"]
    prior = episode["nominal_physics_prior"]
    drive_type = ctx["drive_layout"]["type"]
    brake_type = ctx["brake_layout"]["type"]
    drive = [
        float(drive_type == "FWD"),
        float(drive_type == "RWD"),
        float(drive_type == "AWD"),
    ]
    brake = [
        float(brake_type == "hydraulic_split"),
        float(brake_type == "brake_by_wire"),
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
    return np.asarray(values + drive + brake, dtype=np.float64)


def _causal_gru_proxy(rows: np.ndarray) -> np.ndarray:
    hidden = np.zeros(rows.shape[1], dtype=np.float64)
    for row in rows:
        hidden = 0.62 * hidden + 0.38 * np.tanh(row)
    return hidden


def _episodes_for_group(
    episodes: List[Dict[str, Any]],
    group: str,
) -> List[Dict[str, Any]]:
    if group == "seen-config":
        return [
            ep
            for ep in episodes
            if ep["metadata"].get("split_role") in ["validation", "test"]
        ]
    if group == "held-out-road":
        return [
            ep
            for ep in episodes
            if ep["metadata"].get("road_type") == "transition"
            and ep["metadata"].get("split_role") in ["test", "held-out", "test-window"]
        ]
    if group == "held-out-vehicle":
        return [ep for ep in episodes if ep["metadata"].get("split_role") == "held-out"]
    return [ep for ep in episodes if ep["metadata"].get("split_role") == group]


def _state_matrix(episode: Dict[str, Any]) -> np.ndarray:
    obs = episode["time_series_observable"]
    return np.column_stack([obs[key] for key in STATE_KEYS]).astype(np.float64)


def _control_matrix(episode: Dict[str, Any]) -> np.ndarray:
    obs = episode["time_series_observable"]
    return np.column_stack([obs[key] for key in CONTROL_KEYS]).astype(np.float64)


def _history_block(arr: np.ndarray, end_idx: int, history_len: int) -> np.ndarray:
    rows = []
    max_idx = len(arr) - 1
    for offset in range(history_len - 1, -1, -1):
        idx = min(max(0, end_idx - offset), max_idx)
        rows.append(arr[idx])
    return np.asarray(rows, dtype=np.float64)


def _sanitize_state(state: np.ndarray) -> np.ndarray:
    out = np.asarray(state, dtype=np.float64).copy()
    out[STATE_KEYS.index("vx")] = max(0.03, out[STATE_KEYS.index("vx")])
    for key in ["omega_fl", "omega_fr", "omega_rl", "omega_rr"]:
        out[STATE_KEYS.index(key)] = max(0.0, out[STATE_KEYS.index(key)])
    for key, limit in [
        ("vy", 80.0),
        ("roll", 2.0),
        ("pitch", 2.0),
        ("yaw", 200.0),
        ("p", 20.0),
        ("q", 20.0),
        ("r", 20.0),
    ]:
        idx = STATE_KEYS.index(key)
        out[idx] = float(np.clip(out[idx], -limit, limit))
    return out


def _residual_dot_bounds(hybrid_cfg: Dict[str, Any]) -> np.ndarray:
    defaults = {
        "vx": 30.0,
        "vy": 30.0,
        "roll": 4.0,
        "pitch": 4.0,
        "yaw": 8.0,
        "p": 20.0,
        "q": 20.0,
        "r": 20.0,
        "omega_fl": 800.0,
        "omega_fr": 800.0,
        "omega_rl": 800.0,
        "omega_rr": 800.0,
    }
    defaults.update(hybrid_cfg.get("residual_dot_bounds", {}))
    return np.asarray([float(defaults[key]) for key in STATE_KEYS], dtype=np.float64)


def _summary_metrics(
    seed_reports: List[Dict[str, Any]],
    hybrid_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for group, prefix in [
        ("validation", "validation"),
        ("test", "seen_config"),
        ("seen-config", "seen_config_combined"),
        ("held-out-road", "held_out_road"),
        ("held-out-vehicle", "held_out_vehicle"),
        ("test-window", "test_window"),
    ]:
        values = _collect_horizon_metric(seed_reports, group, "100", "base_vs_physics_ratio")
        if values:
            metrics["%s_base_vs_physics_ratio_h100_mean" % prefix] = float(np.mean(values))
            metrics["%s_base_vs_physics_ratio_h100_std" % prefix] = float(np.std(values))
        bb_values = _collect_horizon_metric(seed_reports, group, "100", "base_vs_black_box_ratio")
        if bb_values:
            metrics["%s_base_vs_black_box_ratio_h100_mean" % prefix] = float(np.mean(bb_values))
    metrics["base_train_transition_count"] = int(
        min((r.get("train_transition_count", 0) for r in seed_reports), default=0)
    )
    residual_ratios = _collect_horizon_metric(
        seed_reports, "validation", "100", "residual_to_physics_ratio"
    )
    residual_smooth = _collect_horizon_metric(
        seed_reports, "validation", "100", "residual_dot_smoothness"
    )
    constraint_rates = _collect_horizon_metric(
        seed_reports, "validation", "100", "base_constraint_violation_rate"
    )
    metrics["base_residual_to_physics_ratio_mean"] = _mean(residual_ratios)
    metrics["base_residual_dot_smoothness_mean"] = _mean(residual_smooth)
    metrics["base_constraint_violation_rate_mean"] = _mean(constraint_rates)
    metrics["base_hybrid_training_passed"] = int(
        metrics.get("base_train_transition_count", 0) > 0
        and metrics.get("validation_base_vs_physics_ratio_h100_mean", 9.0) < 0.98
    )
    metrics["base_seen_config_eval_passed"] = int(
        metrics.get("seen_config_combined_base_vs_physics_ratio_h100_mean", 9.0) < 1.02
    )
    metrics["base_held_out_road_eval_passed"] = int(
        metrics.get("held_out_road_base_vs_physics_ratio_h100_mean", 9.0) < 1.08
    )
    metrics["base_held_out_vehicle_eval_passed"] = int(
        metrics.get("held_out_vehicle_base_vs_physics_ratio_h100_mean", 9.0) < 1.10
    )
    metrics["base_residual_constraint_audit_passed"] = int(
        metrics.get("base_constraint_violation_rate_mean", 1.0) <= 0.01
        and metrics.get("base_residual_to_physics_ratio_mean", 99.0) <= 8.0
        and metrics.get("base_residual_dot_smoothness_mean", 99.0) <= 250.0
    )
    seed_values = _collect_horizon_metric(
        seed_reports, "validation", "100", "base_vs_physics_ratio"
    )
    metrics["base_seed_replication_passed"] = int(
        len(seed_reports) >= int(hybrid_cfg.get("min_seed_count", 3))
        and len(seed_values) >= int(hybrid_cfg.get("min_seed_count", 3))
        and float(np.std(seed_values)) <= max(0.08, 0.25 * max(float(np.mean(seed_values)), 1e-9))
    )
    return metrics


def _required_success_keys(hybrid_cfg: Dict[str, Any]) -> List[str]:
    profile = hybrid_cfg.get("success_profile", "base_training")
    mapping = {
        "base_training": ["base_hybrid_training_passed"],
        "seen_config": ["base_seen_config_eval_passed"],
        "held_out_road": ["base_held_out_road_eval_passed"],
        "held_out_vehicle": ["base_held_out_vehicle_eval_passed"],
        "residual_constraint_audit": ["base_residual_constraint_audit_passed"],
        "seed_replication": ["base_seed_replication_passed"],
        "full_m5": [
            "base_hybrid_training_passed",
            "base_seen_config_eval_passed",
            "base_held_out_road_eval_passed",
            "base_held_out_vehicle_eval_passed",
            "base_residual_constraint_audit_passed",
        ],
    }
    return mapping.get(profile, ["base_hybrid_training_passed"])


def _collect_horizon_metric(
    seed_reports: List[Dict[str, Any]],
    group: str,
    horizon: str,
    metric: str,
) -> List[float]:
    values = []
    for report in seed_reports:
        value = (
            report.get("groups", {})
            .get(group, {})
            .get("horizons", {})
            .get(horizon, {})
            .get(metric)
        )
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return values


def _merge_diagnostics(items: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({key for item in items for key in item})
    return {key: _mean([item[key] for item in items if key in item]) for key in keys}


def _constraint_violation_rate(states: np.ndarray) -> float:
    if len(states) == 0:
        return 1.0
    violations = np.zeros(len(states), dtype=bool)
    violations |= ~np.all(np.isfinite(states), axis=1)
    violations |= states[:, STATE_KEYS.index("vx")] < 0.0
    violations |= np.abs(states[:, STATE_KEYS.index("roll")]) > 1.5
    violations |= np.abs(states[:, STATE_KEYS.index("pitch")]) > 1.5
    violations |= np.abs(states[:, STATE_KEYS.index("r")]) > 10.0
    for key in ["omega_fl", "omega_fr", "omega_rl", "omega_rr"]:
        violations |= states[:, STATE_KEYS.index(key)] < -1e-6
    return float(np.mean(violations))


def _is_stable(states: np.ndarray) -> bool:
    return bool(
        len(states) > 1
        and np.all(np.isfinite(states))
        and _constraint_violation_rate(states) <= 0.05
    )


def _stack_or_empty(rows: List[np.ndarray], width: int = 1) -> np.ndarray:
    if not rows:
        return np.zeros((0, width), dtype=np.float64)
    return np.vstack(rows).astype(np.float64)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.nanpercentile(np.asarray(values, dtype=np.float64), q))


def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _write_hybrid_markdown(report: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    summary = report.get("summary_metrics", {})
    lines = [
        "# B3 Base Hybrid Scaffold Report",
        "",
        "This report is produced by the M5 numpy scaffold for `Base = E1 + T1 + F1 + S1 + M1a + V1 + U0`.",
        "",
        "## Success Gates",
        "",
        "| Gate | Value |",
        "|---|---:|",
    ]
    for key in [
        "base_hybrid_training_passed",
        "base_seen_config_eval_passed",
        "base_held_out_road_eval_passed",
        "base_held_out_vehicle_eval_passed",
        "base_residual_constraint_audit_passed",
        "base_seed_replication_passed",
    ]:
        lines.append("| %s | %s |" % (key, summary.get(key, "n/a")))
    lines.extend(["", "## 100-step RMSE Ratios", "", "| Group | Base / Physics | Base / Black-box |", "|---|---:|---:|"])
    for group, prefix in [
        ("validation", "validation"),
        ("seen-config combined", "seen_config_combined"),
        ("test split", "seen_config"),
        ("held-out-road", "held_out_road"),
        ("held-out-vehicle", "held_out_vehicle"),
        ("test-window", "test_window"),
    ]:
        lines.append(
            "| %s | %.6f | %.6f |"
            % (
                group,
                float(summary.get("%s_base_vs_physics_ratio_h100_mean" % prefix, 0.0)),
                float(summary.get("%s_base_vs_black_box_ratio_h100_mean" % prefix, 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Residual Audit",
            "",
            "| Metric | Value |",
            "|---|---:|",
            "| residual_to_physics_ratio_mean | %.6f |"
            % float(summary.get("base_residual_to_physics_ratio_mean", 0.0)),
            "| residual_dot_smoothness_mean | %.6f |"
            % float(summary.get("base_residual_dot_smoothness_mean", 0.0)),
            "| constraint_violation_rate_mean | %.6f |"
            % float(summary.get("base_constraint_violation_rate_mean", 0.0)),
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
