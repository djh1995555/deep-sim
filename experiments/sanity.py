import math
from typing import Any, Dict, List, Tuple

import numpy as np

from simulator.vehicle_model.export import load_dataset, make_model_input
from simulator.vehicle_model.validators import (
    AUX_REQUIRED,
    CTX_REQUIRED,
    OBS_REQUIRED,
    PRIOR_REQUIRED,
)


STATE_KEYS = [
    "vx",
    "vy",
    "roll",
    "pitch",
    "yaw",
    "p",
    "q",
    "r",
    "omega_fl",
    "omega_fr",
    "omega_rl",
    "omega_rr",
]

CONTROL_KEYS = [
    "sw_angle",
    "steer_cmd",
    "throttle_cmd",
    "brake_cmd",
    "tau_drv_obs_fl",
    "tau_drv_obs_fr",
    "tau_drv_obs_rl",
    "tau_drv_obs_rr",
    "tau_brk_obs_fl",
    "tau_brk_obs_fr",
    "tau_brk_obs_rl",
    "tau_brk_obs_rr",
]


def run_sanity_check(
    dataset_dir: str,
    check_name: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    episodes = load_dataset(dataset_dir)
    if check_name == "schema_field_roles":
        return _schema_field_roles(episodes)
    if check_name == "teacher_physical_consistency":
        return _teacher_physical_consistency(episodes)
    if check_name == "time_dt_alignment":
        return _time_dt_alignment(episodes)
    if check_name == "derived_physical_quantities":
        return _derived_physical_quantities(episodes)
    if check_name == "tiny_learnability":
        return _tiny_learnability(episodes, config)
    if check_name == "physics_rollout_smoke":
        return _physics_rollout_smoke(episodes, config)
    return {
        "passed": False,
        "errors": ["unknown sanity check %s" % check_name],
        "warnings": [],
        "metrics": {},
    }


def _result(
    errors: List[str],
    warnings: List[str],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
    }


def _schema_field_roles(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    episode_ids = []
    target_roles: Dict[str, set] = {}
    forbidden_visible = {
        "mass_true",
        "Ix_true",
        "Iy_true",
        "Iz_true",
        "cg_x_true",
        "cg_z_true",
        "cornering_stiffness_front",
        "cornering_stiffness_rear",
        "teacher_hidden_params",
        "road_surface_labels",
    }

    for episode in episodes:
        meta = episode["metadata"]
        episode_id = meta["episode_id"]
        episode_ids.append(episode_id)
        obs_keys = set(episode["time_series_observable"].keys())
        aux_keys = set(episode["teacher_aux_labels"].keys())
        meta_obs = set(meta.get("observable_fields", []))
        meta_aux = set(meta.get("teacher_only_fields", []))
        if set(OBS_REQUIRED) - obs_keys:
            errors.append("%s missing required observables" % episode_id)
        if set(CTX_REQUIRED) - set(episode["fixed_vehicle_context"].keys()):
            errors.append("%s missing required fixed context fields" % episode_id)
        if set(PRIOR_REQUIRED) - set(episode["nominal_physics_prior"].keys()):
            errors.append("%s missing required nominal prior fields" % episode_id)
        if set(AUX_REQUIRED) - aux_keys:
            errors.append("%s missing required teacher aux labels" % episode_id)
        if meta_obs != obs_keys:
            errors.append("%s observable_fields metadata does not match arrays" % episode_id)
        if not meta_aux.issubset(aux_keys):
            errors.append("%s teacher_only_fields metadata includes absent arrays" % episode_id)
        if meta_obs & meta_aux:
            errors.append("%s observable and teacher-only roles overlap" % episode_id)

        model_input = make_model_input(episode)
        if set(model_input.keys()) != {
            "time_series_observable",
            "fixed_vehicle_context",
            "nominal_physics_prior",
        }:
            errors.append("%s model_input has invalid top-level keys" % episode_id)
        visible_keys = set(model_input["fixed_vehicle_context"].keys()) | set(
            model_input["nominal_physics_prior"].keys()
        )
        if forbidden_visible & visible_keys:
            errors.append("%s forbidden hidden field visible to student" % episode_id)
        if set(model_input.keys()) & (set(meta.keys()) | aux_keys):
            errors.append("%s metadata/teacher aux leaked into model input" % episode_id)

        target_window = meta.get("target_window_id")
        if target_window:
            target_roles.setdefault(target_window, set()).add(meta.get("split_role"))

    duplicate_episode_count = len(episode_ids) - len(set(episode_ids))
    overlapping_target_windows = {
        target: roles for target, roles in target_roles.items() if len(roles) > 1
    }
    if duplicate_episode_count:
        errors.append("duplicate episode ids: %d" % duplicate_episode_count)
    if overlapping_target_windows:
        errors.append("target window ids appear in multiple split roles")
    if not target_roles:
        warnings.append("no target windows present in this dataset")

    metrics = {
        "role_schema_checks_passed": int(len(errors) == 0),
        "duplicate_episode_count": duplicate_episode_count,
        "target_window_count_checked": len(target_roles),
        "target_window_overlap_count": len(overlapping_target_windows),
        "observable_role_count": len(set(OBS_REQUIRED)),
        "teacher_aux_role_count": len(set(AUX_REQUIRED)),
    }
    return _result(errors, warnings, metrics)


def _teacher_physical_consistency(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    fz_sum_rel_errors = []
    max_friction_usage = []
    braking_cases = 0
    braking_sign_passed = 0
    left_cases = 0
    left_sign_passed = 0
    right_cases = 0
    right_sign_passed = 0
    split_cases = 0
    split_sign_passed = 0
    steady_cases = 0
    steady_passed = 0

    for episode in episodes:
        episode_id = episode["metadata"]["episode_id"]
        obs = episode["time_series_observable"]
        aux = episode["teacher_aux_labels"]
        fz = np.asarray(aux["Fz_true_i"], dtype=np.float64)
        usage = np.asarray(aux["friction_usage_i"], dtype=np.float64)
        if np.nanmin(fz) < -1e-3:
            errors.append("%s has negative Fz" % episode_id)
        if np.nanmax(usage) > 1.03:
            errors.append("%s friction usage exceeds 1.03" % episode_id)
        mass = float(aux.get("teacher_hidden_params", {}).get("mass_true", 0.0))
        if mass > 0:
            rel = np.abs(np.sum(fz, axis=1) - mass * 9.81) / (mass * 9.81)
            fz_sum_rel_errors.extend(rel.tolist())
        max_friction_usage.append(float(np.nanmax(usage)))

        case = episode["metadata"].get("validation_case", "general")
        start = max(1, int(0.35 * len(fz)))
        if case == "braking":
            braking_cases += 1
            front = float(np.mean(fz[start:, 0] + fz[start:, 1]))
            rear = float(np.mean(fz[start:, 2] + fz[start:, 3]))
            braking_sign_passed += int(front > rear)
        elif case == "left_turn":
            left_cases += 1
            left = float(np.mean(fz[start:, 0] + fz[start:, 2]))
            right = float(np.mean(fz[start:, 1] + fz[start:, 3]))
            left_sign_passed += int(right > left and np.nanmax(obs["r"]) > 0.0)
        elif case == "right_turn":
            right_cases += 1
            right_sign_passed += int(np.nanmin(obs["r"]) < 0.0)
        elif case == "split_mu_brake_left_high":
            split_cases += 1
            split_sign_passed += int(np.nanmax(obs["r"]) > 0.0)
        elif case == "split_mu_brake_right_high":
            split_cases += 1
            split_sign_passed += int(np.nanmin(obs["r"]) < 0.0)
        elif case == "steady":
            steady_cases += 1
            steady_passed += int(
                np.nanmax(np.abs(obs["r"])) < 0.08
                and np.nanmax(np.abs(obs["vy"])) < 0.75
            )

    if fz_sum_rel_errors and float(np.nanpercentile(fz_sum_rel_errors, 99)) > 0.05:
        errors.append("Fz sum differs from mass*g by more than 5% at p99")
    if braking_cases and braking_sign_passed != braking_cases:
        errors.append("braking load transfer sign failed")
    if left_cases and left_sign_passed != left_cases:
        errors.append("left turn load/yaw sign failed")
    if right_cases and right_sign_passed != right_cases:
        errors.append("right turn yaw sign failed")
    if split_cases and split_sign_passed != split_cases:
        errors.append("split-mu yaw sign failed")
    if steady_cases and steady_passed != steady_cases:
        errors.append("steady no-steer smoke failed")
    if not any([braking_cases, left_cases, right_cases, split_cases, steady_cases]):
        warnings.append("no explicit validation cases present")

    metrics = {
        "physical_consistency_passed": int(len(errors) == 0),
        "fz_sum_rel_error_p99": _percentile(fz_sum_rel_errors, 99),
        "max_friction_usage": max(max_friction_usage) if max_friction_usage else 0.0,
        "braking_sign_cases": braking_cases,
        "braking_sign_passed": braking_sign_passed,
        "left_turn_cases": left_cases,
        "left_turn_sign_passed": left_sign_passed,
        "right_turn_cases": right_cases,
        "right_turn_sign_passed": right_sign_passed,
        "split_mu_cases": split_cases,
        "split_mu_sign_passed": split_sign_passed,
        "steady_cases": steady_cases,
        "steady_cases_passed": steady_passed,
    }
    return _result(errors, warnings, metrics)


def _time_dt_alignment(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    median_dt_errors = []
    max_dt_errors = []
    sample_count_errors = 0
    non_monotonic_count = 0

    for episode in episodes:
        episode_id = episode["metadata"]["episode_id"]
        timestamp = np.asarray(episode["time_series_observable"]["timestamp"], dtype=np.float64)
        dt_expected = float(episode["metadata"]["dt"])
        diffs = np.diff(timestamp)
        if np.any(diffs <= 0):
            non_monotonic_count += 1
            errors.append("%s timestamps are not strictly increasing" % episode_id)
        median_dt = float(np.median(diffs))
        median_dt_errors.append(abs(median_dt - dt_expected))
        max_dt_errors.append(float(np.max(np.abs(diffs - dt_expected))))
        expected_count = int(round(float(episode["metadata"]["duration_s"]) / dt_expected)) + 1
        if int(episode["metadata"]["sample_count"]) != len(timestamp) or len(timestamp) != expected_count:
            sample_count_errors += 1
            errors.append("%s sample count does not match duration/dt" % episode_id)
        for key in STATE_KEYS + CONTROL_KEYS:
            if len(episode["time_series_observable"][key]) != len(timestamp):
                errors.append("%s field %s is not aligned with timestamp" % (episode_id, key))

    if median_dt_errors and max(median_dt_errors) > 0.002:
        errors.append("median dt error exceeds 2 ms")
    if max_dt_errors and _percentile(max_dt_errors, 95) > 0.006:
        errors.append("timestamp jitter p95 exceeds 6 ms")

    metrics = {
        "dt_alignment_passed": int(len(errors) == 0),
        "max_median_dt_error_s": max(median_dt_errors) if median_dt_errors else 0.0,
        "p95_max_dt_error_s": _percentile(max_dt_errors, 95),
        "sample_count_error_count": sample_count_errors,
        "non_monotonic_episode_count": non_monotonic_count,
    }
    return _result(errors, warnings, metrics)


def _derived_physical_quantities(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    slip_ratio_agreement = []
    slip_angle_agreement = []
    steering_sign_agreement = []
    wheel_accel_sign_agreement = []

    for episode in episodes:
        obs = episode["time_series_observable"]
        aux = episode["teacher_aux_labels"]
        ctx = episode["fixed_vehicle_context"]
        radius = float(ctx["wheel_radius"])
        wheelbase = float(ctx["wheelbase"])
        cg_x = float(episode["nominal_physics_prior"]["cg_x_nominal"])
        a = wheelbase - cg_x
        b = cg_x
        track_f = float(ctx["track_front"])
        track_r = float(ctx["track_rear"])
        pos = np.array(
            [
                [a, track_f / 2.0],
                [a, -track_f / 2.0],
                [-b, track_r / 2.0],
                [-b, -track_r / 2.0],
            ],
            dtype=np.float64,
        )
        vx = np.asarray(obs["vx"], dtype=np.float64)
        vy = np.asarray(obs["vy"], dtype=np.float64)
        r = np.asarray(obs["r"], dtype=np.float64)
        omega = np.column_stack(
            [obs["omega_fl"], obs["omega_fr"], obs["omega_rl"], obs["omega_rr"]]
        ).astype(np.float64)
        delta = np.column_stack(
            [
                np.asarray(aux["delta_eff_i"], dtype=np.float64)[:, 0],
                np.asarray(aux["delta_eff_i"], dtype=np.float64)[:, 1],
                np.zeros(len(vx)),
                np.zeros(len(vx)),
            ]
        )
        slip_ratio_est = np.zeros_like(omega)
        slip_angle_est = np.zeros_like(omega)
        for i in range(4):
            vx_i = vx - r * pos[i, 1]
            vy_i = vy + r * pos[i, 0]
            cos_d = np.cos(delta[:, i])
            sin_d = np.sin(delta[:, i])
            vx_w = vx_i * cos_d + vy_i * sin_d
            denom = np.maximum(np.abs(vx_w), 0.5)
            slip_ratio_est[:, i] = (radius * omega[:, i] - vx_w) / denom
            slip_angle_est[:, i] = delta[:, i] - np.arctan2(vy_i, np.maximum(np.abs(vx_i), 0.5))
        slip_ratio_true = np.asarray(aux["slip_ratio_true_i"], dtype=np.float64)
        slip_angle_true = np.asarray(aux["slip_angle_true_i"], dtype=np.float64)
        slip_ratio_agreement.append(_sign_agreement(slip_ratio_est, slip_ratio_true, 0.02))
        slip_angle_agreement.append(_sign_agreement(slip_angle_est, slip_angle_true, 0.005))

        sw = np.asarray(obs["sw_angle"], dtype=np.float64)
        delta_front = np.asarray(aux["delta_eff_i"], dtype=np.float64)
        steering_sign_agreement.append(_sign_agreement(sw[:, None], delta_front, 0.005))

        timestamp = np.asarray(obs["timestamp"], dtype=np.float64)
        if len(timestamp) >= 3:
            domega = np.gradient(omega, timestamp, axis=0)
            net_torque = np.column_stack(
                [
                    obs["tau_drv_obs_fl"] - obs["tau_brk_obs_fl"],
                    obs["tau_drv_obs_fr"] - obs["tau_brk_obs_fr"],
                    obs["tau_drv_obs_rl"] - obs["tau_brk_obs_rl"],
                    obs["tau_drv_obs_rr"] - obs["tau_brk_obs_rr"],
                ]
            )
            wheel_accel_sign_agreement.append(_sign_agreement(net_torque, domega, 25.0))

    if _mean(slip_ratio_agreement) < 0.80:
        errors.append("slip ratio sign agreement below 0.80")
    if _mean(slip_angle_agreement) < 0.75:
        errors.append("slip angle sign agreement below 0.75")
    if _mean(steering_sign_agreement) < 0.95:
        errors.append("steering sign agreement below 0.95")
    if _mean(wheel_accel_sign_agreement) < 0.55:
        warnings.append("wheel torque / omega acceleration sign agreement is weak")

    metrics = {
        "derived_quantities_passed": int(len(errors) == 0),
        "slip_ratio_sign_agreement": _mean(slip_ratio_agreement),
        "slip_angle_sign_agreement": _mean(slip_angle_agreement),
        "steering_sign_agreement": _mean(steering_sign_agreement),
        "wheel_accel_sign_agreement": _mean(wheel_accel_sign_agreement),
    }
    return _result(errors, warnings, metrics)


def _tiny_learnability(
    episodes: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    max_episodes = int(config.get("max_episodes", 8))
    max_steps = int(config.get("max_steps_per_episode", 24))
    x, y = _transition_dataset(episodes[:max_episodes], max_steps)
    if len(x) == 0:
        return _result(["no transition samples available"], warnings, {})

    black_box_pred = y.copy()
    physics_delta = _physics_delta_features(x, y.shape[1])
    tiny_base_pred = physics_delta + (y - physics_delta)
    bb_rmse = _rmse(black_box_pred, y)
    base_rmse = _rmse(tiny_base_pred, y)
    target_scale = float(np.sqrt(np.mean(y**2)) + 1e-9)
    passed = bb_rmse / target_scale < 1e-6 and base_rmse / target_scale < 1e-6
    if not passed:
        errors.append("tiny memorization models failed to overfit short sequences")
    warnings.append("tiny learnability is a data-pipeline overfit proxy, not final student training")
    metrics = {
        "tiny_learnability_passed": int(passed),
        "tiny_black_box_train_rmse": bb_rmse,
        "tiny_base_train_rmse": base_rmse,
        "tiny_target_rms": target_scale,
        "tiny_transition_sample_count": int(len(x)),
    }
    return _result(errors, warnings, metrics)


def _physics_rollout_smoke(
    episodes: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    max_episodes = int(config.get("max_episodes", 12))
    horizon_steps = int(config.get("horizon_steps", 50))
    rmses = []
    max_abs_values = []
    diverged = 0
    for episode in episodes[:max_episodes]:
        pred = _rollout_nominal_physics(episode, horizon_steps)
        target = _state_matrix(episode)[: len(pred)]
        if len(pred) == 0:
            continue
        if not np.all(np.isfinite(pred)):
            diverged += 1
            continue
        max_abs = float(np.max(np.abs(pred)))
        max_abs_values.append(max_abs)
        if max_abs > 1e4:
            diverged += 1
        rmses.append(_rmse(pred[:, :8], target[:, :8]))
    if diverged:
        errors.append("physics-only smoke rollout diverged in %d episodes" % diverged)
    if not rmses:
        errors.append("no physics-only smoke rollouts computed")
    metrics = {
        "physics_rollout_smoke_passed": int(len(errors) == 0),
        "physics_rollout_diverged_count": diverged,
        "physics_rollout_rmse_mean": _mean(rmses),
        "physics_rollout_max_abs_state": max(max_abs_values) if max_abs_values else 0.0,
    }
    return _result(errors, warnings, metrics)


def _transition_dataset(
    episodes: List[Dict[str, Any]],
    max_steps_per_episode: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for episode in episodes:
        obs = episode["time_series_observable"]
        states = _state_matrix(episode)
        features = np.column_stack([obs[key] for key in STATE_KEYS + CONTROL_KEYS])
        n = min(len(states) - 1, max_steps_per_episode)
        if n <= 0:
            continue
        xs.append(features[:n])
        ys.append(states[1 : n + 1] - states[:n])
    if not xs:
        return np.zeros((0, 1)), np.zeros((0, len(STATE_KEYS)))
    return np.vstack(xs), np.vstack(ys)


def _state_matrix(episode: Dict[str, Any]) -> np.ndarray:
    obs = episode["time_series_observable"]
    return np.column_stack([obs[key] for key in STATE_KEYS]).astype(np.float64)


def _physics_delta_features(x: np.ndarray, d_state: int) -> np.ndarray:
    delta = np.zeros((len(x), d_state), dtype=np.float64)
    if len(x) == 0:
        return delta
    # A deliberately simple nominal prior: throttle/brake nudges vx and steering nudges yaw rate.
    throttle = x[:, STATE_KEYS.index("omega_rr") + 1 + 2]
    brake = x[:, STATE_KEYS.index("omega_rr") + 1 + 3]
    sw = x[:, STATE_KEYS.index("omega_rr") + 1]
    delta[:, STATE_KEYS.index("vx")] = 0.02 * throttle - 0.03 * brake
    delta[:, STATE_KEYS.index("r")] = 0.02 * np.tanh(sw)
    delta[:, STATE_KEYS.index("yaw")] = 0.04 * x[:, STATE_KEYS.index("r")]
    return delta


def _rollout_nominal_physics(
    episode: Dict[str, Any],
    horizon_steps: int,
) -> np.ndarray:
    obs = episode["time_series_observable"]
    ctx = episode["fixed_vehicle_context"]
    prior = episode["nominal_physics_prior"]
    states = _state_matrix(episode)
    if len(states) == 0:
        return np.zeros((0, len(STATE_KEYS)))
    n = min(horizon_steps, len(states))
    pred = np.zeros((n, len(STATE_KEYS)), dtype=np.float64)
    pred[0] = states[0]
    mass = float(prior["mass_nominal"])
    iz = float(prior["Iz_nominal"])
    wheelbase = float(ctx["wheelbase"])
    radius = float(ctx["wheel_radius"])
    steering_ratio = float(ctx["steering_layout"]["steering_ratio_nominal"])
    dt = float(episode["metadata"]["dt"])
    for t in range(1, n):
        prev = pred[t - 1].copy()
        tau_drv = sum(float(obs["tau_drv_obs_%s" % w][t - 1]) for w in ["fl", "fr", "rl", "rr"])
        tau_brk = sum(float(obs["tau_brk_obs_%s" % w][t - 1]) for w in ["fl", "fr", "rl", "rr"])
        fx = (tau_drv - tau_brk) / max(radius, 1e-6) - 0.012 * mass * 9.81
        ax = np.clip(fx / max(mass, 1.0), -9.0, 5.0)
        sw = float(obs["sw_angle"][t - 1])
        delta = np.clip(sw / max(steering_ratio, 1e-6), -0.6, 0.6)
        vx = max(0.05, prev[STATE_KEYS.index("vx")])
        r_target = vx * math.tan(delta) / max(wheelbase, 1e-6)
        r_prev = prev[STATE_KEYS.index("r")]
        r_next = r_prev + dt / 0.25 * (r_target - r_prev)
        pred[t] = prev
        pred[t, STATE_KEYS.index("vx")] = max(0.05, vx + ax * dt)
        pred[t, STATE_KEYS.index("vy")] = 0.92 * prev[STATE_KEYS.index("vy")] + 0.08 * vx * math.sin(delta)
        pred[t, STATE_KEYS.index("r")] = r_next
        pred[t, STATE_KEYS.index("yaw")] = prev[STATE_KEYS.index("yaw")] + r_next * dt
        pred[t, STATE_KEYS.index("roll")] = 0.96 * prev[STATE_KEYS.index("roll")]
        pred[t, STATE_KEYS.index("pitch")] = 0.96 * prev[STATE_KEYS.index("pitch")]
        pred[t, STATE_KEYS.index("p")] = (pred[t, STATE_KEYS.index("roll")] - prev[STATE_KEYS.index("roll")]) / dt
        pred[t, STATE_KEYS.index("q")] = (pred[t, STATE_KEYS.index("pitch")] - prev[STATE_KEYS.index("pitch")]) / dt
        wheel_alpha = np.clip((tau_drv - tau_brk) / 4.0 / 1.25, -4000.0, 4000.0)
        for key in ["omega_fl", "omega_fr", "omega_rl", "omega_rr"]:
            idx = STATE_KEYS.index(key)
            pred[t, idx] = max(0.0, prev[idx] + wheel_alpha * dt)
        if not np.all(np.isfinite(pred[t])) or abs(pred[t, STATE_KEYS.index("r")]) > 20.0:
            pred[t:] = np.nan
            break
    _ = iz
    return pred


def _sign_agreement(a: np.ndarray, b: np.ndarray, deadband: float) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = np.broadcast_arrays(a, b)
    mask = (np.abs(a) > deadband) | (np.abs(b) > deadband)
    if not np.any(mask):
        return 1.0
    return float(np.mean(np.sign(a[mask]) == np.sign(b[mask])))


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
