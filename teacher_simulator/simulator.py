from typing import Any, Dict, List

import numpy as np

from teacher_simulator.config import TeacherSimConfig
from teacher_simulator.modules.aero import AeroModel
from teacher_simulator.modules.drive_brake import DriveBrakeActuator
from teacher_simulator.modules.road import RoadModel
from teacher_simulator.modules.sensors import SensorModel
from teacher_simulator.modules.steering import SteeringActuator
from teacher_simulator.modules.suspension import SuspensionModel
from teacher_simulator.modules.tire import TireModel
from teacher_simulator.scenario import ScenarioConfig
from teacher_simulator.state import TeacherState
from teacher_simulator.vehicle_params import vehicle_internal_hash


class TeacherSimulator:
    def __init__(self, config: TeacherSimConfig) -> None:
        config.validate()
        self.config = config
        self.road = RoadModel()
        self.steering = SteeringActuator()
        self.drive_brake = DriveBrakeActuator()
        self.suspension = SuspensionModel()
        self.tire = TireModel()
        self.aero = AeroModel()

    def run_episode(self, scenario: ScenarioConfig) -> Dict[str, Any]:
        vehicle = scenario.vehicle_config
        state = TeacherState(vx=scenario.control_script.initial_speed_mps)
        state.omega = np.full(
            4, state.vx / max(vehicle.wheel_radius, 1e-6), dtype=np.float64
        )
        rng = np.random.default_rng(self.config.seed + scenario.seed)
        sensors = SensorModel(rng)

        obs_rows: List[Dict[str, float]] = []
        aux_rows: Dict[str, List[Any]] = {}
        road_labels: List[List[str]] = []
        n_steps = int(round(self.config.duration_s / self.config.dt_internal))
        export_stride = int(round(self.config.dt_export / self.config.dt_internal))
        last_tire = None
        last_load = None
        last_torques = None
        last_steering = None
        last_road = None
        last_aero = None

        for step in range(n_steps + 1):
            t = step * self.config.dt_internal
            command = scenario.control_script.command_at(t)
            steering = self.steering.step(
                command["sw_angle"],
                state,
                vehicle,
                scenario.actuator_profile,
                self.config.dt_internal,
            )
            torques = self.drive_brake.step(
                command["throttle_cmd"],
                command["brake_cmd"],
                state,
                vehicle,
                scenario.actuator_profile,
                self.config.dt_internal,
            )
            road_contact = self.road.query(t, scenario.road_profile)
            load_state = self.suspension.step(
                state.prev_ax,
                state.prev_ay,
                road_contact,
                vehicle,
                self.config.gravity,
            )
            tire = self.tire.step(
                state,
                steering,
                torques,
                road_contact,
                load_state,
                vehicle,
                self.config.numerical_limits.min_speed_for_slip,
            )
            aero = self.aero.step(state, scenario.environment_profile, vehicle)
            self._integrate_chassis_and_wheels(
                state, tire, torques, aero, vehicle, self.config.dt_internal
            )

            last_tire = tire
            last_load = load_state
            last_torques = torques
            last_steering = steering
            last_road = road_contact
            last_aero = aero

            if step % export_stride == 0:
                obs = sensors.observe(
                    t,
                    state,
                    command,
                    torques,
                    vehicle,
                    scenario.sensor_profile,
                )
                obs_rows.append(obs)
                self._append_aux(
                    aux_rows,
                    tire,
                    load_state,
                    torques,
                    steering,
                    road_contact,
                    aero,
                )
                road_labels.append(list(road_contact["labels"]))

        metadata = self._metadata(
            scenario,
            len(obs_rows),
            vehicle_internal_hash(vehicle.hidden_params),
        )
        metadata["final_state_preview"] = {
            "vx": float(state.vx),
            "vy": float(state.vy),
            "yaw": float(state.yaw),
            "r": float(state.r),
        }
        metadata["last_modules_present"] = all(
            x is not None
            for x in [last_tire, last_load, last_torques, last_steering, last_road, last_aero]
        )
        episode = {
            "metadata": metadata,
            "fixed_vehicle_context": vehicle.fixed_vehicle_context,
            "nominal_physics_prior": vehicle.nominal_physics_prior,
            "time_series_observable": self._stack_obs(obs_rows),
            "teacher_aux_labels": self._stack_aux(
                aux_rows, road_labels, vehicle.hidden_params
            ),
        }
        return episode

    def _integrate_chassis_and_wheels(
        self,
        state: TeacherState,
        tire: Dict[str, np.ndarray],
        torques: Dict[str, np.ndarray],
        aero: Dict[str, np.ndarray],
        vehicle,
        dt: float,
    ) -> None:
        fx_total = float(np.sum(tire["Fx_true_i"]) + aero["force_moment"][0])
        fy_total = float(np.sum(tire["Fy_true_i"]) + aero["force_moment"][1])
        mz_total = float(np.sum(tire["Mz_true_i"]) + aero["force_moment"][5])
        rolling = (
            vehicle.hidden_params["rolling_resistance"]
            * vehicle.mass
            * self.config.gravity
            * np.sign(max(state.vx, 1e-6))
        )
        fx_total -= rolling

        ax_force = fx_total / vehicle.mass
        ay_force = fy_total / vehicle.mass
        ax = ax_force + state.vy * state.r
        ay = ay_force - state.vx * state.r
        rdot = mz_total / vehicle.iz

        state.vx = max(0.05, state.vx + ax * dt)
        state.vy = state.vy + ay * dt
        state.r = state.r + rdot * dt
        state.yaw = state.yaw + state.r * dt
        state.x_world += (state.vx * np.cos(state.yaw) - state.vy * np.sin(state.yaw)) * dt
        state.y_world += (state.vx * np.sin(state.yaw) + state.vy * np.cos(state.yaw)) * dt

        roll_target = -np.clip(ay_force / self.config.gravity, -0.6, 0.6) * 0.08
        pitch_target = np.clip(ax_force / self.config.gravity, -0.6, 0.6) * 0.06
        roll_prev = state.roll
        pitch_prev = state.pitch
        state.roll += dt / 0.22 * (roll_target - state.roll)
        state.pitch += dt / 0.18 * (pitch_target - state.pitch)
        state.p = (state.roll - roll_prev) / dt
        state.q = (state.pitch - pitch_prev) / dt
        state.prev_ax = ax_force
        state.prev_ay = ay_force

        radius = vehicle.wheel_radius
        wheel_inertia = vehicle.hidden_params["wheel_inertia"]
        wheel_alpha = (
            torques["tau_drv_true_i"]
            - torques["tau_brk_true_i"] * np.sign(np.maximum(state.omega, 1e-6))
            - tire["Fx_wheel_i"] * radius
        ) / wheel_inertia
        state.omega = np.maximum(0.0, state.omega + wheel_alpha * dt)

        for name, max_abs in self.config.numerical_limits.max_abs_state.items():
            value = getattr(state, name)
            if not np.isfinite(value) or abs(value) > max_abs:
                raise FloatingPointError("%s exceeded numerical limit" % name)

    def _append_aux(
        self,
        aux_rows: Dict[str, List[Any]],
        tire: Dict[str, np.ndarray],
        load_state: Dict[str, np.ndarray],
        torques: Dict[str, np.ndarray],
        steering: Dict[str, np.ndarray],
        road_contact: Dict[str, object],
        aero: Dict[str, np.ndarray],
    ) -> None:
        items = {
            "Fz_true_i": load_state["Fz_true_i"],
            "Fx_true_i": tire["Fx_true_i"],
            "Fy_true_i": tire["Fy_true_i"],
            "mu_true_i": tire["mu_true_i"],
            "slip_ratio_true_i": tire["slip_ratio_true_i"],
            "slip_angle_true_i": tire["slip_angle_true_i"],
            "Mz_true_i": tire["Mz_true_i"],
            "friction_usage_i": tire["friction_usage_i"],
            "delta_eff_i": steering["delta_eff_i"],
            "suspension_states": load_state["suspension_states"],
            "unsprung_states": load_state["unsprung_states"],
            "camber_true_i": load_state["camber_true_i"],
            "toe_true_i": load_state["toe_true_i"],
            "actuator_delay_states": steering["actuator_delay_states"],
            "tau_drv_true_i": torques["tau_drv_true_i"],
            "tau_brk_true_i": torques["tau_brk_true_i"],
            "brake_pressure_i": torques["brake_pressure_i"],
            "abs_tcs_esc_modulation_states": torques["abs_tcs_esc_modulation_states"],
            "brake_temperature_i": torques["brake_temperature_i"],
            "road_height_true_i": road_contact["height"],
            "road_normal_true_i": road_contact["normal"],
            "grade_true": np.array(float(road_contact["grade"])),
            "bank_true": np.array(float(road_contact["bank"])),
            "aero_force_moment_diagnostics": aero["force_moment"],
        }
        for key, value in items.items():
            aux_rows.setdefault(key, []).append(np.asarray(value, dtype=np.float64))

    @staticmethod
    def _stack_obs(obs_rows: List[Dict[str, float]]) -> Dict[str, np.ndarray]:
        keys = obs_rows[0].keys()
        return {
            key: np.asarray([row[key] for row in obs_rows], dtype=np.float64 if key == "timestamp" else np.float32)
            for key in keys
        }

    @staticmethod
    def _stack_aux(
        aux_rows: Dict[str, List[Any]],
        road_labels: List[List[str]],
        hidden_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        aux: Dict[str, Any] = {
            key: np.asarray(values, dtype=np.float32)
            for key, values in aux_rows.items()
        }
        aux["road_surface_labels"] = road_labels
        aux["teacher_hidden_params"] = hidden_params
        return aux

    def _metadata(
        self, scenario: ScenarioConfig, length: int, vehicle_hash: str
    ) -> Dict[str, Any]:
        road = scenario.road_profile
        vehicle = scenario.vehicle_config
        observable_fields = list(self._observable_field_names())
        teacher_only_fields = list(self._teacher_field_names())
        return {
            "episode_id": "ep_%s_seed_%d" % (scenario.scenario_id, scenario.seed),
            "vehicle_id": vehicle.vehicle_id,
            "vehicle_family": vehicle.vehicle_family,
            "vehicle_config_id": vehicle.vehicle_config_id,
            "scenario_id": scenario.scenario_id,
            "scenario_group": scenario.scenario_group,
            "road_factor_id": scenario.road_factor_id or road.factor_id,
            "longitudinal_factor_id": scenario.longitudinal_factor_id
            or scenario.control_script.longitudinal_id,
            "lateral_factor_id": scenario.lateral_factor_id
            or scenario.control_script.lateral_id,
            "full_matrix_index": scenario.full_matrix_index,
            "road_type": road.road_type,
            "mu_pattern": road.mu_pattern,
            "split_mu_type": (
                "left_%s_right_%s" % (road.split_left, road.split_right)
                if road.road_type == "split"
                else None
            ),
            "transition_type": (
                "%s_to_%s" % (road.transition_from, road.transition_to)
                if road.road_type == "transition"
                else None
            ),
            "control_script": {
                "longitudinal": scenario.control_script.longitudinal,
                "lateral": scenario.control_script.lateral,
                "longitudinal_factor_id": scenario.control_script.longitudinal_id,
                "lateral_factor_id": scenario.control_script.lateral_id,
                "initial_speed_mps": scenario.control_script.initial_speed_mps,
            },
            "seed": int(scenario.seed),
            "duration_s": float(self.config.duration_s),
            "dt": float(self.config.dt_export),
            "sample_count": int(length),
            "schema_version": self.config.schema_version,
            "teacher_model_version": self.config.teacher_model_version,
            "sensor_noise_profile": scenario.sensor_profile.__dict__,
            "actuator_profile": scenario.actuator_profile.__dict__,
            "torque_observability_mode": scenario.actuator_profile.torque_observability_mode,
            "teacher_feature_flags": {
                **self.config.default_feature_flags,
                **scenario.teacher_feature_flags,
            },
            "vehicle_internal_params_hash_algo": "sha256",
            "vehicle_internal_params_hash": vehicle_hash,
            "observable_fields": observable_fields,
            "teacher_only_fields": teacher_only_fields,
            "split_role": scenario.split_metadata.split_role,
            "target_window_id": scenario.split_metadata.target_window_id,
            "fine_tune_data_bucket": scenario.split_metadata.fine_tune_data_bucket,
            "validation_case": scenario.validation_case,
            "perturbation_profile_id": scenario.perturbation_profile_id,
            "target_window_role": scenario.target_window_role,
        }

    @staticmethod
    def _observable_field_names() -> List[str]:
        return [
            "timestamp",
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
            "sw_angle",
            "tau_drv_obs_fl",
            "tau_drv_obs_fr",
            "tau_drv_obs_rl",
            "tau_drv_obs_rr",
            "tau_brk_obs_fl",
            "tau_brk_obs_fr",
            "tau_brk_obs_rl",
            "tau_brk_obs_rr",
            "steer_cmd",
            "throttle_cmd",
            "brake_cmd",
        ]

    @staticmethod
    def _teacher_field_names() -> List[str]:
        return [
            "Fz_true_i",
            "Fx_true_i",
            "Fy_true_i",
            "mu_true_i",
            "slip_ratio_true_i",
            "slip_angle_true_i",
            "Mz_true_i",
            "friction_usage_i",
            "delta_eff_i",
            "suspension_states",
            "unsprung_states",
            "camber_true_i",
            "toe_true_i",
            "actuator_delay_states",
            "tau_drv_true_i",
            "tau_brk_true_i",
            "brake_pressure_i",
            "abs_tcs_esc_modulation_states",
            "brake_temperature_i",
            "road_surface_labels",
            "road_height_true_i",
            "road_normal_true_i",
            "grade_true",
            "bank_true",
            "aero_force_moment_diagnostics",
            "teacher_hidden_params",
        ]
