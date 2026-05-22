import os
import tempfile
import unittest

from simulator.controller import ControllerReference
from simulator.simulator_app import (
    ClosedLoopSimulationRequest,
    build_road_profile,
    build_scenario,
    default_simulator_model_config,
    load_simulation_request,
    run_closed_loop_simulation,
)
from simulator.reference import build_reference_provider
from simulator.vehicle_model.modules.road import RoadModel
from simulator.vehicle_model.config import config_from_dict, load_dataset_config
from simulator.vehicle_model.export import export_dataset, load_dataset
from simulator.vehicle_model.scenario import (
    RoadProfile,
    make_ds1_proxy_scenarios,
    make_ds2_extreme_matrix,
    make_ds2_extreme_scenarios,
    make_proxy_perturbation_profiles,
    make_ds0_scenarios,
    make_ds1_scenario_matrix,
    make_ds1_scenarios,
)
from simulator.vehicle_model.state import TeacherState
from simulator.vehicle_model.model import VehicleModel
from simulator.vehicle_model.vehicle_params import default_vehicle_config
from simulator.vehicle_model.validators import TeacherEpisodeValidator
from simulator.visualizer import DebugTrace


def _sim_model(duration_s=0.5, dt_internal=0.01, dt_export=0.02):
    model = default_simulator_model_config()
    model.update(
        {
            "duration_s": duration_s,
            "dt_internal": dt_internal,
            "dt_export": dt_export,
        }
    )
    return model


def _sim_scenario(
    scenario_id="unit_closed_loop_sim",
    road="single:dry",
    vehicle_index=0,
    seed=5,
    speed_mps=9.0,
    reference=None,
):
    return {
        "id": scenario_id,
        "road": road,
        "vehicle_index": vehicle_index,
        "seed": seed,
        "initial_state": {
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": 0.0,
            "yaw_rad": 0.0,
            "speed_mps": speed_mps,
        },
        "reference": reference or {"type": "fixed", "speed_mps": speed_mps},
    }


class TeacherSimulatorSmokeTest(unittest.TestCase):
    def test_ds0_generation_and_validation(self):
        cfg = load_dataset_config("configs/datasets/ds0_minimal.yaml")
        sim = VehicleModel(cfg)
        episodes = [sim.run_episode(scenario) for scenario in make_ds0_scenarios(cfg.seed)]
        self.assertGreaterEqual(len(episodes), 5)
        with tempfile.TemporaryDirectory() as tmp:
            export_dataset(
                episodes,
                tmp,
                cfg.dataset_id,
                cfg.schema_version,
                cfg.teacher_model_version,
            )
            self.assertTrue(os.path.exists(os.path.join(tmp, "manifest.json")))
            report = TeacherEpisodeValidator().validate_dataset(tmp)
            self.assertTrue(report.passed, report.errors)
            self.assertEqual(report.metrics["leakage_checks_passed"], 1)
            self.assertEqual(report.metrics["has_split_mu"], 1)
            self.assertEqual(report.metrics["has_transition_mu"], 1)

    def test_ds1_matrix_and_vehicle_variants(self):
        matrix = make_ds1_scenario_matrix()
        self.assertEqual(len(matrix), 700)
        scenarios = make_ds1_scenarios(
            seed=23,
            vehicle_count=4,
            samples_per_group_per_vehicle=10,
        )
        self.assertEqual(len(scenarios), 120)
        self.assertEqual(
            len({scenario.scenario_id + str(scenario.seed) for scenario in scenarios}),
            len(scenarios),
        )
        self.assertEqual(
            {scenario.scenario_group for scenario in scenarios},
            {"CG-SINGLE", "CG-SPLIT", "CG-TRANSITION"},
        )
        self.assertGreaterEqual(
            len({scenario.vehicle_config.vehicle_config_id for scenario in scenarios}),
            3,
        )
        self.assertIn(
            "held-out",
            {scenario.split_metadata.split_role for scenario in scenarios},
        )

    def test_ds1_proxy_profiles_and_target_windows(self):
        profiles = make_proxy_perturbation_profiles(seed=31, profile_count=3)
        self.assertEqual(len(profiles), 3)
        self.assertGreaterEqual(
            min(profile["min_abs_magnitude"] for profile in profiles),
            0.05,
        )
        self.assertLessEqual(
            max(profile["max_abs_magnitude"] for profile in profiles),
            0.15,
        )
        scenarios = make_ds1_proxy_scenarios(
            seed=31,
            vehicle_count=4,
            profile_count=3,
            samples_per_group_per_role=3,
        )
        self.assertEqual(len(scenarios), 81)
        self.assertEqual(
            {scenario.target_window_role for scenario in scenarios},
            {"target_train", "target_validation", "target_test"},
        )
        self.assertEqual(
            len({scenario.split_metadata.target_window_id for scenario in scenarios}),
            len(scenarios),
        )
        self.assertEqual(
            len({scenario.perturbation_profile_id for scenario in scenarios}),
            3,
        )

    def test_ds2_extreme_scenarios(self):
        matrix = make_ds2_extreme_matrix()
        self.assertEqual(len(matrix), 6)
        scenarios = make_ds2_extreme_scenarios(
            seed=47,
            vehicle_count=3,
            samples_per_vehicle=6,
        )
        self.assertEqual(len(scenarios), 18)
        self.assertEqual(
            {scenario.scenario_group for scenario in scenarios},
            {"DS2-EXTREME"},
        )
        self.assertIn(
            "fishhook_left_right",
            {scenario.control_script.lateral for scenario in scenarios},
        )
        self.assertIn(
            "held-out",
            {scenario.split_metadata.split_role for scenario in scenarios},
        )

    def test_simulation_app_exports_single_episode_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = ClosedLoopSimulationRequest(
                out_dir=tmp,
                model=_sim_model(duration_s=0.5, dt_export=0.02),
                scenario=_sim_scenario(
                    scenario_id="unit_closed_loop_sim",
                    road="split:dry:ice",
                    vehicle_index=1,
                    seed=5,
                    speed_mps=9.0,
                    reference={
                        "type": "fixed",
                        "speed_mps": 11.0,
                        "y_m": 0.25,
                        "yaw_rad": 0.0,
                    },
                ),
                timestamped_output=False,
            )
            summary = run_closed_loop_simulation(request)
            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["out_dir"], tmp)
            self.assertEqual(summary["episode_count"], 1)
            self.assertNotIn("trajectory_report_path", summary)
            self.assertTrue(os.path.exists(os.path.join(tmp, "manifest.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "simulation_summary.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "debug_trace.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "debug_report.html")))
            self.assertFalse(
                os.path.exists(os.path.join(tmp, "trajectory_comparison.html"))
            )
            with open(os.path.join(tmp, "debug_report.html"), "r", encoding="utf-8") as handle:
                report_html = handle.read()
            self.assertIn("Simulator Debug Report", report_html)
            self.assertIn("Trajectory", report_html)
            self.assertIn("actual vehicle trajectory", report_html)
            self.assertIn("reference trajectory", report_html)
            self.assertIn("Speed Tracking", report_html)
            trace = DebugTrace.read_json(os.path.join(tmp, "debug_trace.json"))
            regenerated = os.path.join(tmp, "debug_report_regenerated.html")
            trace.write_html(
                regenerated,
                panels={"Smoke": ["input.vx", "input.target_speed_mps"]},
                title="Regenerated Debug Report",
            )
            self.assertTrue(os.path.exists(regenerated))
            report = TeacherEpisodeValidator().validate_dataset(tmp)
            self.assertTrue(report.passed, report.errors)
            episodes = load_dataset(tmp)
            self.assertEqual(episodes[0]["metadata"]["scenario_group"], "SIM-CLOSED-LOOP")
            self.assertEqual(episodes[0]["metadata"]["split_role"], "simulation")
            self.assertEqual(episodes[0]["metadata"]["road_type"], "split")
            self.assertEqual(episodes[0]["metadata"]["control_mode"], "closed_loop_pid_lqr")

    def test_simulation_app_road_parser(self):
        single = build_road_profile(
            ClosedLoopSimulationRequest(scenario=_sim_scenario(road="dry"))
        )
        self.assertEqual(single.road_type, "single")
        self.assertEqual(single.single_surface, "dry")
        transition = build_road_profile(
            ClosedLoopSimulationRequest(scenario=_sim_scenario(road="dry->wet"))
        )
        self.assertEqual(transition.road_type, "transition")
        self.assertEqual(transition.transition_from, "dry")
        self.assertEqual(transition.transition_to, "wet")
        with self.assertRaises(ValueError):
            build_road_profile(
                ClosedLoopSimulationRequest(scenario=_sim_scenario(road="single:gravel"))
            )

    def test_split_road_uses_wheel_coordinates(self):
        road = RoadProfile(
            "split",
            "split",
            split_left="dry",
            split_right="ice",
            split_boundary_y_m=0.0,
            split_start_x_m=-10.0,
            split_end_x_m=10.0,
            split_default_surface="wet",
        )
        vehicle = default_vehicle_config()
        model = RoadModel()

        centered = TeacherState(x_world=0.0, y_world=0.0, yaw=0.0)
        centered_contact = model.query(0.0, road, centered, vehicle)
        self.assertEqual(centered_contact["labels"], ["dry", "ice", "dry", "ice"])

        shifted_left = TeacherState(x_world=0.0, y_world=2.0, yaw=0.0)
        left_contact = model.query(0.0, road, shifted_left, vehicle)
        self.assertEqual(left_contact["labels"], ["dry", "dry", "dry", "dry"])

        outside_region = TeacherState(x_world=20.0, y_world=0.0, yaw=0.0)
        outside_contact = model.query(0.0, road, outside_region, vehicle)
        self.assertEqual(outside_contact["labels"], ["wet", "wet", "wet", "wet"])

    def test_reference_providers(self):
        lane_change = build_reference_provider(
            {
                "type": "lane_change",
                "speed_mps": 12.0,
                "start_y_m": 0.0,
                "end_y_m": 3.5,
                "start_x_m": 10.0,
                "length_m": 20.0,
            },
            fallback=ControllerReference(),
        )
        mid_ref = lane_change.query(0.0, TeacherState(x_world=20.0, y_world=0.0))
        self.assertAlmostEqual(mid_ref.target_y_m, 1.75, places=2)
        self.assertGreater(mid_ref.target_yaw_rad, 0.0)

        waypoints = build_reference_provider(
            {
                "type": "waypoints",
                "lookahead_m": 5.0,
                "points": [
                    {
                        "x_m": 0.0,
                        "y_m": 0.0,
                        "yaw_rad": 0.0,
                        "curvature_1pm": 0.0,
                        "speed_mps": 8.0,
                    },
                    {
                        "x_m": 10.0,
                        "y_m": 0.0,
                        "yaw_rad": 0.0,
                        "curvature_1pm": 0.04,
                        "speed_mps": 10.0,
                    },
                    {
                        "x_m": 20.0,
                        "y_m": 4.0,
                        "yaw_rad": 0.1,
                        "curvature_1pm": 0.08,
                        "speed_mps": 12.0,
                    },
                ],
            },
            fallback=ControllerReference(),
        )
        ref = waypoints.query(0.0, TeacherState(x_world=8.0, y_world=0.0))
        self.assertGreater(ref.target_x_m, 8.0)
        self.assertGreaterEqual(ref.target_speed_mps, 10.0)
        self.assertGreater(ref.target_curvature_1pm, 0.04)

        double_lane_change = build_reference_provider(
            {
                "type": "double_lane_change",
                "speed_mps": 5.0,
                "offset_y_m": 3.5,
                "start_x_m": 0.0,
                "first_length_m": 10.0,
                "hold_length_m": 2.0,
                "second_length_m": 10.0,
            },
            fallback=ControllerReference(),
        )
        first_mid = double_lane_change.query(
            0.0,
            TeacherState(x_world=5.0, y_world=0.0),
        )
        hold = double_lane_change.query(0.0, TeacherState(x_world=11.0, y_world=0.0))
        second_mid = double_lane_change.query(
            0.0,
            TeacherState(x_world=17.0, y_world=0.0),
        )
        self.assertAlmostEqual(first_mid.target_y_m, 1.75, places=2)
        self.assertAlmostEqual(hold.target_y_m, 3.5, places=2)
        self.assertAlmostEqual(second_mid.target_y_m, 1.75, places=2)
        self.assertEqual(first_mid.target_speed_mps, 5.0)
        self.assertGreater(first_mid.target_yaw_rad, 0.0)
        self.assertLess(second_mid.target_yaw_rad, 0.0)

        sinusoidal = build_reference_provider(
            {
                "type": "sinusoidal",
                "speed_mps": 5.0,
                "amplitude_m": 5.0,
                "period_m": 20.0,
                "start_x_m": 0.0,
            },
            fallback=ControllerReference(),
        )
        sine_start = sinusoidal.query(0.0, TeacherState(x_world=0.0, y_world=0.0))
        sine_quarter = sinusoidal.query(0.0, TeacherState(x_world=5.0, y_world=0.0))
        self.assertAlmostEqual(sine_start.target_y_m, 0.0, places=6)
        self.assertAlmostEqual(sine_quarter.target_y_m, 5.0, places=6)
        self.assertAlmostEqual(sine_quarter.target_curvature_1pm, -0.49348, places=4)
        self.assertEqual(sine_quarter.target_speed_mps, 5.0)

    def test_waypoint_reference_requires_explicit_fields(self):
        with self.assertRaisesRegex(ValueError, "waypoint missing required fields"):
            build_reference_provider(
                {
                    "type": "waypoints",
                    "points": [
                        {
                            "x_m": 0.0,
                            "y_m": 0.0,
                            "yaw_rad": 0.0,
                            "speed_mps": 8.0,
                        },
                        {
                            "x_m": 10.0,
                            "y_m": 0.0,
                            "yaw_rad": 0.0,
                            "curvature_1pm": 0.0,
                            "speed_mps": 8.0,
                        },
                    ],
                },
                fallback=ControllerReference(),
            )
        with self.assertRaisesRegex(ValueError, "waypoint must be a mapping"):
            build_reference_provider(
                {
                    "type": "waypoints",
                    "points": [
                        [0.0, 0.0, 8.0, 0.0, 0.0, 0.0],
                        [10.0, 0.0, 8.0, 0.0, 0.0, 0.0],
                    ],
                },
                fallback=ControllerReference(),
            )

    def test_simulation_app_uses_dynamic_reference_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = ClosedLoopSimulationRequest(
                out_dir=tmp,
                model=_sim_model(duration_s=0.4, dt_export=0.02),
                scenario=_sim_scenario(
                    scenario_id="unit_lane_change_reference",
                    road="single:dry",
                    vehicle_index=0,
                    seed=17,
                    speed_mps=8.0,
                    reference={
                        "type": "lane_change",
                        "speed_mps": 10.0,
                        "start_y_m": 0.0,
                        "end_y_m": 1.0,
                        "start_x_m": 0.0,
                        "length_m": 8.0,
                    },
                ),
                timestamped_output=False,
            )
            summary = run_closed_loop_simulation(request)
            self.assertEqual(summary["status"], "success")
            episodes = load_dataset(tmp)
            provider = episodes[0]["metadata"]["controller"]["reference_provider"]
            self.assertEqual(provider["type"], "lane_change")

    def test_simulation_request_loads_reference_file_and_initial_state(self):
        request = load_simulation_request("configs/simulator/double_lane_change_5mps.yaml")
        self.assertEqual(
            request.scenario["reference"]["type"],
            "double_lane_change",
        )
        scenario = build_scenario(request)
        runtime = VehicleModel(config_from_dict(_sim_model(duration_s=0.1))).initialize(
            scenario
        )
        self.assertEqual(scenario.scenario_id, "single_dry_vehicle0_double_lane_change_5mps")
        self.assertAlmostEqual(runtime.state.vx, 5.0)
        self.assertAlmostEqual(runtime.state.x_world, 0.0)
        self.assertAlmostEqual(runtime.state.y_world, 0.0)

    def test_simulation_app_uses_timestamped_output_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = os.path.join(tmp, "unit_timestamped_output")
            request = ClosedLoopSimulationRequest(
                out_dir=output_root,
                model=_sim_model(duration_s=0.2, dt_export=0.02),
                scenario=_sim_scenario(
                    scenario_id="unit_timestamped_output",
                    road="single:dry",
                    reference={"type": "fixed", "speed_mps": 12.0},
                ),
                write_debug_html=False,
            )
            summary = run_closed_loop_simulation(request)
            self.assertEqual(summary["status"], "success")
            self.assertNotEqual(summary["out_dir"], output_root)
            self.assertEqual(os.path.dirname(os.path.dirname(summary["out_dir"])), tmp)
            self.assertTrue(os.path.basename(os.path.dirname(summary["out_dir"])))
            self.assertEqual(os.path.basename(summary["out_dir"]), "unit_timestamped_output")
            self.assertTrue(os.path.exists(os.path.join(summary["out_dir"], "manifest.json")))

    def test_simulation_request_rejects_dataset_config_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy_request.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("dataset_config: configs/datasets/ds0_minimal.yaml\n")
            with self.assertRaisesRegex(ValueError, "unknown simulator request fields: dataset_config"):
                load_simulation_request(path)

    def test_simulation_request_rejects_model_config_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy_request.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("model_config: configs/datasets/ds0_minimal.yaml\n")
            with self.assertRaisesRegex(ValueError, "unknown simulator request fields: model_config"):
                load_simulation_request(path)

    def test_simulation_request_rejects_dataset_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dataset_id_request.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("dataset_id: SIM_MANUAL\n")
            with self.assertRaisesRegex(ValueError, "unknown simulator request fields: dataset_id"):
                load_simulation_request(path)

    def test_simulation_request_rejects_top_level_model_timing_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "timing_request.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("duration_s: 1.0\n")
            with self.assertRaisesRegex(ValueError, "unknown simulator request fields: duration_s"):
                load_simulation_request(path)

    def test_simulation_request_rejects_top_level_target_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "target_request.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("target_speed_mps: 10.0\n")
            with self.assertRaisesRegex(ValueError, "unknown simulator request fields: target_speed_mps"):
                load_simulation_request(path)

    def test_simulation_request_rejects_split_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "split_request.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("split_role: train\n")
            with self.assertRaisesRegex(ValueError, "unknown simulator request fields: split_role"):
                load_simulation_request(path)

    def test_simulation_request_rejects_nested_legacy_scenario_fields(self):
        legacy_fields = ["scenario_id", "dataset_id", "split_role", "target_speed_mps"]
        for field in legacy_fields:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "nested_legacy_request.yaml")
                    with open(path, "w", encoding="utf-8") as handle:
                        handle.write(
                            "scenario:\n"
                            "  id: bad_nested_legacy\n"
                            "  road: single:dry\n"
                            "  initial_state:\n"
                            "    speed_mps: 10.0\n"
                            "  reference:\n"
                            "    type: fixed\n"
                            "    speed_mps: 10.0\n"
                            "  %s: bad\n" % field
                        )
                    with self.assertRaisesRegex(
                        ValueError,
                        "unknown simulator scenario fields: %s" % field,
                    ):
                        load_simulation_request(path)

    def test_simulation_request_rejects_unknown_initial_state_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad_initial_state_request.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "scenario:\n"
                    "  id: bad_initial_state\n"
                    "  road: single:dry\n"
                    "  initial_state:\n"
                    "    vx: 10.0\n"
                    "  reference:\n"
                    "    type: fixed\n"
                    "    speed_mps: 10.0\n"
                )
            with self.assertRaisesRegex(
                ValueError,
                "unknown simulator scenario.initial_state fields: vx",
            ):
                load_simulation_request(path)

    def test_simulation_app_requires_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = ClosedLoopSimulationRequest(
                out_dir=tmp,
                scenario={
                    "id": "missing_reference",
                    "road": "single:dry",
                    "initial_state": {"speed_mps": 12.0},
                },
                model=_sim_model(duration_s=0.2, dt_export=0.02),
                timestamped_output=False,
            )
            with self.assertRaisesRegex(ValueError, "must define reference"):
                run_closed_loop_simulation(request)


if __name__ == "__main__":
    unittest.main()
