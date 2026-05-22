import os
import tempfile
import unittest

from simulator.simulator_app import (
    ClosedLoopSimulationRequest,
    build_road_profile,
    run_closed_loop_simulation,
)
from simulator.reference import build_reference_provider
from simulator.vehicle_model.modules.road import RoadModel
from simulator.vehicle_model.config import load_dataset_config
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
                scenario_id="unit_closed_loop_sim",
                dataset_id="SIM_UNIT_TEST",
                road="split:dry:ice",
                vehicle_index=1,
                seed=5,
                initial_speed_mps=9.0,
                target_speed_mps=11.0,
                target_y_m=0.25,
                duration_s=0.5,
                dt_export=0.02,
            )
            summary = run_closed_loop_simulation(request)
            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["out_dir"], tmp)
            self.assertEqual(summary["episode_count"], 1)
            self.assertTrue(os.path.exists(os.path.join(tmp, "manifest.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "simulation_summary.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "debug_trace.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "debug_report.html")))
            with open(os.path.join(tmp, "debug_report.html"), "r", encoding="utf-8") as handle:
                report_html = handle.read()
            self.assertIn("Simulator Debug Report", report_html)
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
        single = build_road_profile(ClosedLoopSimulationRequest(road="dry"))
        self.assertEqual(single.road_type, "single")
        self.assertEqual(single.single_surface, "dry")
        transition = build_road_profile(ClosedLoopSimulationRequest(road="dry->wet"))
        self.assertEqual(transition.road_type, "transition")
        self.assertEqual(transition.transition_from, "dry")
        self.assertEqual(transition.transition_to, "wet")
        with self.assertRaises(ValueError):
            build_road_profile(ClosedLoopSimulationRequest(road="single:gravel"))

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
            fallback=ClosedLoopSimulationRequest().to_reference(),
        )
        mid_ref = lane_change.query(0.0, TeacherState(x_world=20.0, y_world=0.0))
        self.assertAlmostEqual(mid_ref.target_y_m, 1.75, places=2)
        self.assertGreater(mid_ref.target_yaw_rad, 0.0)

        waypoints = build_reference_provider(
            {
                "type": "waypoints",
                "lookahead_m": 5.0,
                "points": [[0.0, 0.0, 8.0], [10.0, 0.0, 10.0], [20.0, 4.0, 12.0]],
            },
            fallback=ClosedLoopSimulationRequest().to_reference(),
        )
        ref = waypoints.query(0.0, TeacherState(x_world=8.0, y_world=0.0))
        self.assertGreater(ref.target_x_m, 8.0)
        self.assertGreaterEqual(ref.target_speed_mps, 10.0)

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
            fallback=ClosedLoopSimulationRequest().to_reference(),
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

    def test_simulation_app_uses_dynamic_reference_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = ClosedLoopSimulationRequest(
                out_dir=tmp,
                scenario_id="unit_lane_change_reference",
                dataset_id="SIM_UNIT_REFERENCE",
                road="single:dry",
                vehicle_index=0,
                seed=17,
                initial_speed_mps=8.0,
                duration_s=0.4,
                dt_export=0.02,
                reference={
                    "type": "lane_change",
                    "speed_mps": 10.0,
                    "start_y_m": 0.0,
                    "end_y_m": 1.0,
                    "start_x_m": 0.0,
                    "length_m": 8.0,
                },
            )
            summary = run_closed_loop_simulation(request)
            self.assertEqual(summary["status"], "success")
            episodes = load_dataset(tmp)
            provider = episodes[0]["metadata"]["controller"]["reference_provider"]
            self.assertEqual(provider["type"], "lane_change")


if __name__ == "__main__":
    unittest.main()
