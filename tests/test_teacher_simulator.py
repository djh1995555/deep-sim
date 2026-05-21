import os
import tempfile
import unittest

from teacher_simulator.config import load_teacher_config
from teacher_simulator.export import export_dataset
from teacher_simulator.scenario import (
    make_ds0_scenarios,
    make_ds1_scenario_matrix,
    make_ds1_scenarios,
)
from teacher_simulator.simulator import TeacherSimulator
from teacher_simulator.validators import TeacherEpisodeValidator


class TeacherSimulatorSmokeTest(unittest.TestCase):
    def test_ds0_generation_and_validation(self):
        cfg = load_teacher_config("configs/teacher/ds0_minimal.yaml")
        sim = TeacherSimulator(cfg)
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


if __name__ == "__main__":
    unittest.main()
