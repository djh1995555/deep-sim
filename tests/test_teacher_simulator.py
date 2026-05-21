import os
import tempfile
import unittest

from teacher_simulator.config import load_teacher_config
from teacher_simulator.export import export_dataset
from teacher_simulator.scenario import make_ds0_scenarios
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


if __name__ == "__main__":
    unittest.main()
