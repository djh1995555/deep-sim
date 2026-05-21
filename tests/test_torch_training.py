import importlib.util
import os
import tempfile
import unittest

from experiments.torch_training import run_torch_training_suite
from teacher_simulator.config import load_yaml


class TorchTrainingRunnerTest(unittest.TestCase):
    def test_data_loader_smoke_reports_blocked_or_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_torch_training_suite(
                "data/ds1_v1",
                {
                    "mode": "data_loader_smoke",
                    "seed": 23,
                    "split_role": "train",
                    "history_len": 8,
                    "batch_size": 2,
                    "max_episodes": 2,
                    "device": "cpu",
                    "model_config": {"hidden_dim": 16, "history_len": 8},
                },
                tmp,
            )
            self.assertTrue(
                os.path.exists(os.path.join(tmp, "artifacts", "torch_training_report.json"))
            )
            if importlib.util.find_spec("torch") is None:
                self.assertTrue(report["blocked"])
                self.assertIn("PyTorch is not installed", report["blocked_reason"])
            else:
                self.assertTrue(report["passed"])
                self.assertEqual(report["metrics"]["torch_data_loader_smoke_passed"], 1)

    def test_pytorch_smoke_configs_are_existing_dataset_runs(self):
        expected_modes = {
            "R100": "data_loader_smoke",
            "R101": "forward_loss_smoke",
            "R102": "tiny_overfit",
            "R103": "rollout_smoke",
            "R104": "checkpoint_smoke",
        }
        for run_id, mode in expected_modes.items():
            with self.subTest(run_id=run_id):
                cfg = load_yaml("configs/runs/%s.yaml" % run_id)
                self.assertEqual(cfg["dataset_source"], "existing")
                self.assertEqual(cfg["data"]["dataset_path"], "data/ds1_v1")
                self.assertEqual(cfg["torch_training"]["mode"], mode)
                self.assertEqual(cfg["logging"]["output_dir"].split("/")[1][:4], run_id)


if __name__ == "__main__":
    unittest.main()
