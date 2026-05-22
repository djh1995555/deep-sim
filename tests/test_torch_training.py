import importlib.util
import json
import os
import tempfile
import unittest

from experiments.torch_training import (
    _build_direct_model,
    _checkpoint_payload,
    _restore_model_from_checkpoint,
    run_torch_training_suite,
)
from simulator.vehicle_model.config import load_yaml


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
            "R105": "forward_loss_smoke",
            "R106": "tiny_overfit",
            "R107": "one_step_train",
            "R108": "rollout_eval",
            "R109": "resume_eval_smoke",
            "R110": "one_step_train",
            "R111": "one_step_train",
            "R112": "fair_compare",
            "R113": "model_variant_smoke",
            "R114": "fine_tune_smoke",
            "R115": "ensemble_train",
        }
        for run_id, mode in expected_modes.items():
            with self.subTest(run_id=run_id):
                cfg = load_yaml("configs/runs/%s.yaml" % run_id)
                self.assertEqual(cfg["dataset_source"], "existing")
                expected_dataset = "data/ds1_proxy_ft_v1" if run_id == "R114" else "data/ds1_v1"
                self.assertEqual(cfg["data"]["dataset_path"], expected_dataset)
                self.assertEqual(cfg["torch_training"]["mode"], mode)
                self.assertEqual(cfg["logging"]["output_dir"].split("/")[1][:4], run_id)
                if run_id in {
                    "R105",
                    "R106",
                    "R107",
                    "R108",
                    "R109",
                    "R110",
                    "R111",
                    "R112",
                    "R113",
                    "R114",
                    "R115",
                }:
                    self.assertTrue(cfg["torch_training"]["require_cuda"])
                    self.assertEqual(cfg["torch_training"]["device"], "cuda")
                if run_id == "R110":
                    self.assertEqual(cfg["torch_training"]["model_type"], "direct_tcn")

    def test_direct_tcn_checkpoint_restores_saved_config(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch is not installed")
        import torch

        device = torch.device("cpu")
        cfg = {
            "model_type": "direct_tcn",
            "baseline_config": {
                "hidden_dim": 32,
                "context_dim": 17,
                "direct_residual_bound": 0.6,
            },
        }
        model = _build_direct_model(torch, cfg["baseline_config"], device)
        payload = _checkpoint_payload(model, None, cfg, step=80, metrics={})
        restored = _restore_model_from_checkpoint(torch, payload, device)
        self.assertEqual(restored.encoder.input_proj.out_channels, 32)

    def test_direct_nbeats_checkpoint_restores(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch is not installed")
        import torch

        device = torch.device("cpu")
        cfg = {
            "model_type": "direct_nbeats",
            "baseline_config": {
                "hidden_dim": 32,
                "context_dim": 17,
                "history_len": 8,
                "direct_arch": "nbeats",
            },
        }
        model = _build_direct_model(torch, cfg["baseline_config"], device)
        payload = _checkpoint_payload(model, None, cfg, step=80, metrics={})
        restored = _restore_model_from_checkpoint(torch, payload, device)
        self.assertTrue(hasattr(restored, "backcast"))

    def test_generated_torch_matrix_manifest(self):
        with open("configs/torch_matrix/MANIFEST.json", "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertEqual(manifest["ablation_count"], 17)
        self.assertEqual(manifest["fine_tune_count"], 35)
        self.assertEqual(manifest["config_count"], 52)
        for item in manifest["configs"]:
            self.assertTrue(os.path.exists(item["path"]))


if __name__ == "__main__":
    unittest.main()
