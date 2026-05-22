import argparse
import csv
import json
import os
import tempfile
import unittest

from experiments.experiment_queue import run_queue
from experiments.matrix_report import build_matrix_report
from experiments.real_data_adapter import convert_csv_to_canonical
from student_model.constants import CONTROL_KEYS, STATE_KEYS
from student_model.data import episode_arrays, load_episode_record, validate_canonical_dataset


class ExperimentEngineeringTest(unittest.TestCase):
    def test_queue_dry_run_writes_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = os.path.join(tmp, "queue_state.json")
            args = argparse.Namespace(
                manifest="",
                configs=["configs/experiments/p6_model_dev/p6_3_pytorch_model_variant_smoke.yaml"],
                run_ids=[],
                limit=1,
                max_retries=0,
                skip_success=False,
                stop_on_failure=False,
                dry_run=True,
                reset_state=True,
                rollout_eval=False,
                rollout_steps=4,
                rollout_max_episodes=1,
                state_path=state_path,
                log_dir=os.path.join(tmp, "logs"),
            )
            state = run_queue(args)
            self.assertTrue(os.path.exists(state_path))
            self.assertEqual(state["jobs"]["R113"]["status"], "dry_run")

    def test_queue_skip_success_can_mark_rollout_needed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "runs", "RTEST_success")
            os.makedirs(run_dir, exist_ok=True)
            config_path = os.path.join(tmp, "RTEST.yaml")
            with open(config_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "\n".join(
                        [
                            "run:",
                            "  id: RTEST",
                            "  name: queue_success_fixture",
                            "logging:",
                            "  output_dir: %s" % run_dir,
                            "",
                        ]
                    )
                )
            with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as handle:
                json.dump({"status": "success", "success_criteria_met": True}, handle)
            args = argparse.Namespace(
                manifest="",
                configs=[config_path],
                run_ids=[],
                limit=1,
                max_retries=0,
                skip_success=True,
                stop_on_failure=False,
                dry_run=True,
                reset_state=True,
                rollout_eval=True,
                rollout_steps=4,
                rollout_max_episodes=1,
                state_path=os.path.join(tmp, "queue_state.json"),
                log_dir=os.path.join(tmp, "logs"),
            )
            state = run_queue(args)
            self.assertEqual(state["jobs"]["RTEST"]["status"], "success_needs_rollout")

    def test_matrix_report_reads_generated_manifest(self):
        report = build_matrix_report("configs/experiments/matrix/MANIFEST.json")
        self.assertEqual(len(report["ablation_rows"]), 17)
        self.assertEqual(len(report["fine_tune_rows"]), 35)
        self.assertIn("status_counts", report)

    def test_real_data_adapter_outputs_canonical_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "episode.csv")
            fieldnames = ["timestamp"] + STATE_KEYS + CONTROL_KEYS
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(4):
                    row = {key: 0.0 for key in fieldnames}
                    row.update(
                        {
                            "timestamp": i * 0.04,
                            "vx": 10.0 + i,
                            "omega_fl": 31.0,
                            "omega_fr": 31.0,
                            "omega_rl": 31.0,
                            "omega_rr": 31.0,
                            "sw_angle": 0.01 * i,
                        }
                    )
                    writer.writerow(row)
            out_dir = os.path.join(tmp, "real_ds")
            summary = convert_csv_to_canonical(
                csv_path,
                out_dir,
                "REAL_TEST",
                "real_ep_000",
            )
            self.assertEqual(summary["episode_count"], 1)
            validated = validate_canonical_dataset(out_dir)
            self.assertEqual(validated["state_dim"], 12)
            record = load_episode_record(out_dir, 0)
            states, controls = episode_arrays(record)
            self.assertEqual(states.shape[0], 4)
            self.assertEqual(controls.shape[1], 12)
            with open(os.path.join(out_dir, "manifest.json"), "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["dataset_id"], "REAL_TEST")
            with open(os.path.join(out_dir, "adapter_quality_report.json"), "r", encoding="utf-8") as handle:
                quality = json.load(handle)
            self.assertEqual(quality["missing_default_field_count"], 0)

    def test_real_data_adapter_rejects_missing_fields_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "episode.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["timestamp", "vx", "sw_angle"])
                writer.writeheader()
                for i in range(4):
                    writer.writerow({"timestamp": i * 0.04, "vx": 10.0 + i, "sw_angle": 0.01 * i})
            with self.assertRaisesRegex(KeyError, "missing required CSV field"):
                convert_csv_to_canonical(csv_path, os.path.join(tmp, "strict_ds"), "REAL_TEST", "real_ep_000")

    def test_real_data_adapter_can_explicitly_default_missing_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "episode.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["timestamp", "vx", "sw_angle"])
                writer.writeheader()
                for i in range(4):
                    writer.writerow({"timestamp": i * 0.04, "vx": 10.0 + i, "sw_angle": 0.01 * i})
            out_dir = os.path.join(tmp, "defaulted_ds")
            summary = convert_csv_to_canonical(
                csv_path,
                out_dir,
                "REAL_TEST",
                "real_ep_000",
                allow_missing_defaults=True,
            )
            self.assertGreater(summary["missing_default_field_count"], 0)


if __name__ == "__main__":
    unittest.main()
