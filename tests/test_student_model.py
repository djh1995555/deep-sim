import importlib.util
import unittest

import numpy as np

from student_model.data import (
    TorchTransitionDataset,
    context_vector,
    episode_arrays,
    load_episode_record,
    observable_history,
)


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
class StudentModelSmokeTest(unittest.TestCase):
    def test_forward_shapes(self):
        import torch

        from student_model.torch_model import HybridStudentConfig, HybridStudentModel

        record = load_episode_record("data/ds1_v1", 0)
        states, controls = episode_arrays(record)
        history = observable_history(states, controls)[:8]
        model = HybridStudentModel(HybridStudentConfig(history_len=8))
        out = model(
            observable_history=torch.from_numpy(history[None, :, :]),
            current_state=torch.from_numpy(states[7:8]),
            current_control=torch.from_numpy(controls[7:8]),
            context=torch.from_numpy(context_vector(record)[None, :]),
            dt=torch.tensor([float(record.metadata["dt"])], dtype=torch.float32),
        )
        self.assertEqual(tuple(out["x_next"].shape), (1, 12))
        self.assertEqual(tuple(out["delta_x"].shape), (1, 12))
        self.assertEqual(tuple(out["fz"].shape), (1, 4))
        self.assertEqual(tuple(out["tire_forces"].shape), (1, 8))
        self.assertTrue(np.isfinite(out["x_next"].detach().numpy()).all())

    def test_encoder_and_component_variants_forward(self):
        import torch

        from student_model.torch_model import HybridStudentConfig, HybridStudentModel

        record = load_episode_record("data/ds1_v1", 0)
        states, controls = episode_arrays(record)
        history = observable_history(states, controls)[:8]
        for encoder_type in ["gru", "tcn", "transformer"]:
            with self.subTest(encoder_type=encoder_type):
                model = HybridStudentModel(
                    HybridStudentConfig(
                        hidden_dim=32,
                        history_len=8,
                        encoder_type=encoder_type,
                        tire_mode="T2",
                        mu_mode="M1a",
                    )
                )
                out = model(
                    observable_history=torch.from_numpy(history[None, :, :]),
                    current_state=torch.from_numpy(states[7:8]),
                    current_control=torch.from_numpy(controls[7:8]),
                    context=torch.from_numpy(context_vector(record)[None, :]),
                    dt=torch.tensor([float(record.metadata["dt"])], dtype=torch.float32),
                )
                self.assertEqual(tuple(out["x_next"].shape), (1, 12))
                self.assertEqual(tuple(out["mu"].shape), (1, 4))
                self.assertEqual(tuple(out["tire_params"].shape), (1, 12))

    def test_fine_tune_trainability_matrix(self):
        from student_model.torch_model import HybridStudentConfig, HybridStudentModel

        for mode in ["FT0", "FT1", "FT2", "FT3", "FT4", "FT5", "FT6"]:
            with self.subTest(mode=mode):
                model = HybridStudentModel(HybridStudentConfig(hidden_dim=32))
                model.set_trainability(mode)
                trainable = [name for name, param in model.named_parameters() if param.requires_grad]
                if mode == "FT0":
                    self.assertEqual(trainable, [])
                elif mode == "FT1":
                    self.assertTrue(all(name.startswith("vehicle_param_adapter") for name in trainable))
                    self.assertTrue(model.config.vehicle_param_adapter_enabled)
                elif mode == "FT6":
                    self.assertGreater(len(trainable), 10)
                    self.assertTrue(model.config.vehicle_param_adapter_enabled)
                else:
                    self.assertGreater(len(trainable), 0)

    def test_transition_dataset_exposes_teacher_aux_labels(self):
        dataset = TorchTransitionDataset(
            "data/ds1_v1",
            history_len=8,
            split_role="train",
            max_samples=1,
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["fz_true"].shape), (4,))
        self.assertEqual(tuple(sample["tire_force_true"].shape), (8,))
        self.assertEqual(tuple(sample["mu_true"].shape), (4,))
        self.assertEqual(tuple(sample["steering_true"].shape), (2,))


if __name__ == "__main__":
    unittest.main()
