import importlib.util
import unittest

import numpy as np

from student_model.data import context_vector, episode_arrays, load_episode_record, observable_history


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


if __name__ == "__main__":
    unittest.main()
