import os
import unittest

from experiments.materialize_data import validate_dataset
from student_model.data import (
    context_vector,
    episode_arrays,
    load_episode_record,
    validate_canonical_dataset,
)


class CanonicalDataTest(unittest.TestCase):
    def test_ds1_canonical_dataset_is_readable(self):
        dataset_dir = "data/ds1_v1"
        self.assertTrue(os.path.exists(dataset_dir))
        materialized = validate_dataset(dataset_dir)
        self.assertEqual(materialized["episode_count"], 120)
        self.assertGreaterEqual(materialized["split_role_count"], 5)
        summary = validate_canonical_dataset(dataset_dir)
        self.assertEqual(summary["state_dim"], 12)
        self.assertEqual(summary["control_dim"], 12)
        self.assertEqual(summary["context_dim"], 17)
        self.assertGreater(summary["sequence_len"], 2)

    def test_episode_record_arrays_and_context(self):
        record = load_episode_record("data/ds1_v1", 0)
        states, controls = episode_arrays(record)
        ctx = context_vector(record)
        self.assertEqual(states.shape[1], 12)
        self.assertEqual(controls.shape[1], 12)
        self.assertEqual(ctx.shape[0], 17)
        self.assertEqual(states.shape[0], controls.shape[0])


if __name__ == "__main__":
    unittest.main()
