import time
import unittest

import numpy as np

from src.transfer_eval import summarize


class TransferEvalTests(unittest.TestCase):
    def test_call_aggregation_and_multiclass_metrics(self):
        groups = {
            "a": [(0, np.array([0.8, 0.1, 0.1]))],
            "b": [(1, np.array([0.1, 0.8, 0.1]))],
            "c": [(2, np.array([0.1, 0.1, 0.8]))],
        }
        metrics = summarize(groups, ["a", "b", "c"], time.perf_counter(), 3, False)
        self.assertEqual(metrics["n_calls"], 3)
        self.assertEqual(metrics["n_chunks"], 3)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["auroc_ovr_macro"], 1.0)
        self.assertFalse(metrics["transcript_generated"])

    def test_binary_auc_does_not_require_normalized_class_probabilities(self):
        groups = {
            "a": [(0, np.array([0.7, 0.1]))],
            "b": [(0, np.array([0.8, 0.2]))],
            "c": [(1, np.array([0.2, 0.9]))],
            "d": [(1, np.array([0.1, 0.7]))],
        }
        metrics = summarize(groups, ["not_urgent", "urgent"], time.perf_counter(), 4, False)
        self.assertEqual(metrics["auroc_ovr_macro"], 1.0)


if __name__ == "__main__":
    unittest.main()
