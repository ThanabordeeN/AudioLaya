import unittest

import torch

from src.models.semantic_alignment import semantic_alignment_loss


class SemanticAlignmentTests(unittest.TestCase):
    def test_matching_audio_text_pairs_have_lower_loss(self):
        audio = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        bank = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        correct = semantic_alignment_loss(audio, bank, torch.tensor([0, 1]))
        swapped = semantic_alignment_loss(audio, bank, torch.tensor([1, 0]))
        self.assertLess(correct.item(), swapped.item())

    def test_temperature_must_be_positive(self):
        with self.assertRaises(ValueError):
            semantic_alignment_loss(torch.ones(1, 2), torch.ones(2, 2), torch.zeros(1, dtype=torch.long), 0)


if __name__ == "__main__":
    unittest.main()
