import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.data.dataset import CallAudioDataset, semantic_text_for_chunk
from src.data.splits import assign_splits


class DatasetTests(unittest.TestCase):
    def test_split_assignment_is_independent_of_input_order(self):
        rows = [
            {"source": "s", "call_id": str(i), "label": i % 2}
            for i in range(30)
        ]
        first = {(r["source"], r["call_id"]): r["split"] for r in assign_splits(rows, seed=17)}
        second = {(r["source"], r["call_id"]): r["split"] for r in assign_splits(rows[::-1], seed=17)}
        self.assertEqual(first, second)

    def test_original_call_stays_in_one_split(self):
        rows = []
        for label in (0, 1):
            for call in range(10):
                rows.append({"source": f"source-{label}", "call_id": str(call), "label": label})
                if call == 0:  # duplicate records/chunks inherit the call's split
                    rows.append({"source": f"source-{label}", "call_id": str(call), "label": label})
        split_rows = assign_splits(rows, seed=7)
        split_by_call = {}
        for row in split_rows:
            key = (row["source"], row["call_id"])
            split_by_call.setdefault(key, row["split"])
            self.assertEqual(split_by_call[key], row["split"])
        for label in (0, 1):
            source_splits = {split_by_call[(f"source-{label}", str(i))] for i in range(10)}
            self.assertEqual(source_splits, {"train", "validation", "test"})

    def test_audio_dataset_ignores_transcript_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "call.wav"
            sf.write(audio_path, np.zeros(1600, dtype=np.float32), 16000)
            row = {
                "source": "test", "call_id": "1", "audio_path": str(audio_path),
                "label": 0, "split": "train", "transcript": "private text",
            }
            self.assertEqual(CallAudioDataset([row], "train")[0]["semantic_text"], "")
            paired = CallAudioDataset([row], "train", include_semantic_text=True)
            self.assertEqual(paired[0]["semantic_text"], "private text")

    def test_chunk_selects_matching_caller_text(self):
        row = {"transcript_segments": [
            {"start_ms": 500, "duration_ms": 200, "text": "first"},
            {"start_ms": 1500, "duration_ms": 200, "text": "second"},
        ]}
        self.assertEqual(semantic_text_for_chunk(row, 0, 1000, 2000, 1), "first")
        self.assertEqual(semantic_text_for_chunk(row, 1000, 2000, 2000, 1), "second")

    def test_full_transcript_only_pairs_with_single_chunk_audio(self):
        row = {"transcript": "whole call"}
        self.assertEqual(semantic_text_for_chunk(row, 0, 30000, 25000, 30), "whole call")
        self.assertEqual(semantic_text_for_chunk(row, 0, 30000, 60000, 30), "")


if __name__ == "__main__":
    unittest.main()
