"""Whisper ASR -> standard Laya text classification baseline."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.data.dataset import CallAudioDataset, collate_audio, read_manifests
from src.data.dataset import SAMPLE_RATE

QUESTIONS = {
    "robocall": {
        "type": "choice",
        "instructions": "Is the call a robocall or spam call?",
        "criteria": {
            "legitimate": "a legitimate call",
            "spam": "a spam or robocall",
        },
    }
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/poc.yaml"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("results/transcript_baseline.json"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    records = read_manifests(config["data"]["manifests"])
    dataset = CallAudioDataset(records, args.split, int(config["data"]["max_chunk_seconds"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["workers"]),
        collate_fn=collate_audio,
    )

    model_id = config["models"]["whisper"]
    processor = WhisperProcessor.from_pretrained(model_id)
    asr = WhisperForConditionalGeneration.from_pretrained(model_id).to(device).eval()
    import laya
    agent = laya.load(config["models"]["laya"], device=str(device))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    by_call: dict[str, list] = defaultdict(list)
    started = time.perf_counter()
    with torch.inference_mode():
        for waveforms, labels, call_ids in loader:
            features = processor.feature_extractor(
                waveforms,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            ).input_features.to(device)
            token_ids = asr.generate(input_features=features, max_new_tokens=128)
            transcripts = processor.batch_decode(token_ids, skip_special_tokens=True)
            for transcript, label, call_id in zip(transcripts, labels, call_ids):
                result = agent.system_one(transcript or " ", QUESTIONS)
                probs = result["answers"]["robocall"]["probabilities"]
                by_call[call_id].append((int(label), [probs["legitimate"], probs["spam"]]))

    call_ids = sorted(by_call)
    y_true = np.asarray([by_call[key][0][0] for key in call_ids])
    y_prob = np.asarray([np.mean([p for _, p in by_call[key]], axis=0) for key in call_ids])
    y_pred = y_prob.argmax(axis=1)
    elapsed = time.perf_counter() - started
    metrics = {
        "model": "Whisper-small.en ASR -> standard Laya",
        "transcript_generated": True,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob[:, 1])),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n_calls": len(call_ids),
        "trainable_parameter_count": 0,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "inference_latency_ms_per_call": 1000 * elapsed / max(1, len(call_ids)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
