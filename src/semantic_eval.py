from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import CallAudioDataset, collate_audio_with_text
from src.data.splits import read_manifests
from src.evaluation import autocast_context
from src.models.audio_laya import AudioLaya
from src.train import evaluate_semantic_retrieval


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate audio-to-text semantic retrieval without storing transcript text")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/poc.yaml"))
    parser.add_argument("--split", choices=("validation", "test"), nargs="+", default=("validation", "test"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    model = AudioLaya(
        payload["model_ids"]["whisper"], payload["model_ids"]["laya"], payload["experiment"], device,
    )
    model.load_trainable_state(payload["trainable_state"])
    model.eval()

    records = read_manifests(config["data"]["manifests"])
    training = config["training"]
    amp = bool(training["mixed_precision"] and model.device.type == "cuda")
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if amp else None
    temperature = float(payload.get("semantic_temperature", 0.07))
    results = {}
    for split in args.split:
        dataset = CallAudioDataset(
            records, split, int(config["data"]["max_chunk_seconds"]), include_semantic_text=True,
        )
        texts = dataset.semantic_texts
        indices = {text: index for index, text in enumerate(texts)}
        bank = model.semantic_text_embeddings(texts).detach()
        loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=int(training["workers"]),
            collate_fn=collate_audio_with_text, pin_memory=model.device.type == "cuda",
        )
        results[split] = evaluate_semantic_retrieval(
            model, loader, bank, indices, temperature, model.device, amp, amp_dtype,
        )

    result = {
        "checkpoint": str(args.checkpoint),
        "semantic_temperature": temperature,
        "text_encoder": payload["model_ids"]["laya"],
        "metric": "audio-to-reference-transcript retrieval among unique transcripts in each held-out split",
        "results": results,
        "transcript_text_saved": False,
    }
    out = args.out or Path("results/semantic_alignment") / f"{args.checkpoint.parent.name}_retrieval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.checkpoint), "results": results, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
