from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.dataset import read_audio_chunks
from src.evaluation import autocast_context
from src.models.audio_laya import AudioLaya


def load_classifier(
    checkpoint: str | Path,
    config_file: str | Path = "configs/poc.yaml",
    device_name: str = "auto",
) -> dict:
    config = yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))
    training = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_ids = payload.get("model_ids", config["models"])
    model = AudioLaya(model_ids["whisper"], model_ids["laya"], payload["experiment"], device)
    device = model.device
    amp = bool(training["mixed_precision"] and device.type == "cuda")
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if amp else None
    model.load_trainable_state(payload["trainable_state"])
    model.eval()
    return {"model": model, "config": config, "device": device, "amp": amp, "amp_dtype": amp_dtype}


@torch.inference_mode()
def classify_audio_file(classifier: dict, audio_path: str | Path) -> dict:
    chunk_seconds = int(classifier["config"]["data"]["max_chunk_seconds"])
    started = time.perf_counter()
    chunks = read_audio_chunks(audio_path, chunk_seconds)
    if not chunks:
        raise ValueError(f"No audio samples found in {audio_path}")

    probabilities = []
    for audio in chunks:
        with autocast_context(classifier["device"], classifier["amp"], classifier["amp_dtype"]):
            logits = classifier["model"]([audio])
        probabilities.append(logits.float().softmax(-1)[0].cpu().numpy())
    average = np.mean(probabilities, axis=0)
    return {
        "label": "spam" if int(average.argmax()) == 1 else "legitimate",
        "probabilities": {"legitimate": float(average[0]), "spam": float(average[1])},
        "chunks": len(chunks),
        "inference_latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "transcript_generated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify an audio file without generating a transcript")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/poc.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    classifier = load_classifier(args.checkpoint, args.config, args.device)
    print(json.dumps(classify_audio_file(classifier, args.audio), indent=2))


if __name__ == "__main__":
    main()
