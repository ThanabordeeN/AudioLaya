from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from src.data.dataset import CallAudioDataset, collate_audio, read_manifests
from src.evaluation import evaluate_calls
from src.models.audio_laya import AudioLaya


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved Audio-Laya checkpoint")
    parser.add_argument("--config", type=Path, default=Path("configs/poc.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training, data = config["training"], config["data"]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = AudioLaya(
        payload.get("model_ids", config["models"])["whisper"],
        payload.get("model_ids", config["models"])["laya"],
        payload["experiment"],
        device,
    )
    device = model.device
    amp = bool(training["mixed_precision"] and device.type == "cuda")
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if amp else None
    model.load_trainable_state(payload["trainable_state"])
    model.eval()

    records = read_manifests(data["manifests"])
    dataset = CallAudioDataset(records, args.split, int(data["max_chunk_seconds"]))
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training["workers"]),
        collate_fn=collate_audio,
        pin_memory=device.type == "cuda",
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metrics, details = evaluate_calls(model, loader, device, amp, amp_dtype)
    metrics.update({
        "experiment": payload["experiment"],
        "checkpoint_epoch": payload["epoch"],
        "trainable_parameter_count": payload["trainable_parameter_count"],
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    })

    out_dir = args.out or Path(config["training"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Legitimate", "Spam"], yticklabels=["Legitimate", "Spam"],
           xlabel="Predicted", ylabel="True", title="Call-level confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(details["labels"], details["probabilities"][:, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUROC = {metrics['auroc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Call-level ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    history_path = args.checkpoint.parent / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        epochs = [row["epoch"] for row in history]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(epochs, [row["train_loss"] for row in history], label="Train")
        ax.plot(epochs, [row["validation"]["loss"] for row in history], label="Validation")
        ax.set(xlabel="Epoch", ylabel="Cross-entropy", title="Training loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "loss_curve.png", dpi=160)
        plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote metrics and plots to {out_dir}")


if __name__ == "__main__":
    main()
