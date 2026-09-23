from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import CallAudioDataset, collate_audio, collate_audio_with_text, read_manifests
from src.evaluation import autocast_context, evaluate_calls
from src.models.audio_laya import AudioLaya
from src.models.semantic_alignment import semantic_alignment_loss


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    return value


@torch.inference_mode()
def evaluate_semantic_retrieval(model, loader, text_bank, text_indices, temperature, device, amp, amp_dtype):
    model.eval()
    audio_embeddings, targets = [], []
    for waveforms, _labels, _call_ids, transcripts in loader:
        valid = [i for i, text in enumerate(transcripts) if text]
        if not valid:
            continue
        selected = [waveforms[i] for i in valid]
        with autocast_context(device, amp, amp_dtype):
            embeddings = model.encode_semantic_audio(selected)
        audio_embeddings.append(embeddings.float())
        targets.extend(text_indices[transcripts[i]] for i in valid)
    if not audio_embeddings:
        raise ValueError("No paired audio/transcript examples in semantic validation split")
    embeddings = torch.cat(audio_embeddings)
    target_ids = torch.tensor(targets, dtype=torch.long, device=device)
    logits = embeddings @ text_bank.T / temperature
    top_k = logits.topk(min(5, logits.shape[1]), dim=1).indices
    return {
        "loss": float(torch.nn.functional.cross_entropy(logits, target_ids)),
        "recall_at_1": float((top_k[:, 0] == target_ids).float().mean()),
        "recall_at_5": float((top_k == target_ids[:, None]).any(dim=1).float().mean()),
        "n_pairs": len(targets),
        "n_candidates": int(text_bank.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a direct Whisper-to-Laya speech classifier")
    parser.add_argument("--config", type=Path, default=Path("configs/poc.yaml"))
    parser.add_argument("--experiment", choices=("projector_only", "projector_head"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, help="override configured epoch count")
    parser.add_argument("--semantic-weight", type=float, default=0.0, help="weight of audio/transcript InfoNCE loss")
    parser.add_argument("--semantic-temperature", type=float, default=0.07)
    args = parser.parse_args()
    if args.semantic_weight < 0 or args.semantic_temperature <= 0 or (args.epochs is not None and args.epochs < 1):
        parser.error("semantic weight must be nonnegative, temperature positive, and epochs at least one")
    training_started = time.perf_counter()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models, data, training = config["models"], config["data"], config["training"]
    epochs = args.epochs or int(training["epochs"])
    experiment = args.experiment or training["experiment"]
    output_dir = args.output_dir or Path(str(training["output_dir"]).format(experiment=experiment))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(training["seed"]))
    device = choose_device(args.device)
    records = read_manifests(data["manifests"])
    augmentation = data.get("augmentation", {})
    train_set = CallAudioDataset(
        records, "train", int(data["max_chunk_seconds"]), augment=True,
        gain_db=augmentation.get("gain_db", 0), noise_snr_db=augmentation.get("noise_snr_db"),
        include_semantic_text=bool(args.semantic_weight),
    )
    val_set = CallAudioDataset(
        records, "validation", int(data["max_chunk_seconds"]),
        include_semantic_text=bool(args.semantic_weight),
    )
    semantic_pair_count = sum(bool(sample[3]) for sample in train_set.samples) if args.semantic_weight else 0
    model = AudioLaya(models["whisper"], models["laya"], experiment, device)
    device = model.device
    amp = bool(training["mixed_precision"] and device.type == "cuda")
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if amp else None
    scaler = torch.cuda.amp.GradScaler(enabled=amp and amp_dtype == torch.float16)
    loader_args = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training["workers"]),
        "collate_fn": collate_audio,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_set, shuffle=True,
        collate_fn=collate_audio_with_text if args.semantic_weight else collate_audio,
        **{key: value for key, value in loader_args.items() if key != "collate_fn"},
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    semantic_val_loader = None
    text_bank = text_indices = val_text_bank = val_text_indices = None
    if args.semantic_weight:
        train_texts, val_texts = train_set.semantic_texts, val_set.semantic_texts
        if len(train_texts) < 2 or not val_texts:
            raise ValueError("Semantic alignment needs at least two train texts and one validation text")
        semantic_val_loader = DataLoader(
            val_set, shuffle=False, collate_fn=collate_audio_with_text,
            **{key: value for key, value in loader_args.items() if key != "collate_fn"},
        )
        text_indices = {text: i for i, text in enumerate(train_texts)}
        val_text_indices = {text: i for i, text in enumerate(val_texts)}
        text_bank = model.semantic_text_embeddings(train_texts).detach()
        val_text_bank = model.semantic_text_embeddings(val_texts).detach()

    optimizer_groups = [{"params": list(model.projector.parameters()), "lr": training["lr_projector"]}]
    if experiment == "projector_head":
        for name in ("head", "type_emb", "scorer"):
            module = model._trainable_head_modules[name]
            optimizer_groups.append({"params": list(module.parameters()), "lr": training["lr_laya_head"]})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=float(training["weight_decay"]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history = []
    best_f1, best_semantic_loss, stale_epochs = -1.0, float("inf"), 0
    checkpoint_path = output_dir / "best.pt"
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_loss, total_task_loss, total_semantic_loss = 0.0, 0.0, 0.0
        total_chunks, semantic_pairs = 0, 0
        for batch in train_loader:
            waveforms, labels, _call_ids = batch[:3]
            transcripts = batch[3] if args.semantic_weight else []
            targets = torch.tensor(labels, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp, amp_dtype):
                if args.semantic_weight:
                    logits, audio_embeddings = model(waveforms, return_embedding=True)
                else:
                    logits = model(waveforms)
                task_loss = torch.nn.functional.cross_entropy(logits, targets)
                valid = [i for i, text in enumerate(transcripts) if text]
                if valid:
                    semantic_targets = torch.tensor(
                        [text_indices[transcripts[i]] for i in valid], dtype=torch.long, device=device
                    )
                    semantic_loss = semantic_alignment_loss(
                        audio_embeddings[valid], text_bank, semantic_targets, args.semantic_temperature
                    )
                else:
                    semantic_loss = task_loss.new_zeros(())
                loss = task_loss + args.semantic_weight * semantic_loss
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), float(training["gradient_clip"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), float(training["gradient_clip"]))
                optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            total_task_loss += float(task_loss.detach()) * len(labels)
            total_semantic_loss += float(semantic_loss.detach()) * len(valid)
            total_chunks += len(labels)
            semantic_pairs += len(valid)

        train_loss = total_loss / max(total_chunks, 1)
        val_metrics, _ = evaluate_calls(model, val_loader, device, amp, amp_dtype)
        validation_semantic = None
        if args.semantic_weight:
            validation_semantic = evaluate_semantic_retrieval(
                model, semantic_val_loader, val_text_bank, val_text_indices,
                args.semantic_temperature, device, amp, amp_dtype,
            )
        row = {
            "epoch": epoch,
            "epoch_seconds": round(time.perf_counter() - epoch_started, 3),
            "train_loss": train_loss,
            "train_task_loss": total_task_loss / max(total_chunks, 1),
            "validation": val_metrics,
        }
        if args.semantic_weight:
            row["train_semantic_loss"] = total_semantic_loss / max(semantic_pairs, 1)
            row["validation_semantic"] = validation_semantic
        history.append(row)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        line = (
            f"epoch {epoch:02d}: train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f}"
        )
        if validation_semantic:
            line += (
                f" val_semantic_loss={validation_semantic['loss']:.4f}"
                f" retrieval@1={validation_semantic['recall_at_1']:.3f}"
            )
        print(line)

        improved = (
            validation_semantic["loss"] < best_semantic_loss
            if validation_semantic else val_metrics["macro_f1"] > best_f1
        )
        if improved:
            best_f1 = max(best_f1, val_metrics["macro_f1"])
            if validation_semantic:
                best_semantic_loss = validation_semantic["loss"]
            stale_epochs = 0
            torch.save({
                "experiment": experiment,
                "epoch": epoch,
                "validation_macro_f1": val_metrics["macro_f1"],
                "validation_semantic": validation_semantic,
                "semantic_weight": args.semantic_weight,
                "semantic_temperature": args.semantic_temperature,
                "semantic_train_pairs": semantic_pair_count,
                "semantic_train_unique_texts": len(train_set.semantic_texts) if args.semantic_weight else 0,
                "trainable_state": cpu_state(model.save_trainable_state()),
                "trainable_parameter_count": model.trainable_parameter_count(),
                "model_ids": models,
            }, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= int(training["patience"]):
                print(f"early stopping after {epoch} epochs")
                break

    peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    run_info = {
        "experiment": experiment,
        "best_validation_macro_f1": best_f1,
        "best_validation_semantic_loss": None if not args.semantic_weight else best_semantic_loss,
        "semantic_weight": args.semantic_weight,
        "semantic_temperature": args.semantic_temperature,
        "semantic_train_pairs": semantic_pair_count,
        "semantic_train_unique_texts": 0 if not args.semantic_weight else len(train_set.semantic_texts),
        "training_wall_seconds": round(time.perf_counter() - training_started, 3),
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "trainable_parameter_count": model.trainable_parameter_count(),
        "peak_vram_bytes": peak_vram,
        "checkpoint": str(checkpoint_path),
    }
    (output_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(json.dumps(run_info, indent=2))


if __name__ == "__main__":
    main()
