"""Zero-shot cross-task evaluation for direct-audio and transcript baselines."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.data.dataset import CallAudioDataset, SAMPLE_RATE, collate_audio
from src.evaluation import autocast_context
from src.models.audio_laya import AudioLaya
from src.data.splits import read_manifests


def summarize(groups: dict, names: list[str], started: float, chunks: int, transcript_generated: bool) -> dict:
    call_ids = sorted(groups)
    y_true = np.asarray([groups[key][0][0] for key in call_ids], dtype=np.int64)
    y_prob = np.asarray([np.mean([prob for _, prob in groups[key]], axis=0) for key in call_ids])
    y_pred = y_prob.argmax(axis=1)
    aucs = []
    for class_id in range(len(names)):
        positive = y_true == class_id
        if positive.any() and (~positive).any():
            aucs.append(roc_auc_score(positive, y_prob[:, class_id]))
    auroc = float(np.mean(aucs)) if len(aucs) == len(names) else None
    majority = Counter(y_true.tolist()).most_common(1)[0][0]
    baseline = np.full_like(y_true, majority)
    return {
        "n_calls": len(call_ids),
        "n_chunks": chunks,
        "transcript_generated": transcript_generated,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=np.arange(len(names)), zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", labels=np.arange(len(names)), zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", labels=np.arange(len(names)), zero_division=0)),
        "auroc_ovr_macro": auroc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(len(names))).tolist(),
        "majority_baseline": {
            "class": names[int(majority)],
            "accuracy": float(accuracy_score(y_true, baseline)),
            "macro_f1": float(f1_score(y_true, baseline, average="macro", labels=np.arange(len(names)), zero_division=0)),
        },
        "inference_seconds": round(time.perf_counter() - started, 3),
        "latency_ms_per_call": round((time.perf_counter() - started) * 1000 / max(1, len(call_ids)), 2),
    }


def direct_eval(checkpoint: Path, loader, config: dict, names: list[str], options: dict[str, str], device: torch.device, amp: bool, amp_dtype) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = AudioLaya(
        payload["model_ids"]["whisper"], payload["model_ids"]["laya"], payload["experiment"], device,
        task_prompt=config["task_prompt"], question_instruction=config["question_instruction"], options=options,
    )
    model.load_trainable_state(payload["trainable_state"])
    model.eval()
    groups, chunks = defaultdict(list), 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for waveforms, labels, call_ids in loader:
            with autocast_context(device, amp, amp_dtype):
                logits = model(waveforms)
            probabilities = logits.float().softmax(-1).cpu().numpy()
            chunks += len(labels)
            for call_id, label, prob in zip(call_ids, labels, probabilities):
                groups[call_id].append((int(label), prob))
    result = summarize(groups, names, started, chunks, False)
    result["checkpoint"] = str(checkpoint)
    result["experiment"] = payload["experiment"]
    result["trainable_parameter_count"] = int(payload["trainable_parameter_count"])
    result["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    del model
    return result


def text_eval(loader, records: list[dict], config: dict, names: list[str], options: dict[str, str], device: torch.device, model_ids: dict, use_asr: bool) -> dict:
    import laya

    processor = WhisperProcessor.from_pretrained(model_ids["whisper"]) if use_asr else None
    asr = WhisperForConditionalGeneration.from_pretrained(model_ids["whisper"]).to(device).eval() if use_asr else None
    agent = laya.load(model_ids["laya"], device=str(device))
    text_by_call = {f"{row['source']}:{row['call_id']}": row.get("transcript", "") for row in records}
    question_key = config.get("question_key", "task")
    questions = {question_key: {"type": "choice", "instructions": config["question_instruction"], "criteria": options}}
    groups, chunks = defaultdict(list), 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for waveforms, labels, call_ids in loader:
            if use_asr:
                features = processor.feature_extractor(waveforms, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(device)
                token_ids = asr.generate(input_features=features, max_new_tokens=128)
                texts = processor.batch_decode(token_ids, skip_special_tokens=True)
            else:
                texts = [text_by_call.get(call_id, "") for call_id in call_ids]
            chunks += len(labels)
            for text, label, call_id in zip(texts, labels, call_ids):
                answer = agent.system_one(text or " ", questions)["answers"][question_key]["probabilities"]
                probabilities = np.asarray([answer[name] for name in names], dtype=np.float64)
                groups[call_id].append((int(label), probabilities))
    result = summarize(groups, names, started, chunks, use_asr)
    result["model"] = "Whisper ASR -> standard Laya" if use_asr else "gold transcript -> standard Laya"
    result["trainable_parameter_count"] = 0
    result["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True, help="Task JSON with labels, descriptions and prompt")
    parser.add_argument("--manifest", type=Path, help="Override the manifest path in the task JSON")
    parser.add_argument("--config", type=Path, default=Path("configs/poc.yaml"))
    parser.add_argument("--checkpoint-a", type=Path, default=Path("runs/projector_only/best.pt"))
    parser.add_argument("--checkpoint-b", type=Path, default=Path("runs/projector_head/best.pt"))
    parser.add_argument("--checkpoint-semantic", type=Path, help="A checkpoint trained with semantic alignment loss")
    parser.add_argument("--skip-text-baselines", action="store_true", help="Do not rerun ASR/text methods")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--skip-gold-transcript", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    task = json.loads(args.task.read_text(encoding="utf-8"))
    manifest = args.manifest or Path(task["manifest"])
    options = task["options"]
    names = task["label_names"]
    if list(options) != names:
        raise ValueError("label_names must use the same order as options")
    records = read_manifests([manifest])
    for row in records:
        row.setdefault("split", "test")
    if not args.skip_gold_transcript and not any(row.get("transcript") for row in records):
        raise ValueError("gold-transcript evaluation requested but the manifest has no transcript field")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    data = config["data"]
    dataset = CallAudioDataset(records, "test", int(data["max_chunk_seconds"]))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(config["training"]["workers"]), collate_fn=collate_audio, pin_memory=device.type == "cuda")
    training = config["training"]
    amp = bool(training["mixed_precision"] and device.type == "cuda")
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if amp else None

    methods = {
        "A_projector_only": direct_eval(args.checkpoint_a, loader, task, names, options, device, amp, amp_dtype),
        "B_projector_plus_head": direct_eval(args.checkpoint_b, loader, task, names, options, device, amp, amp_dtype),
    }
    if args.checkpoint_semantic:
        methods["A_projector_semantic_alignment"] = direct_eval(
            args.checkpoint_semantic, loader, task, names, options, device, amp, amp_dtype
        )
    if not args.skip_text_baselines:
        methods["C_asr_to_laya"] = text_eval(loader, records, task, names, options, device, config["models"], True)
        if not args.skip_gold_transcript:
            methods["C_gold_transcript_to_laya"] = text_eval(loader, records, task, names, options, device, config["models"], False)

    protocol = "zero-shot target-task transfer; no target-task labels used for training or tuning"
    if args.checkpoint_semantic:
        protocol += "; semantic checkpoint used paired FTC/Harper source-training transcripts during training only"
    result = {
        "task": task["name"],
        "manifest": str(manifest),
        "dataset_metadata": task.get("dataset_metadata", {}),
        "target_task_training": False,
        "n_examples": len(records),
        "label_names": names,
        "protocol": protocol,
        "transcripts_saved": False,
        "methods": methods,
    }
    out = args.out or Path("results/transfer") / f"{args.task.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"task": task["name"], "n_examples": len(records), "methods": {k: {x: v.get(x) for x in ("accuracy", "balanced_accuracy", "macro_f1", "auroc_ovr_macro", "latency_ms_per_call", "transcript_generated")} for k, v in methods.items()}, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
