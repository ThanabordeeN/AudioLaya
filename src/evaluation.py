"""Call-aggregated evaluation shared by training and the test-set evaluator."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype | None):
    if device.type == "cuda" and enabled and dtype is not None:
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


@torch.inference_mode()
def evaluate_calls(model, loader, device: torch.device, amp: bool = False, amp_dtype=None) -> tuple[dict, dict]:
    """Average chunk probabilities per original call before computing metrics."""
    model.eval()
    by_call: dict[str, list] = defaultdict(list)
    loss_sum, chunk_count = 0.0, 0
    started = time.perf_counter()
    for waveforms, labels, call_ids in loader:
        targets = torch.tensor(labels, dtype=torch.long, device=device)
        with autocast_context(device, amp, amp_dtype):
            logits = model(waveforms)
            loss = torch.nn.functional.cross_entropy(logits, targets, reduction="sum")
        probabilities = logits.float().softmax(-1).cpu().numpy()
        loss_sum += float(loss)
        chunk_count += len(labels)
        for call_id, label, probability in zip(call_ids, labels, probabilities):
            by_call[call_id].append((int(label), probability))

    elapsed = time.perf_counter() - started
    call_ids = sorted(by_call)
    y_true = np.asarray([by_call[key][0][0] for key in call_ids], dtype=np.int64)
    y_prob = np.asarray([np.mean([p for _, p in by_call[key]], axis=0) for key in call_ids])
    y_pred = y_prob.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob[:, 1])),
        "loss": loss_sum / max(chunk_count, 1),
        "inference_latency_ms_per_call": 1000 * elapsed / max(len(call_ids), 1),
        "n_calls": len(call_ids),
        "n_chunks": chunk_count,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    details = {
        "call_ids": call_ids,
        "labels": y_true,
        "probabilities": y_prob,
        "predictions": y_pred,
    }
    return metrics, details
