from __future__ import annotations

import torch
import torch.nn.functional as F


def semantic_alignment_loss(
    audio_embeddings: torch.Tensor,
    text_bank: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE loss: each projected audio vector retrieves its paired text."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    audio = F.normalize(audio_embeddings.float(), dim=-1)
    texts = F.normalize(text_bank.float(), dim=-1)
    return F.cross_entropy(audio @ texts.T / temperature, targets)
