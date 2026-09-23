"""Frozen Whisper-small.en encoder; the decoder is never called."""

from __future__ import annotations

import math

import torch
from torch import nn
from transformers import WhisperFeatureExtractor, WhisperModel

from src.data.dataset import SAMPLE_RATE


class FrozenWhisperEncoder(nn.Module):
    def __init__(self, model_id: str, device: torch.device):
        super().__init__()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
        full_model = WhisperModel.from_pretrained(model_id)
        self.encoder = full_model.encoder.to(device).eval()
        del full_model
        self.encoder.requires_grad_(False)
        self.device = device
        self.hidden_size = self.encoder.config.d_model

    @torch.no_grad()
    def forward(self, waveforms: list):
        if any(len(audio) == 0 for audio in waveforms):
            raise ValueError("Empty audio chunk")
        features = self.feature_extractor(
            waveforms,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).input_features.to(self.device)
        hidden = self.encoder(input_features=features).last_hidden_state
        # Whisper emits one encoder token per 320 input samples (20 ms at 16 kHz).
        lengths = torch.tensor(
            [min(hidden.shape[1], math.ceil(len(audio) / 320)) for audio in waveforms],
            dtype=torch.long,
            device=self.device,
        )
        mask = torch.arange(hidden.shape[1], device=self.device)[None, :] < lengths[:, None]
        return hidden, mask
