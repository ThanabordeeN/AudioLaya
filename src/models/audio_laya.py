"""Append projected Whisper states to Laya's text-encoder states before its decision head."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.projector import AudioProjector
from src.models.whisper_encoder import FrozenWhisperEncoder


class AudioLaya(nn.Module):
    OPTIONS = {
        "legitimate": "a legitimate call",
        "spam": "a spam or robocall",
    }

    def __init__(
        self,
        whisper_id: str = "openai/whisper-small.en",
        laya_id: str = "convaiinnovations/laya",
        experiment: str = "projector_head",
        device: str | torch.device = "cpu",
        dropout: float = 0.1,
        task_prompt: str = "Classify the supplied audio call.",
        question_instruction: str = "Is this call a robocall?",
        options: dict[str, str] | None = None,
    ):
        super().__init__()
        if experiment not in {"projector_only", "projector_head"}:
            raise ValueError("experiment must be projector_only or projector_head")
        self.experiment = experiment
        self.options = self.OPTIONS if options is None else options
        if len(self.options) < 2 or any(not key or not value for key, value in self.options.items()):
            raise ValueError("classification options must contain at least two non-empty labels and descriptions")
        requested_device = torch.device(device)

        import laya
        from laya.common import QTYPES, build_sequence

        # Laya's high-level API is inference-only; use its loaded PyTorch modules so the
        # pretrained ModernBERT and decision-head weights remain exactly checkpoint weights.
        self.agent = laya.load(laya_id, device=str(requested_device))
        self.device = torch.device(self.agent.device)
        self.laya_model = self.agent.model
        self.laya_model.encoder.config.reference_compile = False
        self.laya_model.requires_grad_(False)

        self.whisper = FrozenWhisperEncoder(whisper_id, self.device)
        self.projector = AudioProjector(
            input_dim=self.whisper.hidden_size,
            output_dim=self.laya_model.encoder.config.hidden_size,
            dropout=dropout,
        ).to(self.device)

        self._trainable_head_modules = {
            "head": self.laya_model.head,
            "type_emb": self.laya_model.type_emb,
            "scorer": self.laya_model.scorer,
        }
        if experiment == "projector_head":
            for module in self._trainable_head_modules.values():
                module.requires_grad_(True)

        question = {
            "t": "choice",
            "ins": question_instruction,
            "crit": self.options,
        }
        cfg = self.agent.cfg
        ids, markers = build_sequence(
            self.agent.tok,
            task_prompt,
            question,
            int(cfg.get("max_len", 512)),
            int(cfg.get("head_max_len", 192)),
        )
        if len(markers) != len(self.options):
            raise ValueError(f"Laya prompt produced {len(markers)} markers for {len(self.options)} options")
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            text_hidden = self.laya_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        self.register_buffer("text_hidden", text_hidden, persistent=False)
        self.register_buffer("text_mask", attention_mask.bool(), persistent=False)
        self.register_buffer(
            "marker_positions",
            torch.tensor([markers], dtype=torch.long, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "question_type",
            torch.tensor([QTYPES["choice"]], dtype=torch.long, device=self.device),
            persistent=False,
        )
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        self.whisper.eval()
        self.laya_model.encoder.eval()
        self.laya_model.act_head.eval()  # unused by the two-class choice objective
        if self.laya_model.head is not None:
            self.laya_model.head.train(mode if self.experiment == "projector_head" else False)
        return self

    @torch.no_grad()
    def semantic_text_embeddings(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        embeddings = []
        max_len = int(self.agent.cfg.get("max_len", 512))
        for start in range(0, len(texts), batch_size):
            encoded = self.agent.tok(
                texts[start:start + batch_size], padding=True, truncation=True,
                max_length=max_len, return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            hidden = self.laya_model.encoder(
                input_ids=input_ids, attention_mask=attention_mask,
            ).last_hidden_state.float()
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            embeddings.append(F.normalize(pooled, dim=-1))
        return torch.cat(embeddings, dim=0)

    def _pool_audio(self, projected: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
        mask = audio_mask.unsqueeze(-1).to(projected.dtype)
        pooled = (projected * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return F.normalize(pooled.float(), dim=-1)

    def encode_semantic_audio(self, waveforms: list) -> torch.Tensor:
        audio_hidden, audio_mask = self.whisper(waveforms)
        return self._pool_audio(self.projector(audio_hidden), audio_mask)

    def forward(self, waveforms: list, return_embedding: bool = False):
        audio_hidden, audio_mask = self.whisper(waveforms)
        projected = self.projector(audio_hidden)
        audio_embedding = self._pool_audio(projected, audio_mask) if return_embedding else None
        batch = projected.shape[0]

        text_hidden = self.text_hidden.expand(batch, -1, -1).to(dtype=projected.dtype)
        hidden = torch.cat((text_hidden, projected), dim=1)
        text_mask = self.text_mask.expand(batch, -1)
        attention_mask = torch.cat((text_mask, audio_mask), dim=1)
        question_type = self.question_type.expand(batch)
        hidden = hidden + self.laya_model.type_emb(question_type)[:, None, :]

        if self.laya_model.head is not None:
            padding_mask = ~attention_mask
            for layer in self.laya_model.head.layers:
                hidden = layer(hidden, src_key_padding_mask=padding_mask)

        marker_positions = self.marker_positions.expand(batch, -1)
        gather_index = marker_positions[:, :, None].expand(-1, -1, hidden.shape[-1])
        marker_hidden = torch.gather(hidden, 1, gather_index)
        logits = self.laya_model.scorer(marker_hidden).squeeze(-1).float()
        return (logits, audio_embedding) if return_embedding else logits

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters())

    def save_trainable_state(self) -> dict:
        state = {"projector": self.projector.state_dict()}
        if self.experiment == "projector_head":
            state["laya_head"] = {
                name: module.state_dict()
                for name, module in self._trainable_head_modules.items()
            }
        return state

    def load_trainable_state(self, state: dict) -> None:
        self.projector.load_state_dict(state["projector"])
        if self.experiment == "projector_head":
            for name, module in self._trainable_head_modules.items():
                module.load_state_dict(state["laya_head"][name])
