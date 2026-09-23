"""Lazy 16 kHz audio chunk loading; chunks inherit their call's preassigned split."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from src.data.splits import read_manifests

SAMPLE_RATE = 16_000
SPLITS = ("train", "validation", "test")


def load_chunk(path: str | Path, start_frame: int, frame_count: int) -> tuple[np.ndarray, int]:
    """Read one chunk, downmix to mono, and resample to 16 kHz."""
    with sf.SoundFile(path) as f:
        sample_rate = f.samplerate
        f.seek(start_frame)
        audio = f.read(frame_count, dtype="float32", always_2d=True)
    if not len(audio):
        return np.zeros(0, dtype=np.float32), SAMPLE_RATE
    audio = audio.mean(axis=1, dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, sample_rate // divisor).astype(np.float32)
    return np.asarray(audio, dtype=np.float32), SAMPLE_RATE


def semantic_text_for_chunk(
    row: dict,
    start_ms: float,
    end_ms: float,
    call_duration_ms: float,
    max_chunk_seconds: int,
) -> str:
    if "transcript_segments" in row:
        return " ".join(
            segment["text"]
            for segment in row["transcript_segments"]
            if segment.get("text")
            and start_ms <= float(segment.get("start_ms", 0)) + float(segment.get("duration_ms", 0)) / 2 < end_ms
        )
    if start_ms == 0 and call_duration_ms <= max_chunk_seconds * 1000:
        return (row.get("transcript") or "").strip()
    return ""


def read_audio_chunks(path: str | Path, chunk_seconds: int = 30) -> list[np.ndarray]:
    """Read a file as non-overlapping, at-most-30-second mono 16 kHz chunks."""
    if chunk_seconds < 1 or chunk_seconds > 30:
        raise ValueError("chunk_seconds must be between 1 and 30")
    info = sf.info(path)
    chunk_frames = round(chunk_seconds * info.samplerate)
    chunks = []
    for start in range(0, info.frames, chunk_frames):
        audio, _ = load_chunk(path, start, min(chunk_frames, info.frames - start))
        if len(audio):
            chunks.append(audio)
    return chunks


class CallAudioDataset(Dataset):
    """Expand already-split call manifests into chunks; all chunks inherit their call split."""

    def __init__(
        self,
        records: list[dict],
        split: str,
        chunk_seconds: int = 30,
        augment: bool = False,
        gain_db: float = 0.0,
        noise_snr_db: tuple[float, float] | list[float] | None = None,
        include_semantic_text: bool = False,
    ):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if not 1 <= chunk_seconds <= 30:
            raise ValueError("chunk_seconds must be between 1 and 30")
        self.augment = augment
        self.gain_db = max(0.0, float(gain_db))
        self.noise_snr_db = noise_snr_db
        self.samples: list[tuple[dict, int, int, str]] = []
        for row in records:
            if row["split"] != split:
                continue
            info = sf.info(row["audio_path"])
            frames_per_chunk = round(chunk_seconds * info.samplerate)
            for start in range(0, info.frames, frames_per_chunk):
                count = min(frames_per_chunk, info.frames - start)
                start_ms = start * 1000 / info.samplerate
                end_ms = (start + count) * 1000 / info.samplerate
                semantic_text = semantic_text_for_chunk(
                    row, start_ms, end_ms, info.frames * 1000 / info.samplerate, chunk_seconds
                ) if include_semantic_text else ""
                self.samples.append((row, start, count, semantic_text))
        if not self.samples:
            raise ValueError(f"No audio chunks found for split={split!r}")
        self.semantic_texts = sorted({text for _, _, _, text in self.samples if text})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        row, start, count, semantic_text = self.samples[index]
        audio, _ = load_chunk(row["audio_path"], start, count)
        if self.augment:
            if self.gain_db:
                gain = 10 ** (np.random.uniform(-self.gain_db, self.gain_db) / 20)
                audio *= np.float32(gain)
            if self.noise_snr_db and audio.size:
                rms = float(np.sqrt(np.mean(audio * audio)))
                if rms > 0:
                    lo, hi = self.noise_snr_db
                    snr = np.random.uniform(float(lo), float(hi))
                    audio += np.random.normal(0, rms / (10 ** (snr / 20)), audio.shape).astype(np.float32)
        return {
            "audio": audio,
            "label": int(row["label"]),
            "call_id": f"{row['source']}:{row['call_id']}",
            "semantic_text": semantic_text,
        }


def collate_audio(batch: list[dict]) -> tuple[list[np.ndarray], list[int], list[str]]:
    return (
        [item["audio"] for item in batch],
        [item["label"] for item in batch],
        [item["call_id"] for item in batch],
    )


def collate_audio_with_text(batch: list[dict]) -> tuple[list[np.ndarray], list[int], list[str], list[str]]:
    audio, labels, call_ids = collate_audio(batch)
    return audio, labels, call_ids, [item["semantic_text"] for item in batch]
