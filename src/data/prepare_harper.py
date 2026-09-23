"""Build a manifest from HarperValleyBank caller-side recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .splits import assign_splits, write_manifest


def prepare(root: Path, output: Path, seed: int = 42) -> int:
    root = root.resolve()
    candidates = (root / "data" / "audio" / "caller", root / "audio" / "caller", root)
    audio_dir = next((p for p in candidates if p.is_dir() and any(p.glob("*.wav"))), None)
    if audio_dir is None:
        raise FileNotFoundError(f"Could not find data/audio/caller/*.wav under {root}")

    rows = []
    for path in sorted(audio_dir.glob("*.wav")):
        transcript_path = root / "data" / "transcript" / f"{path.stem}.json"
        segments = []
        if transcript_path.is_file():
            for segment in json.loads(transcript_path.read_text(encoding="utf-8")):
                if segment.get("channel_index") not in {1, "1"}:
                    continue
                text = (segment.get("human_transcript") or segment.get("transcript") or "").strip()
                start_ms = segment.get("start_ms", segment.get("offset_ms"))
                if text and start_ms is not None:
                    segments.append({
                        "start_ms": int(start_ms),
                        "duration_ms": int(segment.get("duration_ms", 0)),
                        "text": text,
                    })
        rows.append({
            "source": "harper",
            "call_id": path.stem,
            "audio_path": str(path.resolve()),
            "label": 0,
            "transcript": " ".join(segment["text"] for segment in segments),
            "transcript_segments": segments,
        })
    if not rows:
        raise ValueError(f"No caller-side WAVs found in {audio_dir}")
    write_manifest(assign_splits(rows, seed), output)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Cloned HarperValleyBank dataset root")
    parser.add_argument("--out", type=Path, default=Path("data/harper.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"Wrote {prepare(args.root, args.out, args.seed)} caller recordings to {args.out}")


if __name__ == "__main__":
    main()
