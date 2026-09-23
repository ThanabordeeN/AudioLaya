"""Build a call-level manifest for the English FTC Robocall Audio Dataset."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from .splits import assign_splits, write_manifest


def prepare(root: Path, output: Path, seed: int = 42) -> int:
    root = root.resolve()
    metadata = root / "metadata.csv"
    if not metadata.is_file():
        raise FileNotFoundError(f"Expected metadata.csv under {root}")

    rows = []
    seen = set()
    with metadata.open(newline="", encoding="utf-8-sig") as f:
        for item in csv.DictReader(f):
            if item.get("language", "").strip().lower() not in {"en", "english"}:
                continue
            filename = (item.get("file_name") or "").strip()
            if not filename:
                continue
            name = Path(filename)
            # The left channel is the remote caller. Never treat a paired honeypot/right
            # channel as an independent example; it has the same original call ID.
            if name.stem.lower().endswith("_right"):
                continue
            candidates = [root / name, root / "audio-wav-16khz" / name.name]
            audio_path = next(
                (p.resolve() for p in candidates if p.is_file() and p.resolve().is_relative_to(root)),
                None,
            )
            if audio_path is None:
                raise FileNotFoundError(f"Audio listed in metadata is missing: {filename}")
            if audio_path in seen:
                continue
            seen.add(audio_path)
            call_id = re.sub(r"_(?:normalized|left|right)$", "", name.stem, flags=re.IGNORECASE)
            rows.append({
                "source": "ftc",
                "call_id": call_id,
                "audio_path": str(audio_path),
                "label": 1,
                "transcript": (item.get("transcript") or "").strip(),
            })

    if not rows:
        raise ValueError("No English FTC call recordings found")
    write_manifest(assign_splits(rows, seed), output)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Cloned FTC dataset root")
    parser.add_argument("--out", type=Path, default=Path("data/ftc.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"Wrote {prepare(args.root, args.out, args.seed)} English FTC calls to {args.out}")


if __name__ == "__main__":
    main()
