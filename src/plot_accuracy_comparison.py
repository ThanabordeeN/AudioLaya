"""Academic accuracy-comparison figure across the three evaluation sets.

Reads the stored result JSONs (single source of truth) and writes
results/accuracy_comparison.pdf and .png. Run:

    python -m src.plot_accuracy_comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Method key -> (panel label, colour, hatch). Hatch marks the transcript-input route.
METHODS = {
    "A": ("A: projector only", "#4C72B0", ""),
    "A+sem": ("A + semantic", "#8CA9D4", "///"),
    "A+sem3": ("A + semantic (3 ep)", "#C4D2E8", "///"),
    "B": ("B: projector + head", "#DD8452", ""),
    "C": ("C: ASR -> Laya (text)", "#55A868", "xxx"),
}

PANELS = (
    # title, n, majority-class baseline, {method: result file}
    (
        "Robocall screening (binary)",
        282,
        146 / 282,
        {
            "A": ("results/projector_only/metrics.json", None),
            "A+sem": ("results/projector_semantic_test/metrics.json", None),
            "A+sem3": ("results/projector_semantic_3epochs_test/metrics.json", None),
            "B": ("results/metrics.json", None),
            "C": ("results/transcript_baseline.json", None),
        },
    ),
    (
        "MInDS-14 zero-shot (14-way)",
        563,
        0.0852575488454707,
        {
            "A": ("results/transfer/minds14_zero_shot.json", "A_projector_only"),
            "A+sem": ("results/transfer/minds14_semantic_zero_shot.json", "A_projector_semantic_alignment"),
            "A+sem3": ("results/transfer/minds14_semantic_3epochs.json", "A_projector_semantic_alignment"),
            "B": ("results/transfer/minds14_zero_shot.json", "B_projector_plus_head"),
            "C": ("results/transfer/minds14_zero_shot.json", "C_asr_to_laya"),
        },
    ),
    (
        "Urgency tone zero-shot (binary)",
        40,
        0.5,
        {
            "A": ("results/transfer/urgency_zero_shot.json", "A_projector_only"),
            "A+sem": ("results/transfer/urgency_semantic_zero_shot.json", "A_projector_semantic_alignment"),
            "A+sem3": ("results/transfer/urgency_semantic_3epochs.json", "A_projector_semantic_alignment"),
            "B": ("results/transfer/urgency_zero_shot.json", "B_projector_plus_head"),
            "C": ("results/transfer/urgency_zero_shot.json", "C_asr_to_laya"),
        },
    ),
)


def accuracy(path: str | Path, method_key: str | None) -> float:
    """Read accuracy from a result file; method_key selects a nested entry."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if method_key is not None:
        payload = payload["methods"][method_key]
    return float(payload["accuracy"])


def build(root: Path) -> dict:
    """Collect {panel title: (accuracy array, baseline)} for every method."""
    collected = {}
    for title, n, baseline, sources in PANELS:
        values = []
        for key, (path, method_key) in sources.items():
            value = accuracy(root / path, method_key)
            assert 0.0 <= value <= 1.0, f"{title}/{key}: accuracy {value} out of range"
            values.append(value)
        collected[title] = (np.array(values), baseline, n)
    return collected


def plot(collected: dict, out_pdf: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "hatch.linewidth": 0.5,
        }
    )
    keys = list(METHODS)
    x = np.arange(len(keys))
    width = 0.78

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
    for ax, (title, (values, baseline, n)) in zip(axes, collected.items()):
        ax.bar(
            x,
            values,
            width,
            color=[METHODS[k][1] for k in keys],
            hatch=[METHODS[k][2] for k in keys],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        ax.axhline(baseline, color="#444444", linestyle=(0, (4, 2)), linewidth=0.7, zorder=4)
        for xi, value in zip(x, values):
            ax.annotate(
                f"{value:.3f}",
                (xi, value),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.4,
            )
        ax.set_title(f"{title}\n$N = {n}$, majority class {baseline:.3f}", linespacing=1.4, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=30, ha="right")
        ax.set_ylim(0, 1.12)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Accuracy")

    handles = [Patch(facecolor=c, hatch=h, edgecolor="white", linewidth=0.4, label=label) for label, c, h in METHODS.values()]
    handles.append(
        plt.Line2D([], [], color="#444444", linestyle=(0, (4, 2)), linewidth=0.7, label="majority-class baseline")
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
        handlelength=1.5,
        columnspacing=1.1,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the accuracy-comparison figure")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="results/accuracy_comparison.pdf")
    args = parser.parse_args()
    collected = build(Path(args.root))
    plot(collected, Path(args.root) / args.out)
    print(f"wrote {args.out} and {Path(args.out).with_suffix('.png')}")
    for title, (values, baseline, _) in collected.items():
        print(f"  {title}: " + ", ".join(f"{k}={v:.3f}" for k, v in zip(METHODS, values)) + f" (baseline {baseline:.3f})")


if __name__ == "__main__":
    main()
