"""Academic accuracy-comparison figure across the three evaluation sets.

Reads the stored result JSONs (single source of truth) and writes
results/accuracy_comparison.pdf and .png (English labels) or
results/accuracy_comparison_th.pdf and .png (Thai labels). Run:

    python -m src.plot_accuracy_comparison
    python -m src.plot_accuracy_comparison --lang th
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

# Method key -> (colour, hatch). Hatch marks the text route (C) and the semantic variants.
METHODS = {
    "A": ("#4C72B0", ""),
    "A+sem": ("#8CA9D4", "///"),
    "A+sem3": ("#C4D2E8", "///"),
    "B": ("#DD8452", ""),
    "C": ("#55A868", "xxx"),
}

PANELS = (
    # panel key, n, majority-class baseline, {method: result file}
    (
        "robocall",
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
        "minds14",
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
        "urgency",
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

LABELS = {
    "en": {
        "panels": {
            "robocall": "Robocall screening (binary)",
            "minds14": "MInDS-14 zero-shot (14-way)",
            "urgency": "Urgency tone zero-shot (binary)",
        },
        "methods": {
            "A": "A: projector only",
            "A+sem": "A + semantic",
            "A+sem3": "A + semantic (3 ep)",
            "B": "B: projector + head",
            "C": "C: ASR -> Laya (text)",
        },
        "majority": "majority-class baseline",
        "majority_short": "majority class",
        "y_label": "Accuracy",
        "suffix": "",
        "fonts": ["Liberation Serif", "DejaVu Serif"],
        "legend_ncol": 6,
        "legend_bottom": 0.07,
    },
    "th": {
        "panels": {
            "robocall": "การจำแนกสายโทรศัพท์ (สองคลาส)",
            "minds14": "MInDS-14 แบบไม่ฝึกเพิ่ม (14 คลาส)",
            "urgency": "ความเร่งด่วนแบบไม่ฝึกเพิ่ม (สองคลาส)",
        },
        "methods": {
            "A": "A: ฝึกเฉพาะชั้นแปลงมิติ",
            "A+sem": "A + จับคู่เสียงกับข้อความ",
            "A+sem3": "A + จับคู่เสียงกับข้อความ (3 รอบ)",
            "B": "B: ปรับส่วนตัดสินใจร่วมด้วย",
            "C": "C: ASR $\\rightarrow$ Laya (ข้อความ)",
        },
        "majority": "ทายคลาสที่พบบ่อยที่สุด",
        "majority_short": "คลาสที่พบบ่อย",
        "y_label": "ความแม่นยำ",
        "suffix": "_th",
        "fonts": ["TH SarabunPSK", "Noto Sans Thai", "Noto Serif Thai"],
        "legend_ncol": 3,
        "legend_bottom": 0.16,
    },
}


def accuracy(path: str | Path, method_key: str | None) -> float:
    """Read accuracy from a result file; method_key selects a nested entry."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if method_key is not None:
        payload = payload["methods"][method_key]
    return float(payload["accuracy"])


def build(root: Path) -> dict:
    """Collect {panel key: (accuracy array, baseline, n)} for every method."""
    collected = {}
    for key, n, baseline, sources in PANELS:
        values = []
        for method_key, (path, entry) in sources.items():
            value = accuracy(root / path, entry)
            assert 0.0 <= value <= 1.0, f"{key}/{method_key}: accuracy {value} out of range"
            values.append(value)
        collected[key] = (np.array(values), baseline, n)
    return collected


def plot(collected: dict, out_pdf: Path, labels: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": labels["fonts"],
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
    for ax, (panel, (values, baseline, n)) in zip(axes, collected.items()):
        ax.bar(
            x,
            values,
            width,
            color=[METHODS[k][0] for k in keys],
            hatch=[METHODS[k][1] for k in keys],
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
        ax.set_title(
            f"{labels['panels'][panel]}\n$N = {n}$, {labels['majority_short']} {baseline:.3f}",
            linespacing=1.4,
            pad=10,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=30, ha="right")
        ax.set_ylim(0, 1.12)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(labels["y_label"])

    handles = [
        Patch(facecolor=METHODS[k][0], hatch=METHODS[k][1], edgecolor="white", linewidth=0.4, label=labels["methods"][k])
        for k in keys
    ]
    handles.append(
        plt.Line2D([], [], color="#444444", linestyle=(0, (4, 2)), linewidth=0.7, label=labels["majority"])
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=labels["legend_ncol"],
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
        handlelength=1.5,
        columnspacing=1.1,
    )
    fig.tight_layout(rect=(0, labels["legend_bottom"], 1, 1))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the accuracy-comparison figure")
    parser.add_argument("--root", default=".")
    parser.add_argument("--lang", choices=tuple(LABELS), default="en", help="figure label language")
    parser.add_argument("--out", help="override the output path")
    args = parser.parse_args()
    labels = LABELS[args.lang]
    out = Path(args.out) if args.out else Path(f"results/accuracy_comparison{labels['suffix']}.pdf")
    collected = build(Path(args.root))
    plot(collected, Path(args.root) / out, labels)
    print(f"wrote {out} and {out.with_suffix('.png')}")
    for panel, (values, baseline, _) in collected.items():
        print(f"  {panel}: " + ", ".join(f"{k}={v:.3f}" for k, v in zip(METHODS, values)) + f" (baseline {baseline:.3f})")


if __name__ == "__main__":
    main()
