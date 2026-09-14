from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "svm",
    "mlp",
    "cnn",
]


def plot_raw_vs_canonical(
    comparison: pd.DataFrame,
    output_dir: Path,
) -> None:

    comparison = (
        comparison
        .set_index("model")
        .reindex(MODEL_ORDER)
        .dropna()
        .reset_index()
    )

    x = np.arange(
        len(comparison)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        x - width / 2,
        comparison[
            "raw_accuracy"
        ],
        width,
        label="RAW",
    )

    ax.bar(
        x + width / 2,
        comparison[
            "canonical_accuracy"
        ],
        width,
        label="Canonicalized",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        comparison[
            "model"
        ].str.upper()
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_ylabel(
        "Accuracy"
    )

    ax.set_title(
        "External Domain: RAW vs Canonicalized"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "raw_vs_canonical_accuracy.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_accuracy_gain(
    comparison: pd.DataFrame,
    output_dir: Path,
) -> None:

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ordered = (
        comparison
        .set_index("model")
        .reindex(MODEL_ORDER)
        .dropna()
        .reset_index()
    )

    ax.bar(
        ordered[
            "model"
        ].str.upper(),
        ordered[
            "accuracy_gain"
        ],
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_ylabel(
        "Accuracy Gain"
    )

    ax.set_title(
        "Gain from Input Canonicalization"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "canonicalization_accuracy_gain.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_error_confidence(
    comparison: pd.DataFrame,
    output_dir: Path,
) -> None:

    ordered = (
        comparison
        .set_index("model")
        .reindex(MODEL_ORDER)
        .dropna()
        .reset_index()
    )

    x = np.arange(
        len(ordered)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        x - width / 2,
        ordered[
            "raw_error_confidence"
        ],
        width,
        label="RAW",
    )

    ax.bar(
        x + width / 2,
        ordered[
            "canonical_error_confidence"
        ],
        width,
        label="Canonicalized",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        ordered[
            "model"
        ].str.upper()
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_ylabel(
        "Confidence on Errors"
    )

    ax.set_title(
        "Confidence on Wrong Predictions"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "raw_vs_canonical_error_confidence.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def generate_v2_1_plots(
    comparison: pd.DataFrame,
    output_dir: Path,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_raw_vs_canonical(
        comparison,
        output_dir,
    )

    plot_accuracy_gain(
        comparison,
        output_dir,
    )

    plot_error_confidence(
        comparison,
        output_dir,
    )

    print(
        "[OK] Gráficos V2A.1 gerados."
    )