from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
)


MODEL_ORDER = [
    "svm",
    "mlp",
    "cnn",
]


def plot_model_comparison(
    results: pd.DataFrame,
    output_dir: Path,
) -> None:

    results = (
        results
        .set_index("model")
        .reindex(MODEL_ORDER)
        .dropna()
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    x = np.arange(
        len(results)
    )

    width = 0.35

    ax.bar(
        x - width / 2,
        results["accuracy"],
        width,
        label="Accuracy",
    )

    ax.bar(
        x + width / 2,
        results["f1_macro"],
        width,
        label="F1 Macro",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        results["model"]
        .str.upper()
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "External Domain Performance"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "external_model_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_confidence_vs_accuracy(
    results: pd.DataFrame,
    output_dir: Path,
) -> None:

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    for _, row in results.iterrows():

        ax.scatter(
            row["accuracy"],
            row["mean_confidence"],
            s=100,
        )

        ax.annotate(
            row["model"].upper(),
            (
                row["accuracy"],
                row["mean_confidence"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlim(
        0.0,
        1.0,
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_xlabel(
        "Accuracy"
    )

    ax.set_ylabel(
        "Mean Confidence"
    )

    ax.set_title(
        "Accuracy vs Prediction Confidence"
    )

    ax.grid(
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "accuracy_vs_confidence.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_confusion_matrix(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> None:

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(
                len(
                    class_names
                )
            )
        ),
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    display = (
        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=class_names,
        )
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
        colorbar=False,
    )

    ax.set_title(
        f"{model_name.upper()} - External Domain"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / f"{model_name}_confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def generate_v2_plots(
    results: pd.DataFrame,
    predictions: dict,
    class_names: list[str],
    output_dir: Path,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_model_comparison(
        results,
        output_dir,
    )

    plot_confidence_vs_accuracy(
        results,
        output_dir,
    )

    for (
        model_name,
        prediction_data,
    ) in predictions.items():

        plot_confusion_matrix(
            model_name=model_name,
            y_true=prediction_data[
                "y_true"
            ],
            y_pred=prediction_data[
                "y_pred"
            ],
            class_names=class_names,
            output_dir=output_dir,
        )

    print(
        "[OK] Gráficos Domain Shift V2 gerados."
    )