from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "svm",
    "mlp",
    "cnn",
]


def _ordered_models(
    values,
) -> list[str]:

    available = set(values)

    return [
        model
        for model in MODEL_ORDER
        if model in available
    ]


def plot_domain_accuracy(
    results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Accuracy dos modelos em cada domínio.
    """

    pivot = results.pivot(
        index="domain",
        columns="model",
        values="accuracy",
    )

    models = _ordered_models(
        pivot.columns
    )

    pivot = pivot[
        models
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    x = np.arange(
        len(pivot.index)
    )

    width = (
        0.8
        / len(models)
    )

    for index, model in enumerate(models):

        offset = (
            index
            - (
                len(models) - 1
            ) / 2
        ) * width

        ax.bar(
            x + offset,
            pivot[model],
            width,
            label=model.upper(),
        )

    ax.set_title(
        "Domain Shift Accuracy"
    )

    ax.set_ylabel(
        "Accuracy"
    )

    ax.set_xlabel(
        "Domain"
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        pivot.index,
        rotation=30,
        ha="right",
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "domain_accuracy.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_accuracy_drop(
    results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Queda de accuracy comparada ao domínio clean.
    """

    shifted = results[
        results["domain"]
        != "clean"
    ]

    pivot = shifted.pivot(
        index="domain",
        columns="model",
        values="accuracy_drop",
    )

    models = _ordered_models(
        pivot.columns
    )

    pivot = pivot[
        models
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    x = np.arange(
        len(pivot.index)
    )

    width = (
        0.8
        / len(models)
    )

    for index, model in enumerate(models):

        offset = (
            index
            - (
                len(models) - 1
            ) / 2
        ) * width

        ax.bar(
            x + offset,
            pivot[model],
            width,
            label=model.upper(),
        )

    ax.set_title(
        "Accuracy Drop under Domain Shift"
    )

    ax.set_ylabel(
        "Accuracy Drop"
    )

    ax.set_xlabel(
        "Domain"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        pivot.index,
        rotation=30,
        ha="right",
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "domain_accuracy_drop.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_mean_confidence(
    results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Confiança média por domínio.
    """

    pivot = results.pivot(
        index="domain",
        columns="model",
        values="mean_confidence",
    )

    models = _ordered_models(
        pivot.columns
    )

    pivot = pivot[
        models
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for model in models:

        ax.plot(
            pivot.index,
            pivot[model],
            marker="o",
            label=model.upper(),
        )

    ax.set_title(
        "Mean Prediction Confidence"
    )

    ax.set_ylabel(
        "Mean Confidence"
    )

    ax.set_xlabel(
        "Domain"
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    ax.legend()

    ax.grid(
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "mean_confidence.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def generate_domain_shift_plots(
    results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Gera todas as evidências gráficas.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_domain_accuracy(
        results,
        output_dir,
    )

    plot_accuracy_drop(
        results,
        output_dir,
    )

    plot_mean_confidence(
        results,
        output_dir,
    )

    print(
        "[OK] Gráficos de domain shift gerados."
    )