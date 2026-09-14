from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["svm", "mlp", "cnn"]


def _ensure_output_dir(
    output_dir: Path,
) -> Path:
    """
    Garante que o diretório de saída exista.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def _ordered_models(
    values,
) -> list[str]:
    """
    Mantém ordem padrão SVM -> MLP -> CNN.
    """

    available = set(values)

    return [
        model
        for model in MODEL_ORDER
        if model in available
    ]


def plot_severity_curves(
    results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Gera uma curva de accuracy x severity
    para cada tipo de corrupção.
    """

    curves_dir = (
        output_dir / "curves"
    )

    curves_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    corrupted = results[
        results["corruption"] != "clean"
    ]

    corruptions = (
        corrupted["corruption"]
        .drop_duplicates()
        .tolist()
    )

    models = _ordered_models(
        results["model"].unique()
    )

    for corruption in corruptions:

        subset = corrupted[
            corrupted["corruption"]
            == corruption
        ]

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        for model in models:

            model_data = subset[
                subset["model"]
                == model
            ].sort_values(
                "severity"
            )

            if model_data.empty:
                continue

            ax.plot(
                model_data["severity"],
                model_data["accuracy"],
                marker="o",
                label=model.upper(),
            )

        ax.set_title(
            f"Robustez - {corruption.title()}"
        )

        ax.set_xlabel(
            "Severity"
        )

        ax.set_ylabel(
            "Accuracy"
        )

        ax.set_ylim(
            0.0,
            1.0,
        )

        ax.grid(
            alpha=0.25
        )

        ax.legend()

        fig.tight_layout()

        path = (
            curves_dir
            / f"{corruption}_accuracy_curve.png"
        )

        fig.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)


def plot_mean_accuracy_drop(
    results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Compara a queda média de accuracy
    entre os modelos.
    """

    corrupted = results[
        results["corruption"] != "clean"
    ]

    summary = (
        corrupted
        .groupby("model")[
            "accuracy_drop"
        ]
        .mean()
        .reindex(
            _ordered_models(
                corrupted[
                    "model"
                ].unique()
            )
        )
        .dropna()
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.bar(
        [
            model.upper()
            for model in summary.index
        ],
        summary.values,
    )

    ax.set_title(
        "Mean Accuracy Drop"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "Mean Accuracy Drop"
    )

    ax.set_ylim(
        0.0,
        max(
            0.5,
            float(
                summary.max()
                * 1.15
            ),
        ),
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "mean_accuracy_drop.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_clean_vs_worst_case(
    results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Compara accuracy clean com o pior cenário
    observado para cada modelo.
    """

    clean = (
        results[
            results["corruption"]
            == "clean"
        ]
        .set_index("model")[
            "accuracy"
        ]
    )

    corrupted = results[
        results["corruption"]
        != "clean"
    ]

    worst = (
        corrupted
        .groupby("model")[
            "accuracy"
        ]
        .min()
    )

    models = _ordered_models(
        clean.index
    )

    clean_values = [
        clean[model]
        for model in models
    ]

    worst_values = [
        worst[model]
        for model in models
    ]

    x = np.arange(
        len(models)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        x - width / 2,
        clean_values,
        width,
        label="Clean",
    )

    ax.bar(
        x + width / 2,
        worst_values,
        width,
        label="Worst Case",
    )

    ax.set_title(
        "Clean vs Worst Case"
    )

    ax.set_ylabel(
        "Accuracy"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            model.upper()
            for model in models
        ]
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "clean_vs_worst_case.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_robustness_heatmap(
    results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Gera heatmap contendo accuracy média
    por modelo e tipo de corrupção.
    """

    corrupted = results[
        results["corruption"]
        != "clean"
    ]

    pivot = (
        corrupted
        .groupby([
            "model",
            "corruption",
        ])[
            "accuracy"
        ]
        .mean()
        .unstack()
    )

    models = _ordered_models(
        pivot.index
    )

    pivot = pivot.reindex(
        models
    )

    matrix = pivot.values

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title(
        "Mean Accuracy by Corruption"
    )

    ax.set_xlabel(
        "Corruption"
    )

    ax.set_ylabel(
        "Model"
    )

    ax.set_xticks(
        np.arange(
            len(
                pivot.columns
            )
        )
    )

    ax.set_xticklabels(
        pivot.columns,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        np.arange(
            len(
                pivot.index
            )
        )
    )

    ax.set_yticklabels(
        [
            model.upper()
            for model in pivot.index
        ]
    )

    for row in range(
        matrix.shape[0]
    ):

        for col in range(
            matrix.shape[1]
        ):

            value = (
                matrix[
                    row,
                    col,
                ]
            )

            ax.text(
                col,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Accuracy",
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "robustness_heatmap.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def generate_robustness_plots(
    results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Gera todos os gráficos do benchmark
    de robustez.
    """

    output_dir = (
        _ensure_output_dir(
            output_dir
        )
    )

    plot_severity_curves(
        results,
        output_dir,
    )

    plot_mean_accuracy_drop(
        results,
        output_dir,
    )

    plot_clean_vs_worst_case(
        results,
        output_dir,
    )

    plot_robustness_heatmap(
        results,
        output_dir,
    )

    print(
        "\n[OK] Gráficos de robustez gerados."
    )