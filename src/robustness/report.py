from pathlib import Path

import pandas as pd


def _percentage(
    value: float,
) -> str:
    """
    Formata valor decimal como percentual.
    """

    return f"{value * 100:.2f}%"


def get_best_model_by_corruption(
    corruption_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Seleciona o modelo com maior accuracy média
    para cada corrupção.
    """

    indexes = (
        corruption_summary
        .groupby(
            "corruption"
        )[
            "mean_accuracy"
        ]
        .idxmax()
    )

    return (
        corruption_summary
        .loc[indexes]
        .sort_values(
            "corruption"
        )
        .reset_index(
            drop=True
        )
    )


def get_worst_scenario_by_model(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Retorna o pior cenário observado
    para cada modelo.
    """

    corrupted = results[
        results["corruption"]
        != "clean"
    ]

    indexes = (
        corrupted
        .groupby("model")[
            "accuracy"
        ]
        .idxmin()
    )

    return (
        corrupted
        .loc[indexes]
        [
            [
                "model",
                "corruption",
                "severity",
                "accuracy",
                "f1_macro",
                "accuracy_drop",
            ]
        ]
        .sort_values(
            "model"
        )
        .reset_index(
            drop=True
        )
    )


def get_clean_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Retorna os resultados clean.
    """

    return (
        results[
            results["corruption"]
            == "clean"
        ]
        [
            [
                "model",
                "accuracy",
                "f1_macro",
            ]
        ]
        .sort_values(
            "accuracy",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def generate_robustness_report(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    corruption_summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Gera relatório Markdown consolidado.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        output_dir
        / "robustness_report.md"
    )

    clean = get_clean_results(
        results
    )

    best_by_corruption = (
        get_best_model_by_corruption(
            corruption_summary
        )
    )

    worst_cases = (
        get_worst_scenario_by_model(
            results
        )
    )

    global_ranking = (
        summary
        .sort_values(
            "mean_accuracy_drop"
        )
        .reset_index(
            drop=True
        )
    )

    lines = []

    # =====================================================
    # Header
    # =====================================================

    lines.extend([
        "# Fashion-MNIST Robustness Benchmark",
        "",
        "## Stress Test V2",
        "",
        (
            "Este relatório consolida o desempenho "
            "dos modelos SVM, MLP e CNN sob diferentes "
            "tipos e níveis de corrupção."
        ),
        "",
    ])

    # =====================================================
    # Clean benchmark
    # =====================================================

    lines.extend([
        "## 1. Clean Benchmark",
        "",
        "| Model | Accuracy | F1 Macro |",
        "|---|---:|---:|",
    ])

    for _, row in clean.iterrows():

        lines.append(
            "| "
            f"{row['model'].upper()} | "
            f"{_percentage(row['accuracy'])} | "
            f"{_percentage(row['f1_macro'])} |"
        )

    lines.append("")

    # =====================================================
    # Global robustness
    # =====================================================

    lines.extend([
        "## 2. Global Robustness Ranking",
        "",
        (
            "Menor `Mean Accuracy Drop` representa "
            "maior robustez média."
        ),
        "",
        (
            "| Rank | Model | Mean Corrupted Accuracy | "
            "Mean Accuracy Drop | Worst Accuracy |"
        ),
        "|---:|---|---:|---:|---:|",
    ])

    for index, row in (
        global_ranking.iterrows()
    ):

        lines.append(
            "| "
            f"{index + 1} | "
            f"{row['model'].upper()} | "
            f"{_percentage(row['mean_corrupted_accuracy'])} | "
            f"{_percentage(row['mean_accuracy_drop'])} | "
            f"{_percentage(row['min_corrupted_accuracy'])} |"
        )

    lines.append("")

    # =====================================================
    # Best model by corruption
    # =====================================================

    lines.extend([
        "## 3. Best Model by Corruption",
        "",
        (
            "| Corruption | Best Model | "
            "Mean Accuracy | Mean Accuracy Drop |"
        ),
        "|---|---|---:|---:|",
    ])

    for _, row in (
        best_by_corruption.iterrows()
    ):

        lines.append(
            "| "
            f"{row['corruption']} | "
            f"{row['model'].upper()} | "
            f"{_percentage(row['mean_accuracy'])} | "
            f"{_percentage(row['mean_accuracy_drop'])} |"
        )

    lines.append("")

    # =====================================================
    # Worst case
    # =====================================================

    lines.extend([
        "## 4. Worst Case by Model",
        "",
        (
            "| Model | Corruption | Severity | "
            "Accuracy | F1 Macro | Accuracy Drop |"
        ),
        "|---|---|---:|---:|---:|---:|",
    ])

    for _, row in (
        worst_cases.iterrows()
    ):

        lines.append(
            "| "
            f"{row['model'].upper()} | "
            f"{row['corruption']} | "
            f"{row['severity']} | "
            f"{_percentage(row['accuracy'])} | "
            f"{_percentage(row['f1_macro'])} | "
            f"{_percentage(row['accuracy_drop'])} |"
        )

    lines.append("")

    # =====================================================
    # Findings
    # =====================================================

    best_global = (
        global_ranking.iloc[0]
    )

    worst_global = (
        global_ranking.iloc[-1]
    )

    lines.extend([
        "## 5. Main Findings",
        "",
        (
            f"- **{best_global['model'].upper()}** apresentou "
            f"a menor queda média de accuracy "
            f"({_percentage(best_global['mean_accuracy_drop'])})."
        ),
        (
            f"- **{worst_global['model'].upper()}** apresentou "
            f"a maior queda média de accuracy "
            f"({_percentage(worst_global['mean_accuracy_drop'])})."
        ),
        (
            "- O comportamento varia conforme o tipo de corrupção; "
            "portanto, robustez não deve ser avaliada apenas "
            "pela accuracy clean."
        ),
        (
            "- As curvas de severidade permitem observar "
            "o ponto em que cada arquitetura começa a "
            "degradar significativamente."
        ),
        "",
    ])

    # =====================================================
    # Artifacts
    # =====================================================

    lines.extend([
        "## 6. Generated Visual Evidence",
        "",
        "- `curves/noise_accuracy_curve.png`",
        "- `curves/rotation_accuracy_curve.png`",
        "- `curves/blur_accuracy_curve.png`",
        "- `curves/contrast_accuracy_curve.png`",
        "- `curves/brightness_accuracy_curve.png`",
        "- `curves/occlusion_accuracy_curve.png`",
        "- `robustness_heatmap.png`",
        "- `clean_vs_worst_case.png`",
        "- `mean_accuracy_drop.png`",
        "",
    ])

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"[OK] Relatório gerado: "
        f"{report_path}"
    )

    return report_path