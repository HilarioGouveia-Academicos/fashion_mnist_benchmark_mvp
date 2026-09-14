from pathlib import Path

import pandas as pd


def pct(
    value: float,
) -> str:

    return (
        f"{value * 100:.2f}%"
    )


def generate_v2_report(
    results: pd.DataFrame,
    class_results: pd.DataFrame,
    output_dir: Path,
) -> Path:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        output_dir
        / "domain_shift_v2_report.md"
    )

    ranking = (
        results
        .sort_values(
            "accuracy",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    lines = [
        "# Fashion-MNIST Domain Shift Benchmark",
        "",
        "## Domain Shift V2 — External Images",
        "",
        (
            "Este experimento avalia modelos treinados "
            "exclusivamente em Fashion-MNIST utilizando "
            "imagens externas não utilizadas durante "
            "treinamento ou validação."
        ),
        "",
        "## 1. Global Ranking",
        "",
        (
            "| Rank | Model | Accuracy | F1 Macro | "
            "Mean Confidence | Confidence on Errors | "
            "Entropy |"
        ),
        "|---:|---|---:|---:|---:|---:|---:|",
    ]

    for index, row in (
        ranking.iterrows()
    ):

        lines.append(
            "| "
            f"{index + 1} | "
            f"{row['model'].upper()} | "
            f"{pct(row['accuracy'])} | "
            f"{pct(row['f1_macro'])} | "
            f"{pct(row['mean_confidence'])} | "
            f"{pct(row['confidence_on_errors'])} | "
            f"{row['mean_entropy']:.4f} |"
        )

    lines.extend([
        "",
        "## 2. Per-Class Recall",
        "",
        "| Model | Class | Recall |",
        "|---|---|---:|",
    ])

    for _, row in (
        class_results.iterrows()
    ):

        lines.append(
            "| "
            f"{row['model'].upper()} | "
            f"{row['class_name']} | "
            f"{pct(row['recall'])} |"
        )

    best = ranking.iloc[0]

    lines.extend([
        "",
        "## 3. Main Finding",
        "",
        (
            f"**{best['model'].upper()}** apresentou "
            f"o melhor desempenho no domínio externo, "
            f"com accuracy de {pct(best['accuracy'])}."
        ),
        "",
        (
            "Resultados em imagens externas devem ser "
            "interpretados em conjunto com confiança e "
            "entropia. Alta confiança combinada com baixa "
            "accuracy representa comportamento potencialmente "
            "arriscado em produção."
        ),
        "",
        "## 4. Evidence",
        "",
        "- `external_results.csv`",
        "- `external_predictions.csv`",
        "- `external_class_results.csv`",
        "- `external_model_comparison.png`",
        "- `accuracy_vs_confidence.png`",
        "- `svm_confusion_matrix.png`",
        "- `mlp_confusion_matrix.png`",
        "- `cnn_confusion_matrix.png`",
        "",
    ])

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"[OK] Relatório V2 gerado: "
        f"{report_path}"
    )

    return report_path