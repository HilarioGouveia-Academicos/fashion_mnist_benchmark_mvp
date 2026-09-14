from pathlib import Path

import pandas as pd


def _percentage(
    value: float,
) -> str:

    return (
        f"{value * 100:.2f}%"
    )


def generate_domain_shift_report(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Gera relatório Markdown do Domain Shift V1.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        output_dir
        / "domain_shift_report.md"
    )

    ranking = (
        summary
        .sort_values(
            "mean_accuracy_drop"
        )
        .reset_index(
            drop=True
        )
    )

    clean = (
        results[
            results["domain"]
            == "clean"
        ]
        .sort_values(
            "accuracy",
            ascending=False,
        )
    )

    shifted = results[
        results["domain"]
        != "clean"
    ]

    worst_indexes = (
        shifted
        .groupby("model")[
            "accuracy"
        ]
        .idxmin()
    )

    worst = (
        shifted
        .loc[worst_indexes]
        .sort_values("model")
    )

    lines = [
        "# Fashion-MNIST Domain Shift Benchmark",
        "",
        "## Domain Shift V1",
        "",
        (
            "Este experimento mede a capacidade de "
            "generalização dos modelos SVM, MLP e CNN "
            "quando a distribuição visual de entrada "
            "difere do Fashion-MNIST original."
        ),
        "",
        "## 1. Clean Benchmark",
        "",
        "| Model | Accuracy | F1 Macro | Mean Confidence |",
        "|---|---:|---:|---:|",
    ]

    for _, row in clean.iterrows():

        lines.append(
            "| "
            f"{row['model'].upper()} | "
            f"{_percentage(row['accuracy'])} | "
            f"{_percentage(row['f1_macro'])} | "
            f"{_percentage(row['mean_confidence'])} |"
        )

    lines.extend([
        "",
        "## 2. Domain Shift Ranking",
        "",
        "| Rank | Model | Mean Shifted Accuracy | Mean Accuracy Drop | Mean Confidence |",
        "|---:|---|---:|---:|---:|",
    ])

    for index, row in ranking.iterrows():

        lines.append(
            "| "
            f"{index + 1} | "
            f"{row['model'].upper()} | "
            f"{_percentage(row['mean_shifted_accuracy'])} | "
            f"{_percentage(row['mean_accuracy_drop'])} | "
            f"{_percentage(row['mean_confidence'])} |"
        )

    lines.extend([
        "",
        "## 3. Worst Domain by Model",
        "",
        "| Model | Domain | Accuracy | F1 Macro | Confidence | Accuracy Drop |",
        "|---|---|---:|---:|---:|---:|",
    ])

    for _, row in worst.iterrows():

        lines.append(
            "| "
            f"{row['model'].upper()} | "
            f"{row['domain']} | "
            f"{_percentage(row['accuracy'])} | "
            f"{_percentage(row['f1_macro'])} | "
            f"{_percentage(row['mean_confidence'])} | "
            f"{_percentage(row['accuracy_drop'])} |"
        )

    best = ranking.iloc[0]

    lines.extend([
        "",
        "## 4. Main Finding",
        "",
        (
            f"**{best['model'].upper()}** apresentou "
            f"a menor queda média de accuracy sob "
            f"domain shift: "
            f"{_percentage(best['mean_accuracy_drop'])}."
        ),
        "",
        (
            "Além da accuracy, a confiança das previsões "
            "deve ser observada. Um modelo que apresenta "
            "baixa accuracy e alta confiança sob domain shift "
            "representa maior risco operacional."
        ),
        "",
        "## 5. Generated Evidence",
        "",
        "- `domain_shift_results.csv`",
        "- `domain_shift_summary.csv`",
        "- `domain_shift_results.json`",
        "- `domain_accuracy.png`",
        "- `domain_accuracy_drop.png`",
        "- `mean_confidence.png`",
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