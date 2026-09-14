import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tensorflow import keras

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from config.settings import (
    ARTIFACTS_DIR,
    MODELS_DIR,
)

from src.data.loader import (
    load_fashion_mnist,
)

from src.preprocessing.svm import (
    preprocess_svm,
)

from src.preprocessing.mlp import (
    preprocess_mlp,
)

from src.preprocessing.cnn import (
    preprocess_cnn,
)

from src.domain_shift.datasets import (
    generate_domain_shift_scenarios,
)

from src.domain_shift.plots import (
    generate_domain_shift_plots,
)

from src.domain_shift.report import (
    generate_domain_shift_report,
)


DOMAIN_SHIFT_DIR = (
    ARTIFACTS_DIR
    / "domain_shift"
)

DOMAIN_SHIFT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_models() -> dict:
    """
    Carrega os modelos persistidos.
    """

    models = {}

    svm_path = (
        MODELS_DIR
        / "svm.joblib"
    )

    mlp_path = (
        MODELS_DIR
        / "mlp.keras"
    )

    cnn_path = (
        MODELS_DIR
        / "cnn.keras"
    )

    if svm_path.exists():

        models["svm"] = (
            joblib.load(
                svm_path
            )
        )

        print(
            f"[OK] SVM carregado: "
            f"{svm_path}"
        )

    else:

        print(
            f"[WARNING] SVM não encontrado: "
            f"{svm_path}"
        )

    if mlp_path.exists():

        models["mlp"] = (
            keras.models.load_model(
                mlp_path
            )
        )

        print(
            f"[OK] MLP carregado: "
            f"{mlp_path}"
        )

    else:

        print(
            f"[WARNING] MLP não encontrado: "
            f"{mlp_path}"
        )

    if cnn_path.exists():

        models["cnn"] = (
            keras.models.load_model(
                cnn_path
            )
        )

        print(
            f"[OK] CNN carregado: "
            f"{cnn_path}"
        )

    else:

        print(
            f"[WARNING] CNN não encontrado: "
            f"{cnn_path}"
        )

    return models


def preprocess_for_model(
    model_name: str,
    images: np.ndarray,
) -> np.ndarray:
    """
    Aplica exatamente o mesmo preprocessing
    utilizado no treinamento.
    """

    if model_name == "svm":

        return preprocess_svm(
            images
        )

    if model_name == "mlp":

        return preprocess_mlp(
            images
        )

    if model_name == "cnn":

        return preprocess_cnn(
            images
        )

    raise ValueError(
        f"Modelo desconhecido: "
        f"{model_name}"
    )


def prediction_entropy(
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Calcula entropia das probabilidades.

    Maior valor = maior incerteza.
    """

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    return -np.sum(
        probabilities
        * np.log(
            probabilities
        ),
        axis=1,
    )


def predict_with_probabilities(
    model_name: str,
    model,
    x: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Retorna classe predita e probabilidades.
    """

    if model_name == "svm":

        probabilities = (
            model.predict_proba(
                x
            )
        )

    else:

        probabilities = (
            model.predict(
                x,
                verbose=0,
            )
        )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    return (
        predictions,
        probabilities,
    )


def evaluate_domain(
    model_name: str,
    model,
    images: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """
    Avalia modelo em um domínio específico.
    """

    x = preprocess_for_model(
        model_name,
        images,
    )

    predictions, probabilities = (
        predict_with_probabilities(
            model_name,
            model,
            x,
        )
    )

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    f1 = f1_score(
        y_true,
        predictions,
        average="macro",
    )

    confidence = np.max(
        probabilities,
        axis=1,
    )

    entropy = prediction_entropy(
        probabilities
    )

    correct_mask = (
        predictions
        == y_true
    )

    error_mask = (
        ~correct_mask
    )

    mean_confidence = float(
        confidence.mean()
    )

    mean_entropy = float(
        entropy.mean()
    )

    if np.any(error_mask):

        confidence_on_errors = float(
            confidence[
                error_mask
            ].mean()
        )

    else:

        confidence_on_errors = 0.0

    if np.any(correct_mask):

        confidence_on_correct = float(
            confidence[
                correct_mask
            ].mean()
        )

    else:

        confidence_on_correct = 0.0

    return {
        "accuracy": float(
            accuracy
        ),
        "f1_macro": float(
            f1
        ),
        "mean_confidence": (
            mean_confidence
        ),
        "confidence_on_correct": (
            confidence_on_correct
        ),
        "confidence_on_errors": (
            confidence_on_errors
        ),
        "mean_entropy": (
            mean_entropy
        ),
    }


def build_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Consolida métricas por modelo,
    ignorando clean.
    """

    shifted = results[
        results["domain"]
        != "clean"
    ]

    summary = (
        shifted
        .groupby("model")
        .agg(
            mean_shifted_accuracy=(
                "accuracy",
                "mean",
            ),
            min_shifted_accuracy=(
                "accuracy",
                "min",
            ),
            mean_shifted_f1=(
                "f1_macro",
                "mean",
            ),
            mean_accuracy_drop=(
                "accuracy_drop",
                "mean",
            ),
            max_accuracy_drop=(
                "accuracy_drop",
                "max",
            ),
            mean_confidence=(
                "mean_confidence",
                "mean",
            ),
            mean_error_confidence=(
                "confidence_on_errors",
                "mean",
            ),
            mean_entropy=(
                "mean_entropy",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "mean_accuracy_drop"
        )
    )

    return summary


def main() -> None:

    print(
        "\n"
        "========================================\n"
        "FASHION-MNIST DOMAIN SHIFT V1\n"
        "========================================"
    )

    (_, _), (x_test, y_test) = load_fashion_mnist()

    print(
        f"\nTest original: "
        f"{x_test.shape}"
    )

    models = load_models()


    scenarios = (
        generate_domain_shift_scenarios(
            x_test
        )
    )

    print(
        f"\nDomínios gerados: "
        f"{len(scenarios)}"
    )

    print(
        f"Modelos: "
        f"{len(models)}"
    )

    print(
        "Avaliações totais: "
        f"{len(scenarios) * len(models)}"
    )

    results_rows = []

    for model_name, model in (
        models.items()
    ):

        print(
            "\n"
            "========================================"
        )

        print(
            f"Modelo: "
            f"{model_name.upper()}"
        )

        print(
            "========================================"
        )

        clean_metrics = (
            evaluate_domain(
                model_name,
                model,
                x_test,
                y_test,
            )
        )

        clean_accuracy = (
            clean_metrics[
                "accuracy"
            ]
        )

        clean_f1 = (
            clean_metrics[
                "f1_macro"
            ]
        )

        print(
            "\n"
            f"Baseline clean "
            f"| accuracy={clean_accuracy:.4f} "
            f"| f1={clean_f1:.4f}"
        )

        for (
            domain_name,
            domain_images,
        ) in scenarios.items():

            metrics = (
                evaluate_domain(
                    model_name,
                    model,
                    domain_images,
                    y_test,
                )
            )

            accuracy_drop = (
                clean_accuracy
                - metrics["accuracy"]
            )

            f1_drop = (
                clean_f1
                - metrics["f1_macro"]
            )

            row = {
                "model": (
                    model_name
                ),
                "domain": (
                    domain_name
                ),
                **metrics,
                "accuracy_drop": float(
                    accuracy_drop
                ),
                "f1_drop": float(
                    f1_drop
                ),
            }

            results_rows.append(
                row
            )

            print(
                f"{domain_name:<20} "
                f"| acc="
                f"{metrics['accuracy']:.4f} "
                f"| f1="
                f"{metrics['f1_macro']:.4f} "
                f"| confidence="
                f"{metrics['mean_confidence']:.4f} "
                f"| entropy="
                f"{metrics['mean_entropy']:.4f} "
                f"| drop="
                f"{accuracy_drop:.4f}"
            )

    results = pd.DataFrame(
        results_rows
    )

    summary = build_summary(
        results
    )

    # ============================================
    # Persistência
    # ============================================

    results_csv = (
        DOMAIN_SHIFT_DIR
        / "domain_shift_results.csv"
    )

    summary_csv = (
        DOMAIN_SHIFT_DIR
        / "domain_shift_summary.csv"
    )

    results_json = (
        DOMAIN_SHIFT_DIR
        / "domain_shift_results.json"
    )

    results.to_csv(
        results_csv,
        index=False,
    )

    summary.to_csv(
        summary_csv,
        index=False,
    )

    results_json.write_text(
        json.dumps(
            results.to_dict(
                orient="records"
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    # ============================================
    # Visualizações
    # ============================================

    generate_domain_shift_plots(
        results=results,
        output_dir=DOMAIN_SHIFT_DIR,
    )

    # ============================================
    # Relatório
    # ============================================

    generate_domain_shift_report(
        results=results,
        summary=summary,
        output_dir=DOMAIN_SHIFT_DIR,
    )

    print(
        "\n"
        "========================================\n"
        "DOMAIN SHIFT SUMMARY\n"
        "========================================"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nArquivos gerados:"
    )

    print(
        f"- {results_csv}"
    )

    print(
        f"- {summary_csv}"
    )

    print(
        f"- {results_json}"
    )

    print(
        f"- "
        f"{DOMAIN_SHIFT_DIR / 'domain_shift_report.md'}"
    )

    print(
        f"- "
        f"{DOMAIN_SHIFT_DIR / 'domain_accuracy.png'}"
    )

    print(
        f"- "
        f"{DOMAIN_SHIFT_DIR / 'domain_accuracy_drop.png'}"
    )

    print(
        f"- "
        f"{DOMAIN_SHIFT_DIR / 'mean_confidence.png'}"
    )


    # DEBUG: verificar se os arrays são iguais
    print("inverted mean:", scenarios["inverted"].mean())
    print("gray_background mean:", scenarios["gray_background"].mean())
    print("arrays iguais:", np.array_equal(scenarios["inverted"], scenarios["gray_background"], ))


if __name__ == "__main__":
    main()