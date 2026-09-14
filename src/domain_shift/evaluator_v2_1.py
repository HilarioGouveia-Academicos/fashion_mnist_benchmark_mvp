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

from src.preprocessing.svm import (
    preprocess_svm,
)

from src.preprocessing.mlp import (
    preprocess_mlp,
)

from src.preprocessing.cnn import (
    preprocess_cnn,
)

from src.domain_shift.external_dataset import (
    load_external_dataset,
    load_canonicalized_dataset,
)

from src.domain_shift.plots_v2_1 import (
    generate_v2_1_plots,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

EXTERNAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
)

RAW_RESULTS_PATH = (
    ARTIFACTS_DIR
    / "domain_shift_v2"
    / "external_results.csv"
)

OUTPUT_DIR = (
    ARTIFACTS_DIR
    / "domain_shift_v2_1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_models() -> dict:

    return {
        "svm": joblib.load(
            MODELS_DIR
            / "svm.joblib"
        ),

        "mlp": (
            keras.models.load_model(
                MODELS_DIR
                / "mlp.keras"
            )
        ),

        "cnn": (
            keras.models.load_model(
                MODELS_DIR
                / "cnn.keras"
            )
        ),
    }


def preprocess_for_model(
    model_name: str,
    images: np.ndarray,
) -> np.ndarray:

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
        model_name
    )


def predict_probabilities(
    model_name: str,
    model,
    x: np.ndarray,
) -> np.ndarray:

    if model_name == "svm":

        return model.predict_proba(
            x
        )

    return model.predict(
        x,
        verbose=0,
    )


def calculate_entropy(
    probabilities: np.ndarray,
) -> np.ndarray:

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    return -np.sum(
        probabilities
        * np.log(probabilities),
        axis=1,
    )


def evaluate(
    model_name: str,
    model,
    images: np.ndarray,
    labels: np.ndarray,
) -> dict:

    x = preprocess_for_model(
        model_name,
        images,
    )

    probabilities = (
        predict_probabilities(
            model_name,
            model,
            x,
        )
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    confidence = np.max(
        probabilities,
        axis=1,
    )

    entropy = calculate_entropy(
        probabilities
    )

    errors = (
        predictions != labels
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    if np.any(errors):

        error_confidence = float(
            confidence[
                errors
            ].mean()
        )

    else:

        error_confidence = 0.0

    return {
        "accuracy": float(
            accuracy
        ),
        "f1_macro": float(
            f1
        ),
        "mean_confidence": float(
            confidence.mean()
        ),
        "confidence_on_errors": (
            error_confidence
        ),
        "mean_entropy": float(
            entropy.mean()
        ),
        "predictions": predictions,
    }


def main() -> None:

    print(
        "\n"
        "========================================\n"
        "DOMAIN SHIFT V2A.1\n"
        "CANONICALIZED EXTERNAL\n"
        "========================================"
    )

    (
        raw_images,
        labels,
        metadata,
    ) = load_external_dataset(
        EXTERNAL_DIR
    )

    canonical_images = (
        load_canonicalized_dataset(
            metadata
        )
    )

    print(
        f"\nRAW shape: "
        f"{raw_images.shape}"
    )

    print(
        f"Canonical shape: "
        f"{canonical_images.shape}"
    )

    print(
        f"Canonical min/max: "
        f"{canonical_images.min()} / "
        f"{canonical_images.max()}"
    )

    if not RAW_RESULTS_PATH.exists():

        raise FileNotFoundError(
            "Execute primeiro o V2A Raw: "
            "python -m src.domain_shift.evaluator_v2"
        )

    raw_results = pd.read_csv(
        RAW_RESULTS_PATH
    )

    models = load_models()

    canonical_rows = []

    prediction_rows = []

    for (
        model_name,
        model,
    ) in models.items():

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

        metrics = evaluate(
            model_name,
            model,
            canonical_images,
            labels,
        )

        canonical_rows.append({
            "model": model_name,
            "accuracy": (
                metrics["accuracy"]
            ),
            "f1_macro": (
                metrics["f1_macro"]
            ),
            "mean_confidence": (
                metrics[
                    "mean_confidence"
                ]
            ),
            "confidence_on_errors": (
                metrics[
                    "confidence_on_errors"
                ]
            ),
            "mean_entropy": (
                metrics[
                    "mean_entropy"
                ]
            ),
        })

        for index, prediction in enumerate(
            metrics["predictions"]
        ):

            prediction_rows.append({
                "model": model_name,
                "file": (
                    metadata.iloc[
                        index
                    ]["file"]
                ),
                "true_class": int(
                    labels[index]
                ),
                "predicted_class": int(
                    prediction
                ),
                "correct": bool(
                    prediction
                    == labels[index]
                ),
            })

        print(
            f"accuracy             = "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"f1_macro             = "
            f"{metrics['f1_macro']:.4f}"
        )

        print(
            f"mean_confidence      = "
            f"{metrics['mean_confidence']:.4f}"
        )

        print(
            f"confidence_on_errors = "
            f"{metrics['confidence_on_errors']:.4f}"
        )

        print(
            f"mean_entropy         = "
            f"{metrics['mean_entropy']:.4f}"
        )

    canonical_results = pd.DataFrame(
        canonical_rows
    )

    predictions = pd.DataFrame(
        prediction_rows
    )

    comparison = (
        raw_results[
            [
                "model",
                "accuracy",
                "f1_macro",
                "mean_confidence",
                "confidence_on_errors",
                "mean_entropy",
            ]
        ]
        .merge(
            canonical_results,
            on="model",
            suffixes=(
                "_raw",
                "_canonical",
            ),
        )
    )

    comparison = comparison.rename(
        columns={
            "accuracy_raw":
                "raw_accuracy",

            "accuracy_canonical":
                "canonical_accuracy",

            "f1_macro_raw":
                "raw_f1",

            "f1_macro_canonical":
                "canonical_f1",

            "mean_confidence_raw":
                "raw_confidence",

            "mean_confidence_canonical":
                "canonical_confidence",

            "confidence_on_errors_raw":
                "raw_error_confidence",

            "confidence_on_errors_canonical":
                "canonical_error_confidence",

            "mean_entropy_raw":
                "raw_entropy",

            "mean_entropy_canonical":
                "canonical_entropy",
        }
    )

    comparison[
        "accuracy_gain"
    ] = (
        comparison[
            "canonical_accuracy"
        ]
        - comparison[
            "raw_accuracy"
        ]
    )

    comparison[
        "f1_gain"
    ] = (
        comparison[
            "canonical_f1"
        ]
        - comparison[
            "raw_f1"
        ]
    )

    canonical_results.to_csv(
        OUTPUT_DIR
        / "canonical_results.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "canonical_predictions.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "raw_vs_canonical.csv",
        index=False,
    )

    generate_v2_1_plots(
        comparison,
        OUTPUT_DIR,
    )

    print(
        "\n"
        "========================================\n"
        "RAW VS CANONICALIZED\n"
        "========================================"
    )

    columns = [
        "model",
        "raw_accuracy",
        "canonical_accuracy",
        "accuracy_gain",
        "raw_error_confidence",
        "canonical_error_confidence",
    ]

    print(
        comparison[
            columns
        ].to_string(
            index=False
        )
    )

    print(
        "\n[OK] Domain Shift V2A.1 concluído."
    )


if __name__ == "__main__":
    main()