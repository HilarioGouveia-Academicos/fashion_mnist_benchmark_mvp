import json

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tensorflow import keras

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
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
    CLASS_NAMES,
    load_external_dataset,
)

from src.domain_shift.plots_v2 import (
    generate_v2_plots,
)

from src.domain_shift.report_v2 import (
    generate_v2_report,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

EXTERNAL_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
)

OUTPUT_DIR = (
    ARTIFACTS_DIR
    / "domain_shift_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_models() -> dict:

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
            f"[OK] SVM: {svm_path}"
        )

    if mlp_path.exists():

        models["mlp"] = (
            keras.models.load_model(
                mlp_path
            )
        )

        print(
            f"[OK] MLP: {mlp_path}"
        )

    if cnn_path.exists():

        models["cnn"] = (
            keras.models.load_model(
                cnn_path
            )
        )

        print(
            f"[OK] CNN: {cnn_path}"
        )

    return models


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
        f"Modelo inválido: "
        f"{model_name}"
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
        * np.log(
            probabilities
        ),
        axis=1,
    )


def evaluate_model(
    model_name: str,
    model,
    images: np.ndarray,
    y_true: np.ndarray,
) -> tuple[
    dict,
    np.ndarray,
    np.ndarray,
]:

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

    y_pred = np.argmax(
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
        y_pred != y_true
    )

    correct = (
        ~errors
    )

    metrics = {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "mean_confidence": float(
            confidence.mean()
        ),
        "mean_entropy": float(
            entropy.mean()
        ),
    }

    if np.any(errors):

        metrics[
            "confidence_on_errors"
        ] = float(
            confidence[
                errors
            ].mean()
        )

    else:

        metrics[
            "confidence_on_errors"
        ] = 0.0

    if np.any(correct):

        metrics[
            "confidence_on_correct"
        ] = float(
            confidence[
                correct
            ].mean()
        )

    else:

        metrics[
            "confidence_on_correct"
        ] = 0.0

    return (
        metrics,
        y_pred,
        probabilities,
    )


def calculate_per_class_results(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict]:

    recalls = recall_score(
        y_true,
        y_pred,
        labels=list(
            CLASS_NAMES.keys()
        ),
        average=None,
        zero_division=0,
    )

    rows = []

    for class_id, recall in enumerate(
        recalls
    ):

        rows.append(
            {
                "model": (
                    model_name
                ),
                "class_id": (
                    class_id
                ),
                "class_name": (
                    CLASS_NAMES[
                        class_id
                    ]
                ),
                "recall": float(
                    recall
                ),
            }
        )

    return rows


def main() -> None:

    print(
        "\n"
        "========================================\n"
        "FASHION-MNIST DOMAIN SHIFT V2\n"
        "EXTERNAL IMAGE BENCHMARK\n"
        "========================================"
    )

    (
        images,
        labels,
        metadata,
    ) = load_external_dataset(
        EXTERNAL_DATA_DIR
    )

    print(
        f"\nImagens externas: "
        f"{images.shape}"
    )

    print(
        f"Classes presentes: "
        f"{len(np.unique(labels))}"
    )

    print(
        "\nDistribuição:"
    )

    distribution = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
    )

    for (
        class_id,
        count,
    ) in distribution.items():

        print(
            f"{class_id} - "
            f"{CLASS_NAMES[class_id]:<12} "
            f": {count}"
        )

    models = load_models()

    results_rows = []

    prediction_rows = []

    class_rows = []

    prediction_data = {}

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

        (
            metrics,
            y_pred,
            probabilities,
        ) = evaluate_model(
            model_name,
            model,
            images,
            labels,
        )

        results_rows.append(
            {
                "model": (
                    model_name
                ),
                **metrics,
            }
        )

        confidence = np.max(
            probabilities,
            axis=1,
        )

        entropy = (
            calculate_entropy(
                probabilities
            )
        )

        for index in range(
            len(labels)
        ):

            prediction_rows.append(
                {
                    "model": (
                        model_name
                    ),
                    "file": (
                        metadata.iloc[
                            index
                        ]["file"]
                    ),
                    "true_class": int(
                        labels[index]
                    ),
                    "true_name": (
                        CLASS_NAMES[
                            int(
                                labels[index]
                            )
                        ]
                    ),
                    "predicted_class": int(
                        y_pred[index]
                    ),
                    "predicted_name": (
                        CLASS_NAMES[
                            int(
                                y_pred[index]
                            )
                        ]
                    ),
                    "correct": bool(
                        y_pred[index]
                        == labels[index]
                    ),
                    "confidence": float(
                        confidence[
                            index
                        ]
                    ),
                    "entropy": float(
                        entropy[
                            index
                        ]
                    ),
                }
            )

        class_rows.extend(
            calculate_per_class_results(
                model_name,
                labels,
                y_pred,
            )
        )

        prediction_data[
            model_name
        ] = {
            "y_true": labels,
            "y_pred": y_pred,
        }

        print(
            f"accuracy              = "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"f1_macro              = "
            f"{metrics['f1_macro']:.4f}"
        )

        print(
            f"mean_confidence       = "
            f"{metrics['mean_confidence']:.4f}"
        )

        print(
            f"confidence_on_errors  = "
            f"{metrics['confidence_on_errors']:.4f}"
        )

        print(
            f"mean_entropy          = "
            f"{metrics['mean_entropy']:.4f}"
        )

    results = pd.DataFrame(
        results_rows
    )

    predictions = pd.DataFrame(
        prediction_rows
    )

    class_results = pd.DataFrame(
        class_rows
    )

    results_csv = (
        OUTPUT_DIR
        / "external_results.csv"
    )

    predictions_csv = (
        OUTPUT_DIR
        / "external_predictions.csv"
    )

    class_csv = (
        OUTPUT_DIR
        / "external_class_results.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "external_results.json"
    )

    results.to_csv(
        results_csv,
        index=False,
    )

    predictions.to_csv(
        predictions_csv,
        index=False,
    )

    class_results.to_csv(
        class_csv,
        index=False,
    )

    json_path.write_text(
        json.dumps(
            results.to_dict(
                orient="records"
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    generate_v2_plots(
        results=results,
        predictions=prediction_data,
        class_names=[
            CLASS_NAMES[index]
            for index in range(10)
        ],
        output_dir=OUTPUT_DIR,
    )

    generate_v2_report(
        results=results,
        class_results=class_results,
        output_dir=OUTPUT_DIR,
    )

    print(
        "\n"
        "========================================\n"
        "DOMAIN SHIFT V2 SUMMARY\n"
        "========================================"
    )

    print(
        results
        .sort_values(
            "accuracy",
            ascending=False,
        )
        .to_string(
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
        f"- {predictions_csv}"
    )

    print(
        f"- {class_csv}"
    )

    print(
        f"- {json_path}"
    )

    print(
        f"- "
        f"{OUTPUT_DIR / 'domain_shift_v2_report.md'}"
    )


if __name__ == "__main__":
    main()