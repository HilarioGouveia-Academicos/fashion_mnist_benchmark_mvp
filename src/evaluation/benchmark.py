from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow import keras

from config.settings import (
    ARTIFACTS_DIR,
    EXPERIMENT_NAME,
    MODELS_DIR,
    MLFLOW_TRACKING_URI,
)

from src.data.loader import load_fashion_mnist
from src.preprocessing.svm import preprocess_svm
from src.preprocessing.mlp import preprocess_mlp
from src.preprocessing.cnn import preprocess_cnn


# ---------------------------------------------------------
# MLflow
# ---------------------------------------------------------

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate(model_name, model, x, y):

    if model_name == "svm":
        predictions = model.predict(x)

    else:
        probabilities = model.predict(
            x,
            verbose=0,
        )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

    return {
        "accuracy": accuracy_score(
            y,
            predictions,
        ),

        "precision_macro": precision_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        ),

        "recall_macro": recall_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        ),

        "f1_macro": f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


# ---------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------

def main():

    print("\n=== Fashion-MNIST Benchmark ===\n")

    _, (x_test, y_test) = load_fashion_mnist()

    entries = [
        {
            "name": "svm",
            "path": MODELS_DIR / "svm.joblib",
            "loader": joblib.load,
            "preprocessor": preprocess_svm,
        },
        {
            "name": "mlp",
            "path": MODELS_DIR / "mlp.keras",
            "loader": keras.models.load_model,
            "preprocessor": preprocess_mlp,
        },
        {
            "name": "cnn",
            "path": MODELS_DIR / "cnn.keras",
            "loader": keras.models.load_model,
            "preprocessor": preprocess_cnn,
        },
    ]

    results = []

    for entry in entries:

        name = entry["name"]
        path = entry["path"]

        if not path.exists():
            print(f"[SKIP] Modelo não encontrado: {path}")
            continue

        print(f"[INFO] Avaliando {name.upper()}...")

        model = entry["loader"](path)

        x_eval = entry["preprocessor"](
            x_test
        )

        metrics = evaluate(
            name,
            model,
            x_eval,
            y_test,
        )

        results.append({
            "model": name,
            **metrics,
        })

        with mlflow.start_run(
            run_name=f"benchmark_{name}"
        ):

            mlflow.log_param(
                "model",
                name,
            )

            mlflow.log_param(
                "evaluation_dataset",
                "fashion_mnist_test",
            )

            mlflow.log_metrics(
                metrics
            )

            mlflow.set_tag(
                "stage",
                "clean_benchmark",
            )

        print(
            f"  Accuracy : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"  F1 Macro : "
            f"{metrics['f1_macro']:.4f}"
        )

    # -----------------------------------------------------
    # Save benchmark
    # -----------------------------------------------------

    if not results:
        print(
            "\n[ERROR] Nenhum modelo disponível."
        )
        return

    df = pd.DataFrame(results)

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        ARTIFACTS_DIR
        / "benchmark.csv"
    )

    df.to_csv(
        output,
        index=False,
    )

    print("\n=== Resultado ===\n")

    print(
        df.to_string(
            index=False
        )
    )

    print(
        f"\n[SAVED] {output}"
    )


if __name__ == "__main__":
    main()