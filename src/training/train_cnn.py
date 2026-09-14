
from pathlib import Path
import json
import time

import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np

from mlflow.models import infer_signature

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

from config.settings import (
    BATCH_SIZE,
    CNN_EPOCHS,
    EXPERIMENT_NAME,
    MODELS_DIR,
    MLFLOW_TRACKING_URI,
)

from src.data.loader import load_fashion_mnist
from src.models.cnn import build_cnn
from src.preprocessing.cnn import preprocess_cnn


# ---------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

tracking_uri = mlflow.get_tracking_uri()

print(f"[MLflow] Tracking URI: {tracking_uri}")

if tracking_uri.startswith("file:"):
    raise RuntimeError(
        f"MLflow está usando FileStore antigo: {tracking_uri}. "
        "Esperado SQLite."
    )

mlflow.set_experiment(EXPERIMENT_NAME)


# ---------------------------------------------------------
# Directories
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "cnn"

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Training history
# ---------------------------------------------------------

def save_training_history(
    history,
    output_path: Path,
) -> None:

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        history.history["accuracy"],
        label="train_accuracy",
    )

    ax.plot(
        history.history["val_accuracy"],
        label="val_accuracy",
    )

    ax.set_title(
        "CNN - Training History"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    (x_train, y_train), (x_test, y_test) = (
        load_fashion_mnist()
    )

    print(
        f"Train original: {x_train.shape}"
    )

    print(
        f"Test original:  {x_test.shape}"
    )

    # -----------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------

    x_train = preprocess_cnn(x_train)
    x_test = preprocess_cnn(x_test)

    print(
        f"Train preprocessado: {x_train.shape}"
    )

    print(
        f"Test preprocessado:  {x_test.shape}"
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = build_cnn()

    # -----------------------------------------------------
    # MLflow Run
    # -----------------------------------------------------

    with mlflow.start_run(
        run_name="cnn_baseline"
    ) as run:

        run_id = run.info.run_id

        print(
            f"[MLflow] Run ID: {run_id}"
        )

        print(
            f"[MLflow] Artifact URI: "
            f"{mlflow.get_artifact_uri()}"
        )

        # -------------------------------------------------
        # Parameters
        # -------------------------------------------------

        mlflow.log_params({
            "model": "CNN",
            "epochs": CNN_EPOCHS,
            "batch_size": BATCH_SIZE,
            "input_shape": "28x28x1",
            "filters_1": 32,
            "filters_2": 64,
            "dense_units": 128,
            "dropout_rate": 0.30,
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy",
            "normalization": "0_1",
        })

        # -------------------------------------------------
        # Training
        # -------------------------------------------------

        start = time.perf_counter()

        history = model.fit(
            x_train,
            y_train,
            validation_split=0.1,
            epochs=CNN_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
        )

        training_time = (
            time.perf_counter() - start
        )

        # -------------------------------------------------
        # Inference
        # -------------------------------------------------

        start = time.perf_counter()

        probabilities = model.predict(
            x_test,
            batch_size=BATCH_SIZE,
            verbose=0,
        )

        inference_time = (
            time.perf_counter() - start
        )

        y_pred = np.argmax(
            probabilities,
            axis=1,
        )

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        metrics = {
            "accuracy": accuracy_score(
                y_test,
                y_pred,
            ),

            "precision_macro": precision_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            ),

            "recall_macro": recall_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            ),

            "f1_macro": f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            ),

            "final_train_accuracy":
                history.history["accuracy"][-1],

            "final_val_accuracy":
                history.history["val_accuracy"][-1],

            "training_time_seconds":
                training_time,

            "inference_time_seconds":
                inference_time,
        }

        mlflow.log_metrics(metrics)

        print("\nMétricas:")

        for name, value in metrics.items():

            print(
                f"{name}: {value:.4f}"
            )

        # -------------------------------------------------
        # Metrics JSON
        # -------------------------------------------------

        metrics_path = (
            ARTIFACTS_DIR / "metrics.json"
        )

        with open(
            metrics_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4,
            )

        # -------------------------------------------------
        # Classification Report
        # -------------------------------------------------

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        report_path = (
            ARTIFACTS_DIR
            / "classification_report.json"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
            )

        # -------------------------------------------------
        # Confusion Matrix
        # -------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 10)
        )

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            ax=ax,
            cmap="Blues",
            values_format="d",
        )

        ax.set_title(
            "CNN - Fashion MNIST - Confusion Matrix"
        )

        confusion_matrix_path = (
            ARTIFACTS_DIR
            / "confusion_matrix.png"
        )

        fig.savefig(
            confusion_matrix_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        # -------------------------------------------------
        # Training History
        # -------------------------------------------------

        history_path = (
            ARTIFACTS_DIR
            / "training_history.png"
        )

        save_training_history(
            history,
            history_path,
        )

        # -------------------------------------------------
        # Local Model Persistence
        # -------------------------------------------------

        model_path = (
            MODELS_DIR / "cnn.keras"
        )

        model.save(model_path)

        # -------------------------------------------------
        # MLflow Artifacts
        # -------------------------------------------------

        mlflow.log_artifacts(
            str(ARTIFACTS_DIR),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(model_path),
            artifact_path="keras",
        )

        # -------------------------------------------------
        # MLflow Native Model
        # -------------------------------------------------

        signature = infer_signature(
            x_test[:5],
            probabilities[:5],
        )

        model_info = (
            mlflow.tensorflow.log_model(
                model=model,
                name="cnn_model",
                signature=signature,
                input_example=x_test[:5],
            )
        )

        # -------------------------------------------------
        # Tags
        # -------------------------------------------------

        mlflow.set_tags({
            "dataset": "Fashion-MNIST",
            "task": "multiclass_classification",
            "framework": "TensorFlow/Keras",
            "model_family": "CNN",
            "stage": "baseline",
        })

        # -------------------------------------------------
        # Summary
        # -------------------------------------------------

        print(
            "\nPersistência concluída:"
        )

        print(
            f"Modelo local: {model_path}"
        )

        print(
            f"Artefatos locais: "
            f"{ARTIFACTS_DIR}"
        )

        print(
            f"MLflow Run ID: {run_id}"
        )

        print(
            f"MLflow Artifact URI: "
            f"{mlflow.get_artifact_uri()}"
        )

        print(
            f"MLflow Model URI: "
            f"{model_info.model_uri}"
        )


if __name__ == "__main__":
    main()