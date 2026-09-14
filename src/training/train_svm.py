from pathlib import Path
import json
import time

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
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
    EXPERIMENT_NAME,
    MODELS_DIR,
    SVM_TRAIN_SAMPLES,
    MLFLOW_TRACKING_URI,
)

from src.data.loader import (
    load_fashion_mnist,
    sample_training_data,
)

from src.preprocessing.svm import preprocess_svm
from src.models.svm import build_svm


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

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "svm"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()

    print(f"Train original: {x_train.shape}")
    print(f"Test original:  {x_test.shape}")

    # -----------------------------------------------------
    # Sampling
    # -----------------------------------------------------

    x_train, y_train = sample_training_data(
        x_train,
        y_train,
        SVM_TRAIN_SAMPLES,
    )

    # -----------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------

    x_train = preprocess_svm(x_train)
    x_test = preprocess_svm(x_test)

    print(f"Train preprocessado: {x_train.shape}")
    print(f"Test preprocessado:  {x_test.shape}")

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = build_svm()

    # -----------------------------------------------------
    # MLflow Run
    # -----------------------------------------------------

    with mlflow.start_run(run_name="svm_baseline") as run:

        run_id = run.info.run_id

        print(f"[MLflow] Run ID: {run_id}")
        print(f"[MLflow] Artifact URI: {mlflow.get_artifact_uri()}")

        # -------------------------------------------------
        # Parameters
        # -------------------------------------------------

        mlflow.log_params({
            "model": "SVM",
            "kernel": "rbf",
            "C": 10.0,
            "train_samples": len(x_train),
            "input_shape": "784",
            "normalization": "0_1",
        })

        # -------------------------------------------------
        # Training
        # -------------------------------------------------

        start = time.perf_counter()

        model.fit(x_train, y_train)

        training_time = time.perf_counter() - start

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        start = time.perf_counter()

        y_pred = model.predict(x_test)

        inference_time = time.perf_counter() - start

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
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
            "training_time_seconds": training_time,
            "inference_time_seconds": inference_time,
        }

        mlflow.log_metrics(metrics)

        print("\nMétricas:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        # -------------------------------------------------
        # Classification report
        # -------------------------------------------------

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        report_path = ARTIFACTS_DIR / "classification_report.json"

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
        # Metrics JSON
        # -------------------------------------------------

        metrics_path = ARTIFACTS_DIR / "metrics.json"

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
        # Confusion Matrix
        # -------------------------------------------------

        fig, ax = plt.subplots(figsize=(10, 10))

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            ax=ax,
            cmap="Blues",
            values_format="d",
        )

        ax.set_title("SVM - Fashion MNIST - Confusion Matrix")

        confusion_matrix_path = (
            ARTIFACTS_DIR / "confusion_matrix.png"
        )

        fig.savefig(
            confusion_matrix_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        # -------------------------------------------------
        # Local model persistence
        # -------------------------------------------------

        model_path = MODELS_DIR / "svm.joblib"

        joblib.dump(
            model,
            model_path,
        )

        # -------------------------------------------------
        # MLflow artifacts
        # -------------------------------------------------

        mlflow.log_artifacts(
            str(ARTIFACTS_DIR),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(model_path),
            artifact_path="joblib",
        )

        # -------------------------------------------------
        # MLflow native model
        # -------------------------------------------------

        signature = infer_signature(
            x_test[:5],
            model.predict(x_test[:5]),
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="svm_model",
            signature=signature,
            input_example=x_test[:5],
        )

        # -------------------------------------------------
        # Tags
        # -------------------------------------------------

        mlflow.set_tags({
            "dataset": "Fashion-MNIST",
            "task": "multiclass_classification",
            "framework": "scikit-learn",
            "model_family": "SVM",
            "stage": "baseline",
        })

        print("\nPersistência concluída:")
        print(f"Modelo local: {model_path}")
        print(f"Artefatos locais: {ARTIFACTS_DIR}")
        print(f"MLflow Run ID: {run_id}")
        print(f"MLflow Artifact URI: {mlflow.get_artifact_uri()}")
        print(f"MLflow Model URI: {model_info.model_uri}")


if __name__ == "__main__":
    main()