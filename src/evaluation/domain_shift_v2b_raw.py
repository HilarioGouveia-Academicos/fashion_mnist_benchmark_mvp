"""
Real External Domain Generalization — V2B RAW
==============================================

Evaluates the frozen V2B PRODUCT_ONLY v1 dataset with the already-trained
Fashion-MNIST SVM, MLP and CNN models, without canonicalization.

Methodological role
-------------------
This script establishes the RAW external-domain baseline:

    Generalization Gap = Clean Performance - V2B RAW Performance

The frozen V2B manifest is the only source of evaluation samples.
No model prediction is used to alter, remove or replace samples.

Expected project structure
--------------------------
artifacts/
└── v2b/
    └── v2b_evaluation_manifest_v1.csv

models/
├── svm.joblib
├── mlp.keras
└── cnn.keras

src/
└── preprocessing/
    ├── svm.py   -> preprocess_svm(images)
    ├── mlp.py   -> preprocess_mlp(images)
    └── cnn.py   -> preprocess_cnn(images)

Run
---
python -m src.evaluation.domain_shift_v2b_raw

or, if placed elsewhere:

python domain_shift_v2b_raw.py

Dependencies
------------
pandas, numpy, pillow, scikit-learn, matplotlib, joblib,
tensorflow, datasets, mlflow
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow import keras

from src.preprocessing.svm import preprocess_svm
from src.preprocessing.mlp import preprocess_mlp
from src.preprocessing.cnn import preprocess_cnn


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Frozen clean-domain references already established in the project.
# These are NOT recomputed here.
CLEAN_BASELINES = {
    "svm": {"accuracy": 0.8661, "f1_macro": 0.8652},
    "mlp": {"accuracy": 0.8630, "f1_macro": 0.8595},
    "cnn": {"accuracy": 0.8964, "f1_macro": 0.8941},
}

DEFAULT_HF_DATASET = "ashraq/fashion-product-images-small"
DEFAULT_EXPERIMENT = "Fashion-MNIST-Benchmark"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"

EPS = 1e-12


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_path: Path
    preprocess: Callable[[np.ndarray], np.ndarray]
    kind: str  # "sklearn" | "keras"


# ---------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------

def validate_frozen_manifest(df: pd.DataFrame) -> None:
    """Fail fast if the frozen V2B evaluation contract has been violated."""

    required = {
        "source_id",
        "dataset_index",
        "mapped_class_id",
        "mapped_class_name",
        "semantic_status",
        "presentation_type",
        "human_present",
        "composition_status",
        "human_review_status",
        "file_integrity",
        "exact_duplicate_status",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"V2B manifest missing required columns: {sorted(missing)}"
        )

    if len(df) == 0:
        raise ValueError("V2B manifest is empty.")

    if df["source_id"].duplicated().any():
        raise ValueError(
            f"Duplicate source_id values found: "
            f"{int(df['source_id'].duplicated().sum())}"
        )

    dataset_idx = pd.to_numeric(df["dataset_index"], errors="coerce")
    if dataset_idx.isna().any():
        raise ValueError(
            f"Invalid dataset_index values found: {int(dataset_idx.isna().sum())}"
        )

    checks = {
        "human_review_status": (df["human_review_status"] == "VALID_SAMPLE").all(),
        "semantic_status": (df["semantic_status"] == "VALID").all(),
        "composition_status": (df["composition_status"] == "CLEAN").all(),
        "presentation_type": (df["presentation_type"] == "PRODUCT_ONLY").all(),
        "human_present": (df["human_present"] == "NO").all(),
        "file_integrity": (df["file_integrity"] == "PASS").all(),
        "exact_duplicate_status": (
            df["exact_duplicate_status"] == "UNIQUE"
        ).all(),
    }

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(
            "Frozen V2B manifest violates evaluation policy: "
            + ", ".join(failed)
        )

    classes = set(pd.to_numeric(df["mapped_class_id"]).astype(int))
    expected = set(range(10))
    if classes != expected:
        raise ValueError(
            f"Expected all Fashion-MNIST classes 0..9. "
            f"Found={sorted(classes)}"
        )


# ---------------------------------------------------------------------
# RAW external representation
# ---------------------------------------------------------------------

def raw_image_to_28x28(image: Image.Image) -> np.ndarray:
    """
    RAW V2B representation.

    Important:
    - no foreground extraction
    - no crop-to-object
    - no polarity normalization
    - no centering
    - no canonicalizer

    Only the minimum representation required to make an external image
    consumable by the Fashion-MNIST models is applied:
        RGB/PIL -> grayscale -> resize 28x28

    Numerical normalization remains the responsibility of the existing
    model-specific preprocessing functions.
    """
    img = image.convert("L")
    img = img.resize((28, 28), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def load_v2b_images(
    manifest: pd.DataFrame,
    hf_dataset_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load exactly the samples referenced by the frozen manifest."""

    print(f"[INFO] Loading source dataset: {hf_dataset_name}")
    ds = load_dataset(hf_dataset_name, split="train")

    images: list[np.ndarray] = []
    labels: list[int] = []
    source_ids: list[str] = []

    total = len(manifest)

    for pos, row in enumerate(manifest.itertuples(index=False), start=1):
        dataset_index = int(row.dataset_index)
        item = ds[dataset_index]

        image = item["image"]
        images.append(raw_image_to_28x28(image))
        labels.append(int(row.mapped_class_id))
        source_ids.append(str(row.source_id))

        if pos % 50 == 0 or pos == total:
            print(f"[INFO] Loaded {pos}/{total} V2B samples")

    X = np.stack(images)
    y = np.asarray(labels, dtype=np.int64)

    return X, y, source_ids


# ---------------------------------------------------------------------
# Model loading and prediction
# ---------------------------------------------------------------------

def load_model(spec: ModelSpec) -> Any:
    if not spec.model_path.exists():
        raise FileNotFoundError(
            f"{spec.name.upper()} model not found: {spec.model_path}"
        )

    if spec.kind == "sklearn":
        return joblib.load(spec.model_path)

    if spec.kind == "keras":
        return keras.models.load_model(spec.model_path)

    raise ValueError(f"Unsupported model kind: {spec.kind}")


def predict_probabilities(
    model: Any,
    X: np.ndarray,
    kind: str,
) -> np.ndarray:
    """
    Return a probability matrix with shape (n_samples, 10).

    For SVM this assumes the persisted estimator exposes predict_proba,
    which is consistent with the calibrated/probabilistic project model.
    """

    if kind == "sklearn":
        if not hasattr(model, "predict_proba"):
            raise TypeError(
                "SVM model does not expose predict_proba(). "
                "Use the persisted calibrated/probabilistic SVM."
            )
        proba = model.predict_proba(X)

    elif kind == "keras":
        proba = model.predict(X, verbose=0)

    else:
        raise ValueError(f"Unsupported model kind: {kind}")

    proba = np.asarray(proba, dtype=float)

    if proba.ndim != 2 or proba.shape[1] != 10:
        raise ValueError(
            f"Expected probability matrix (n, 10), got {proba.shape}"
        )

    # Defensive numerical normalization.
    proba = np.clip(proba, EPS, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)

    return proba


# ---------------------------------------------------------------------
# Uncertainty / reliability metrics
# ---------------------------------------------------------------------

def entropy(proba: np.ndarray) -> np.ndarray:
    return -np.sum(proba * np.log(proba + EPS), axis=1)


def normalized_entropy(proba: np.ndarray) -> np.ndarray:
    return entropy(proba) / math.log(proba.shape[1])


def prediction_margin(proba: np.ndarray) -> np.ndarray:
    sorted_p = np.sort(proba, axis=1)
    return sorted_p[:, -1] - sorted_p[:, -2]


def build_prediction_frame(
    model_name: str,
    source_ids: list[str],
    y_true: np.ndarray,
    proba: np.ndarray,
) -> pd.DataFrame:
    y_pred = proba.argmax(axis=1)
    confidence = proba.max(axis=1)
    ent = entropy(proba)
    norm_ent = normalized_entropy(proba)
    margin = prediction_margin(proba)
    correct = y_pred == y_true

    out = pd.DataFrame(
        {
            "model": model_name,
            "source_id": source_ids,
            "y_true": y_true,
            "true_class": [CLASS_NAMES[i] for i in y_true],
            "y_pred": y_pred,
            "predicted_class": [CLASS_NAMES[i] for i in y_pred],
            "correct": correct,
            "confidence": confidence,
            "entropy": ent,
            "normalized_entropy": norm_ent,
            "margin": margin,
        }
    )

    # Store class probabilities for later audit/error analysis.
    for class_id, class_name in enumerate(CLASS_NAMES):
        safe_name = (
            class_name.lower()
            .replace("/", "_")
            .replace("-", "_")
            .replace(" ", "_")
        )
        out[f"p_{class_id}_{safe_name}"] = proba[:, class_id]

    return out


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def safe_mean(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.mean())


def evaluate_predictions(
    model_name: str,
    pred_df: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame, dict]:
    y_true = pred_df["y_true"].to_numpy()
    y_pred = pred_df["y_pred"].to_numpy()

    correct = pred_df["correct"]
    errors = ~correct

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "mean_confidence": float(pred_df["confidence"].mean()),
        "correct_confidence": safe_mean(
            pred_df.loc[correct, "confidence"]
        ),
        "error_confidence": safe_mean(
            pred_df.loc[errors, "confidence"]
        ),
        "mean_entropy": float(pred_df["entropy"].mean()),
        "mean_normalized_entropy": float(
            pred_df["normalized_entropy"].mean()
        ),
        "mean_margin": float(pred_df["margin"].mean()),
        "high_confidence_error_rate_090": float(
            ((errors) & (pred_df["confidence"] >= 0.90)).mean()
        ),
        "n_samples": int(len(pred_df)),
        "n_correct": int(correct.sum()),
        "n_errors": int(errors.sum()),
    }

    clean = CLEAN_BASELINES[model_name]
    metrics["clean_accuracy"] = clean["accuracy"]
    metrics["clean_f1_macro"] = clean["f1_macro"]
    metrics["generalization_gap_accuracy"] = (
        clean["accuracy"] - metrics["accuracy"]
    )
    metrics["generalization_gap_f1_macro"] = (
        clean["f1_macro"] - metrics["f1_macro"]
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(10)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    per_class_rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        row = report[class_name]
        support = int(row["support"])

        class_mask = y_true == class_id
        class_accuracy = (
            float((y_pred[class_mask] == y_true[class_mask]).mean())
            if support > 0
            else float("nan")
        )

        per_class_rows.append(
            {
                "model": model_name,
                "class_id": class_id,
                "class_name": class_name,
                "support": support,
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1-score"]),
                "class_accuracy": class_accuracy,
            }
        )

    per_class = pd.DataFrame(per_class_rows)

    return metrics, per_class, report


# ---------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------

def save_confusion_matrix(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(cm)

    ax.set_title(f"V2B RAW — {model_name.upper()} — Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_summary_plot(summary: pd.DataFrame, output_path: Path) -> None:
    names = summary["model"].str.upper().tolist()
    clean = summary["clean_accuracy"].to_numpy()
    external = summary["accuracy"].to_numpy()

    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, clean, width, label="Clean")
    ax.bar(x + width / 2, external, width, label="V2B RAW")

    ax.set_ylabel("Accuracy")
    ax.set_title("Clean vs V2B RAW Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(
    summary: pd.DataFrame,
    distribution: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# V2B RAW — Real External Domain Generalization",
        "",
        "**Dataset:** V2B PRODUCT_ONLY v1 — FROZEN",
        "",
        "**Canonicalization:** NONE",
        "",
        "## Dataset distribution",
        "",
        distribution.to_markdown(index=False),
        "",
        "## Main results",
        "",
        summary[
            [
                "model",
                "clean_accuracy",
                "accuracy",
                "generalization_gap_accuracy",
                "clean_f1_macro",
                "f1_macro",
                "generalization_gap_f1_macro",
                "mean_confidence",
                "error_confidence",
                "mean_normalized_entropy",
                "high_confidence_error_rate_090",
            ]
        ].to_markdown(index=False),
        "",
        "## Interpretation rule",
        "",
        "The RAW condition quantifies external-domain generalization before any "
        "canonicalization. A performance drop must be interpreted as an "
        "**external generalization gap**, not as isolated temporal drift.",
        "",
        "Because V2B PRODUCT_ONLY v1 is class-imbalanced, Macro-F1 and per-class "
        "metrics must be interpreted together with overall accuracy.",
        "",
        "No V2B sample may be removed or replaced based on these predictions. "
        "Any dataset change requires a new V2B version.",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------

def log_run_to_mlflow(
    model_name: str,
    metrics: dict[str, float],
    model_path: Path,
    manifest_path: Path,
    output_dir: Path,
    manifest_sha256: str | None,
) -> None:
    run_name = f"{model_name}_domain_shift_v2b_raw"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "model": model_name,
                "stage": "external_real_v2b",
                "dataset": "V2B_PRODUCT_ONLY_v1",
                "representation": "raw",
                "canonicalizer": "none",
                "evaluation_type": "external_domain_generalization",
                "dataset_status": "frozen",
            }
        )

        mlflow.log_params(
            {
                "model_path": str(model_path),
                "manifest_path": str(manifest_path),
                "n_classes": 10,
                "raw_resize": "28x28",
                "raw_grayscale": True,
                "canonicalization": False,
                "manifest_sha256": manifest_sha256 or "unknown",
            }
        )

        numeric_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float, np.integer, np.floating))
            and np.isfinite(value)
        }
        mlflow.log_metrics(
            {key: float(value) for key, value in numeric_metrics.items()}
        )

        model_dir = output_dir / model_name
        if model_dir.exists():
            mlflow.log_artifacts(
                str(model_dir),
                artifact_path=f"v2b_raw/{model_name}",
            )


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    manifest_path = project_root / args.manifest
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Frozen V2B manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    validate_frozen_manifest(manifest)

    print("[OK] Frozen V2B manifest integrity validated.")
    print(f"[INFO] Samples: {len(manifest)}")

    manifest_sha256 = None
    freeze_report_path = project_root / args.freeze_report
    if freeze_report_path.exists():
        try:
            freeze_report = json.loads(
                freeze_report_path.read_text(encoding="utf-8")
            )
            manifest_sha256 = (
                freeze_report.get("integrity", {})
                .get("manifest_sha256")
            )
        except Exception:
            pass

    X_raw, y_true, source_ids = load_v2b_images(
        manifest=manifest,
        hf_dataset_name=args.hf_dataset,
    )

    distribution = (
        manifest.groupby(
            ["mapped_class_id", "mapped_class_name"],
            as_index=False,
        )
        .agg(n=("source_id", "size"))
    )
    distribution.to_csv(
        output_dir / "v2b_raw_class_distribution.csv",
        index=False,
    )

    specs = [
        ModelSpec(
            name="svm",
            model_path=project_root / args.svm_model,
            preprocess=preprocess_svm,
            kind="sklearn",
        ),
        ModelSpec(
            name="mlp",
            model_path=project_root / args.mlp_model,
            preprocess=preprocess_mlp,
            kind="keras",
        ),
        ModelSpec(
            name="cnn",
            model_path=project_root / args.cnn_model,
            preprocess=preprocess_cnn,
            kind="keras",
        ),
    ]

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    summary_rows: list[dict[str, Any]] = []
    all_per_class: list[pd.DataFrame] = []

    for spec in specs:
        print()
        print("=" * 72)
        print(f"[MODEL] {spec.name.upper()}")
        print("=" * 72)

        model_out = output_dir / spec.name
        model_out.mkdir(parents=True, exist_ok=True)

        model = load_model(spec)

        X_model = spec.preprocess(X_raw.copy())
        print(f"[INFO] Preprocessed shape: {X_model.shape}")

        start = time.perf_counter()
        proba = predict_probabilities(
            model=model,
            X=X_model,
            kind=spec.kind,
        )
        inference_seconds = time.perf_counter() - start

        pred_df = build_prediction_frame(
            model_name=spec.name,
            source_ids=source_ids,
            y_true=y_true,
            proba=proba,
        )

        metrics, per_class, report = evaluate_predictions(
            model_name=spec.name,
            pred_df=pred_df,
        )

        metrics["inference_seconds"] = float(inference_seconds)
        metrics["inference_ms_per_sample"] = float(
            1000.0 * inference_seconds / len(pred_df)
        )

        pred_df.to_csv(
            model_out / "predictions.csv",
            index=False,
        )
        per_class.to_csv(
            model_out / "per_class_metrics.csv",
            index=False,
        )
        (model_out / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (model_out / "classification_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        save_confusion_matrix(
            model_name=spec.name,
            y_true=pred_df["y_true"].to_numpy(),
            y_pred=pred_df["y_pred"].to_numpy(),
            output_path=model_out / "confusion_matrix.png",
        )

        summary_rows.append(
            {"model": spec.name, **metrics}
        )
        all_per_class.append(per_class)

        log_run_to_mlflow(
            model_name=spec.name,
            metrics=metrics,
            model_path=spec.model_path,
            manifest_path=manifest_path,
            output_dir=output_dir,
            manifest_sha256=manifest_sha256,
        )

        print(
            f"[RESULT] accuracy={metrics['accuracy']:.4f} | "
            f"f1_macro={metrics['f1_macro']:.4f} | "
            f"gap_acc={metrics['generalization_gap_accuracy']:.4f} | "
            f"error_conf={metrics['error_confidence']:.4f}"
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        output_dir / "v2b_raw_summary.csv",
        index=False,
    )

    per_class_all = pd.concat(
        all_per_class,
        ignore_index=True,
    )
    per_class_all.to_csv(
        output_dir / "v2b_raw_per_class_metrics.csv",
        index=False,
    )

    save_summary_plot(
        summary=summary,
        output_path=output_dir / "clean_vs_v2b_raw_accuracy.png",
    )

    write_markdown_report(
        summary=summary,
        distribution=distribution,
        output_path=output_dir / "v2b_raw_report.md",
    )

    # Aggregate comparison run: evidence only, no model artifact logging.
    with mlflow.start_run(run_name="v2b_raw_comparison"):
        mlflow.set_tags(
            {
                "stage": "external_real_v2b",
                "dataset": "V2B_PRODUCT_ONLY_v1",
                "representation": "raw",
                "canonicalizer": "none",
                "run_type": "comparison",
            }
        )

        for _, row in summary.iterrows():
            name = row["model"]
            mlflow.log_metric(
                f"{name}_accuracy",
                float(row["accuracy"]),
            )
            mlflow.log_metric(
                f"{name}_f1_macro",
                float(row["f1_macro"]),
            )
            mlflow.log_metric(
                f"{name}_generalization_gap_accuracy",
                float(row["generalization_gap_accuracy"]),
            )
            mlflow.log_metric(
                f"{name}_generalization_gap_f1_macro",
                float(row["generalization_gap_f1_macro"]),
            )

        mlflow.log_artifacts(
            str(output_dir),
            artifact_path="v2b_raw_comparison",
        )

    print()
    print("[OK] V2B RAW experiment completed.")
    print(f"[OK] Results: {output_dir}")
    print()
    print(
        summary[
            [
                "model",
                "clean_accuracy",
                "accuracy",
                "generalization_gap_accuracy",
                "clean_f1_macro",
                "f1_macro",
                "generalization_gap_f1_macro",
            ]
        ].to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SVM/MLP/CNN on frozen V2B PRODUCT_ONLY v1 "
            "without canonicalization."
        )
    )

    parser.add_argument(
        "--project-root",
        default=".",
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/v2b/v2b_evaluation_manifest_v1.csv",
    )
    parser.add_argument(
        "--freeze-report",
        default="artifacts/v2b/v2b_freeze_report_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/domain_shift_v2b/raw",
    )
    parser.add_argument(
        "--hf-dataset",
        default=DEFAULT_HF_DATASET,
    )

    parser.add_argument(
        "--svm-model",
        default="models/svm.joblib",
    )
    parser.add_argument(
        "--mlp-model",
        default="models/mlp.keras",
    )
    parser.add_argument(
        "--cnn-model",
        default="models/cnn.keras",
    )

    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT,
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
