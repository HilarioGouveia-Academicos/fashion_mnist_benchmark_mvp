"""
V2B + Canonicalizer V2
======================

Evaluates the frozen V2B PRODUCT_ONLY v1 dataset after applying the
already-frozen Canonicalizer V2, using the same trained SVM, MLP and CNN
models used in V2B RAW.

Experimental question
---------------------
How much of the external generalization gap can be recovered by changing
ONLY the input representation?

    Recovery_V2 = Performance_V2 - Performance_RAW

Important controls
------------------
- Same frozen V2B manifest
- Same 238 samples
- Same trained models
- No retraining
- No sample removal based on predictions
- No change to labels
- Canonicalizer V2 must already be frozen before this evaluation

Recommended run
---------------
python -m src.evaluation.domain_shift_v2b_v2

Default canonicalizer target
----------------------------
src.preprocessing.canonicalizer_v2:canonicalize_image

If your project uses another module/function name:

python -m src.evaluation.domain_shift_v2b_v2 ^
  --canonicalizer-target src.preprocessing.canonicalizer_v2:canonicalize

The callable must accept one image and return a 28x28 canonical image.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import mlflow
import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image

# Reuse the validated V2B RAW experimental infrastructure.
from src.evaluation.domain_shift_v2b_raw import (
    CLASS_NAMES,
    CLEAN_BASELINES,
    DEFAULT_EXPERIMENT,
    DEFAULT_HF_DATASET,
    DEFAULT_TRACKING_URI,
    ModelSpec,
    build_prediction_frame,
    evaluate_predictions,
    load_model,
    log_run_to_mlflow,
    predict_probabilities,
    save_confusion_matrix,
    validate_frozen_manifest,
)

from src.preprocessing.svm import preprocess_svm
from src.preprocessing.mlp import preprocess_mlp
from src.preprocessing.cnn import preprocess_cnn


# ---------------------------------------------------------------------
# Canonicalizer loading
# ---------------------------------------------------------------------

DEFAULT_CANONICALIZER_TARGET = (
    #"src.preprocessing.canonicalizer_v2:canonicalize_image"
    "src.domain_shift.canonicalizer:canonicalize_image"    
)


def import_callable(target: str) -> Callable:
    """
    Import callable from:
        package.module:function_name
    """
    if ":" not in target:
        raise ValueError(
            "--canonicalizer-target must have format "
            "'package.module:function_name'"
        )

    module_name, function_name = target.split(":", 1)

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Could not import canonicalizer module '{module_name}'. "
            f"Received target: {target}"
        ) from exc

    try:
        fn = getattr(module, function_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Module '{module_name}' has no callable '{function_name}'. "
            f"Use --canonicalizer-target with the actual V2 function."
        ) from exc

    if not callable(fn):
        raise TypeError(
            f"Canonicalizer target '{target}' exists but is not callable."
        )

    return fn


# ---------------------------------------------------------------------
# Canonical representation adapter
# ---------------------------------------------------------------------

def canonical_output_to_28x28(result: Any) -> np.ndarray:
    """
    Normalize only the Python representation of the canonicalizer output.

    This function DOES NOT:
    - crop
    - threshold
    - invert
    - center
    - change the canonicalizer algorithm
    - perform model normalization

    Those operations belong either to Canonicalizer V2 or the existing
    model-specific preprocessing functions.
    """
    if isinstance(result, Image.Image):
        arr = np.asarray(result.convert("L"))
    else:
        arr = np.asarray(result)

    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[-1] in (3, 4):
            # This conversion is only an output-format adapter.
            # Canonicalizer logic itself must already be frozen.
            arr = np.asarray(
                Image.fromarray(arr.astype(np.uint8)).convert("L")
            )
        else:
            raise ValueError(
                f"Unsupported canonicalizer output shape: {arr.shape}"
            )

    if arr.ndim != 2:
        raise ValueError(
            f"Canonicalizer V2 must return a 2D image. Got {arr.shape}"
        )

    if arr.shape != (28, 28):
        raise ValueError(
            "Canonicalizer V2 output contract violation: "
            f"expected (28, 28), got {arr.shape}. "
            "Do not silently resize here; fix/use the frozen canonicalizer."
        )

    if not np.isfinite(arr).all():
        raise ValueError(
            "Canonicalizer V2 returned NaN or infinite values."
        )

    # Preserve 0..1 outputs if that is what V2 returns; otherwise convert
    # image-like values to uint8. Model preprocessors remain responsible
    # for numerical normalization.
    if np.issubdtype(arr.dtype, np.floating):
        amin = float(arr.min())
        amax = float(arr.max())
        if 0.0 <= amin and amax <= 1.0:
            arr = np.rint(arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def apply_canonicalizer_v2(
    image: Image.Image,
    canonicalizer: Callable,
) -> np.ndarray:
    """
    Apply the project's frozen Canonicalizer V2 to one source image.

    Preferred contract: canonicalizer(PIL.Image) -> 28x28 image/array.

    If the project implementation accepts a numpy array instead of PIL,
    pass --canonicalizer-input numpy.
    """
    result = canonicalizer(image)
    return canonical_output_to_28x28(result)


def apply_canonicalizer_v2_numpy(
    image: Image.Image,
    canonicalizer: Callable,
) -> np.ndarray:
    arr = np.asarray(image)
    result = canonicalizer(arr)
    return canonical_output_to_28x28(result)


def load_v2b_canonicalized_images(
    manifest: pd.DataFrame,
    hf_dataset_name: str,
    canonicalizer: Callable,
    canonicalizer_input: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load exactly the frozen V2B samples and apply Canonicalizer V2.
    """
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

        if canonicalizer_input == "pil":
            canonical = apply_canonicalizer_v2(
                image=image,
                canonicalizer=canonicalizer,
            )
        elif canonicalizer_input == "numpy":
            canonical = apply_canonicalizer_v2_numpy(
                image=image,
                canonicalizer=canonicalizer,
            )
        else:
            raise ValueError(
                f"Unsupported canonicalizer input: {canonicalizer_input}"
            )

        images.append(canonical)
        labels.append(int(row.mapped_class_id))
        source_ids.append(str(row.source_id))

        if pos % 50 == 0 or pos == total:
            print(
                f"[INFO] Canonicalized {pos}/{total} V2B samples"
            )

    X = np.stack(images)
    y = np.asarray(labels, dtype=np.int64)

    if X.shape != (len(manifest), 28, 28):
        raise ValueError(
            f"Unexpected canonicalized batch shape: {X.shape}"
        )

    return X, y, source_ids


# ---------------------------------------------------------------------
# RAW reference
# ---------------------------------------------------------------------

def load_raw_reference(
    raw_summary_path: Path,
) -> pd.DataFrame:
    """
    Load the already-frozen V2B RAW summary.

    We do not recompute RAW here. This preserves the experimental sequence.
    """
    if not raw_summary_path.exists():
        raise FileNotFoundError(
            "Frozen V2B RAW summary not found: "
            f"{raw_summary_path}\n"
            "Run/freeze the RAW experiment first."
        )

    raw = pd.read_csv(raw_summary_path)

    required = {"model", "accuracy", "f1_macro"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"RAW summary missing columns: {sorted(missing)}"
        )

    if set(raw["model"]) != {"svm", "mlp", "cnn"}:
        raise ValueError(
            "RAW summary must contain exactly svm, mlp and cnn."
        )

    return raw


# ---------------------------------------------------------------------
# V2 comparison metrics
# ---------------------------------------------------------------------

def add_recovery_metrics(
    model_name: str,
    metrics: dict[str, float],
    raw_summary: pd.DataFrame,
) -> dict[str, float]:
    row = raw_summary.loc[raw_summary["model"] == model_name].iloc[0]

    raw_acc = float(row["accuracy"])
    raw_f1 = float(row["f1_macro"])

    metrics["raw_accuracy"] = raw_acc
    metrics["raw_f1_macro"] = raw_f1

    metrics["canonicalization_recovery_accuracy"] = (
        metrics["accuracy"] - raw_acc
    )
    metrics["canonicalization_recovery_f1_macro"] = (
        metrics["f1_macro"] - raw_f1
    )

    # Fraction of the original clean->RAW gap recovered.
    raw_gap_acc = CLEAN_BASELINES[model_name]["accuracy"] - raw_acc
    raw_gap_f1 = CLEAN_BASELINES[model_name]["f1_macro"] - raw_f1

    metrics["gap_recovery_ratio_accuracy"] = (
        metrics["canonicalization_recovery_accuracy"] / raw_gap_acc
        if raw_gap_acc != 0
        else float("nan")
    )
    metrics["gap_recovery_ratio_f1_macro"] = (
        metrics["canonicalization_recovery_f1_macro"] / raw_gap_f1
        if raw_gap_f1 != 0
        else float("nan")
    )

    return metrics


# ---------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------

def dataframe_to_markdown_safe(df: pd.DataFrame) -> str:
    """
    Prevent an optional formatting dependency from killing the experiment.
    """
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def write_v2_report(
    summary: pd.DataFrame,
    output_path: Path,
    canonicalizer_target: str,
) -> None:
    comparison_cols = [
        "model",
        "clean_accuracy",
        "raw_accuracy",
        "accuracy",
        "canonicalization_recovery_accuracy",
        "generalization_gap_accuracy",
        "gap_recovery_ratio_accuracy",
        "raw_f1_macro",
        "f1_macro",
        "canonicalization_recovery_f1_macro",
        "error_confidence",
        "mean_normalized_entropy",
    ]

    lines = [
        "# V2B + Canonicalizer V2",
        "",
        "**Dataset:** V2B PRODUCT_ONLY v1 — FROZEN",
        "",
        "**Representation:** Canonicalizer V2",
        "",
        f"**Canonicalizer target:** `{canonicalizer_target}`",
        "",
        "## Main comparison",
        "",
        dataframe_to_markdown_safe(summary[comparison_cols]),
        "",
        "## Core quantities",
        "",
        "`Recovery_V2 = Performance_V2 - Performance_RAW`",
        "",
        "`Gap Recovery Ratio = Recovery_V2 / (Clean - RAW)`",
        "",
        "A positive recovery indicates that changing the representation, "
        "without retraining the model, recovered part of the external-domain "
        "performance loss.",
        "",
        "A limited recovery indicates that representation alignment alone is "
        "insufficient and that training-domain mismatch or other domain "
        "factors may remain.",
        "",
        "No sample was removed or replaced based on model predictions.",
    ]

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------

def log_v2_run(
    model_name: str,
    metrics: dict[str, float],
    model_path: Path,
    manifest_path: Path,
    output_dir: Path,
    manifest_sha256: str | None,
    canonicalizer_target: str,
) -> None:
    run_name = f"{model_name}_domain_shift_v2b_canonical_v2"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "model": model_name,
                "stage": "external_real_v2b",
                "dataset": "V2B_PRODUCT_ONLY_v1",
                "representation": "canonicalized",
                "canonicalizer": "v2",
                "evaluation_type": "external_domain_generalization",
                "dataset_status": "frozen",
            }
        )

        mlflow.log_params(
            {
                "model_path": str(model_path),
                "manifest_path": str(manifest_path),
                "n_classes": 10,
                "canonicalization": True,
                "canonicalizer_version": "v2",
                "canonicalizer_target": canonicalizer_target,
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
                artifact_path=f"v2b_canonical_v2/{model_name}",
            )


# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()

    manifest_path = project_root / args.manifest
    freeze_report_path = project_root / args.freeze_report
    raw_summary_path = project_root / args.raw_summary
    output_dir = project_root / args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Frozen V2B manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path,
        dtype=str,
    ).fillna("")

    validate_frozen_manifest(manifest)

    print("[OK] Frozen V2B manifest integrity validated.")
    print(f"[INFO] Samples: {len(manifest)}")

    raw_summary = load_raw_reference(raw_summary_path)
    print("[OK] Frozen V2B RAW reference loaded.")

    canonicalizer = import_callable(args.canonicalizer_target)
    print(
        "[OK] Canonicalizer V2 loaded: "
        f"{args.canonicalizer_target}"
    )

    manifest_sha256 = None
    if freeze_report_path.exists():
        try:
            report = json.loads(
                freeze_report_path.read_text(encoding="utf-8")
            )
            manifest_sha256 = (
                report.get("integrity", {})
                .get("manifest_sha256")
            )
        except Exception:
            pass

    X_v2, y_true, source_ids = load_v2b_canonicalized_images(
        manifest=manifest,
        hf_dataset_name=args.hf_dataset,
        canonicalizer=canonicalizer,
        canonicalizer_input=args.canonicalizer_input,
    )

    # Save exact canonicalized representations used by this experiment.
    if args.save_canonicalized_npz:
        np.savez_compressed(
            output_dir / "v2b_canonicalized_v2_inputs.npz",
            images=X_v2,
            labels=y_true,
            source_ids=np.asarray(source_ids),
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

    summary_rows = []
    per_class_all = []

    for spec in specs:
        print()
        print("=" * 72)
        print(f"[MODEL] {spec.name.upper()} — CANONICALIZER V2")
        print("=" * 72)

        model_out = output_dir / spec.name
        model_out.mkdir(parents=True, exist_ok=True)

        model = load_model(spec)
        X_model = spec.preprocess(X_v2.copy())

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

        metrics = add_recovery_metrics(
            model_name=spec.name,
            metrics=metrics,
            raw_summary=raw_summary,
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
            model_name=f"{spec.name} / V2",
            y_true=pred_df["y_true"].to_numpy(),
            y_pred=pred_df["y_pred"].to_numpy(),
            output_path=model_out / "confusion_matrix.png",
        )

        summary_rows.append(
            {"model": spec.name, **metrics}
        )
        per_class_all.append(per_class)

        log_v2_run(
            model_name=spec.name,
            metrics=metrics,
            model_path=spec.model_path,
            manifest_path=manifest_path,
            output_dir=output_dir,
            manifest_sha256=manifest_sha256,
            canonicalizer_target=args.canonicalizer_target,
        )

        print(
            f"[RESULT] accuracy={metrics['accuracy']:.4f} | "
            f"f1_macro={metrics['f1_macro']:.4f} | "
            f"recovery_acc="
            f"{metrics['canonicalization_recovery_accuracy']:+.4f} | "
            f"recovery_f1="
            f"{metrics['canonicalization_recovery_f1_macro']:+.4f} | "
            f"gap_recovered_acc="
            f"{metrics['gap_recovery_ratio_accuracy']:.2%}"
        )

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        output_dir / "v2b_canonical_v2_summary.csv",
        index=False,
    )

    pd.concat(
        per_class_all,
        ignore_index=True,
    ).to_csv(
        output_dir / "v2b_canonical_v2_per_class_metrics.csv",
        index=False,
    )

    write_v2_report(
        summary=summary,
        output_path=output_dir / "v2b_canonical_v2_report.md",
        canonicalizer_target=args.canonicalizer_target,
    )

    # Consolidated MLflow comparison run.
    with mlflow.start_run(
        run_name="v2b_canonical_v2_comparison"
    ):
        mlflow.set_tags(
            {
                "stage": "external_real_v2b",
                "dataset": "V2B_PRODUCT_ONLY_v1",
                "representation": "canonicalized",
                "canonicalizer": "v2",
                "run_type": "comparison",
            }
        )

        mlflow.log_param(
            "canonicalizer_target",
            args.canonicalizer_target,
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
                f"{name}_recovery_accuracy",
                float(
                    row["canonicalization_recovery_accuracy"]
                ),
            )
            mlflow.log_metric(
                f"{name}_recovery_f1_macro",
                float(
                    row["canonicalization_recovery_f1_macro"]
                ),
            )
            mlflow.log_metric(
                f"{name}_gap_recovery_ratio_accuracy",
                float(row["gap_recovery_ratio_accuracy"]),
            )

        mlflow.log_artifacts(
            str(output_dir),
            artifact_path="v2b_canonical_v2_comparison",
        )

    print()
    print("[OK] V2B Canonicalizer V2 experiment completed.")
    print(f"[OK] Results: {output_dir}")
    print()
    print(
        summary[
            [
                "model",
                "clean_accuracy",
                "raw_accuracy",
                "accuracy",
                "canonicalization_recovery_accuracy",
                "generalization_gap_accuracy",
                "gap_recovery_ratio_accuracy",
                "raw_f1_macro",
                "f1_macro",
                "canonicalization_recovery_f1_macro",
            ]
        ].to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen V2B PRODUCT_ONLY v1 after "
            "Canonicalizer V2."
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
        "--raw-summary",
        default="artifacts/domain_shift_v2b/raw/v2b_raw_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/domain_shift_v2b/canonical_v2",
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
        "--canonicalizer-target",
        default=DEFAULT_CANONICALIZER_TARGET,
        help=(
            "Frozen V2 canonicalizer callable as "
            "package.module:function_name"
        ),
    )
    parser.add_argument(
        "--canonicalizer-input",
        choices=["pil", "numpy"],
        default="pil",
        help=(
            "Input type expected by your V2 canonicalizer. "
            "Default: PIL.Image."
        ),
    )
    parser.add_argument(
        "--save-canonicalized-npz",
        action="store_true",
        help=(
            "Persist exact 28x28 canonicalized V2 inputs as NPZ "
            "for audit/reproducibility."
        ),
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
