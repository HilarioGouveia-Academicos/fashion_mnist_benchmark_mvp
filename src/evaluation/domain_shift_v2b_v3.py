"""
V2B + Canonicalizer V3
======================

Evaluates the frozen V2B PRODUCT_ONLY v1 dataset after applying the
already-frozen Canonicalizer V3, using the same trained SVM, MLP and CNN
models used in V2B RAW and V2.

Primary questions
-----------------
1. Does V3 recover more external-domain performance than RAW?
2. Does V3 outperform V2 on the same frozen 238 samples?

Core quantities
---------------
Recovery_V3 = Performance_V3 - Performance_RAW
Delta_V3_V2 = Performance_V3 - Performance_V2

Experimental controls
---------------------
- Same frozen V2B manifest
- Same 238 samples
- Same labels
- Same trained SVM / MLP / CNN
- No retraining
- No sample removal based on predictions
- No modification of V2 or V3 after inspecting V2B predictions

Recommended run
---------------
python -m src.evaluation.domain_shift_v2b_v3 \
    --canonicalizer-input numpy \
    --save-canonicalized-npz

Default V3 callable
-------------------
src.preprocessing.canonicalizer_v3_candidate:canonicalize_image_v3

Override when needed:
python -m src.evaluation.domain_shift_v2b_v3 \
    --canonicalizer-target src.some_module:canonicalize_image \
    --canonicalizer-input numpy
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

from src.evaluation.domain_shift_v2b_raw import (
    CLEAN_BASELINES,
    DEFAULT_EXPERIMENT,
    DEFAULT_HF_DATASET,
    DEFAULT_TRACKING_URI,
    ModelSpec,
    build_prediction_frame,
    evaluate_predictions,
    load_model,
    predict_probabilities,
    save_confusion_matrix,
    validate_frozen_manifest,
)

from src.preprocessing.svm import preprocess_svm
from src.preprocessing.mlp import preprocess_mlp
from src.preprocessing.cnn import preprocess_cnn


DEFAULT_CANONICALIZER_TARGET = (
#    "src.preprocessing.canonicalizer_v3_candidate:canonicalize_image_v3"
    "src.preprocessing.canonicalizer_v3_candidate:canonicalize_image_v3"
)


# ---------------------------------------------------------------------
# Dynamic canonicalizer import
# ---------------------------------------------------------------------

def import_callable(target: str) -> Callable:
    if ":" not in target:
        raise ValueError(
            "--canonicalizer-target must use "
            "'package.module:function_name'"
        )

    module_name, function_name = target.split(":", 1)

    module = importlib.import_module(module_name)

    if not hasattr(module, function_name):
        raise AttributeError(
            f"Module '{module_name}' has no callable "
            f"'{function_name}'."
        )

    fn = getattr(module, function_name)

    if not callable(fn):
        raise TypeError(
            f"Canonicalizer target '{target}' is not callable."
        )

    return fn


# ---------------------------------------------------------------------
# Canonicalizer V3 adapter
# ---------------------------------------------------------------------

def canonical_output_to_28x28(result: Any) -> np.ndarray:
    """
    Validate/adapt the returned object without changing V3 algorithm logic.

    Expected final representation:
        grayscale 28x28 uint8
    """
    if isinstance(result, Image.Image):
        arr = np.asarray(
            result.convert("L"),
            dtype=np.uint8,
        )
    else:
        arr = np.asarray(result)

    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[-1] in (3, 4):
            arr = np.asarray(
                Image.fromarray(
                    np.clip(arr, 0, 255).astype(np.uint8)
                ).convert("L"),
                dtype=np.uint8,
            )
        else:
            raise ValueError(
                f"Unsupported V3 output shape: {arr.shape}"
            )

    if arr.ndim != 2:
        raise ValueError(
            f"Canonicalizer V3 must return a 2D image. "
            f"Got {arr.shape}"
        )

    if arr.shape != (28, 28):
        raise ValueError(
            "Canonicalizer V3 output contract violation: "
            f"expected (28, 28), got {arr.shape}. "
            "Do not silently resize in the evaluation script."
        )

    if not np.isfinite(arr).all():
        raise ValueError(
            "Canonicalizer V3 returned NaN/inf values."
        )

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



# ---------------------------------------------------------------------
# Canonicalizer V3 adapter
# ---------------------------------------------------------------------

def extract_canonical_image(result: Any) -> np.ndarray:
    """
    Extrai a imagem canonicalizada do retorno do V3.

    Contrato atual:
        CanonicalizationResult(
            canonical=np.ndarray,
            mask=np.ndarray,
            roi=np.ndarray,
        )

    Mantém compatibilidade defensiva com ndarray, tuple e dict.
    """
    if hasattr(result, "canonical"):
        canonical = getattr(result, "canonical")

        if canonical is None:
            raise ValueError(
                "Canonicalizer V3 returned result.canonical=None."
            )

        return canonical

    if isinstance(result, np.ndarray):
        return result

    if isinstance(result, tuple):
        if not result:
            raise ValueError(
                "Canonicalizer returned an empty tuple."
            )
        return result[0]

    if isinstance(result, dict):
        for key in (
            "canonical",
            "canonical_image",
            "image",
        ):
            if key in result:
                return result[key]

        raise ValueError(
            "Canonicalizer returned a dict but no canonical "
            "image key was found."
        )

    raise TypeError(
        "Unsupported Canonicalizer V3 return type: "
        f"{type(result).__name__}. Expected an object with "
        "attribute 'canonical', ndarray, tuple or dict."
    )


def apply_v3_pil(
    image: Image.Image,
    canonicalizer: Callable,
) -> np.ndarray:
    """
    Adapter para origem PIL.

    O V3 trabalha com np.ndarray; portanto PIL é convertido
    explicitamente para RGB uint8 antes da canonicalização.
    """
    arr = np.asarray(
        image.convert("RGB"),
        dtype=np.uint8,
    )

    result = canonicalizer(arr)

    canonical_image = extract_canonical_image(
        result
    )

    return canonical_output_to_28x28(
        canonical_image
    )


def apply_v3_numpy(
    image: Image.Image,
    canonicalizer: Callable,
) -> np.ndarray:
    """
    Adapter principal usado no experimento V2B V3.

    Converte a imagem externa para grayscale 2D uint8. Esse formato
    é aceito pelo canonicalizer V3, que internamente converte para BGR.
    """
    arr = np.asarray(
        image.convert("L"),
        dtype=np.uint8,
    )

    if arr.ndim != 2:
        raise ValueError(
            f"Expected grayscale 2D input for V3, got {arr.shape}"
        )

    result = canonicalizer(arr)

    canonical_image = extract_canonical_image(
        result
    )

    return canonical_output_to_28x28(
        canonical_image
    )


def load_v2b_canonicalized_v3(
    manifest: pd.DataFrame,
    hf_dataset_name: str,
    canonicalizer: Callable,
    canonicalizer_input: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    print(f"[INFO] Loading source dataset: {hf_dataset_name}")
    ds = load_dataset(hf_dataset_name, split="train")

    images = []
    labels = []
    source_ids = []

    total = len(manifest)

    for pos, row in enumerate(
        manifest.itertuples(index=False),
        start=1,
    ):
        dataset_index = int(row.dataset_index)
        image = ds[dataset_index]["image"]

        if canonicalizer_input == "numpy":
            canonical = apply_v3_numpy(
                image,
                canonicalizer,
            )
        else:
            canonical = apply_v3_pil(
                image,
                canonicalizer,
            )

        images.append(canonical)
        labels.append(int(row.mapped_class_id))
        source_ids.append(str(row.source_id))

        if pos % 50 == 0 or pos == total:
            print(
                f"[INFO] Canonicalized V3 "
                f"{pos}/{total} V2B samples"
            )

    X = np.stack(images)
    y = np.asarray(labels, dtype=np.int64)

    if X.shape != (len(manifest), 28, 28):
        raise ValueError(
            f"Unexpected V3 batch shape: {X.shape}"
        )

    return X, y, source_ids


# ---------------------------------------------------------------------
# Frozen reference summaries
# ---------------------------------------------------------------------

def load_reference_summary(
    path: Path,
    label: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} summary not found: {path}"
        )

    df = pd.read_csv(path)

    required = {"model", "accuracy", "f1_macro"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{label} summary missing columns: {sorted(missing)}"
        )

    if set(df["model"]) != {"svm", "mlp", "cnn"}:
        raise ValueError(
            f"{label} summary must contain exactly svm, mlp and cnn."
        )

    return df


def add_comparison_metrics(
    model_name: str,
    metrics: dict[str, float],
    raw_summary: pd.DataFrame,
    v2_summary: pd.DataFrame,
) -> dict[str, float]:

    raw_row = raw_summary.loc[
        raw_summary["model"] == model_name
    ].iloc[0]

    v2_row = v2_summary.loc[
        v2_summary["model"] == model_name
    ].iloc[0]

    raw_acc = float(raw_row["accuracy"])
    raw_f1 = float(raw_row["f1_macro"])

    v2_acc = float(v2_row["accuracy"])
    v2_f1 = float(v2_row["f1_macro"])

    v3_acc = float(metrics["accuracy"])
    v3_f1 = float(metrics["f1_macro"])

    metrics["raw_accuracy"] = raw_acc
    metrics["raw_f1_macro"] = raw_f1

    metrics["v2_accuracy"] = v2_acc
    metrics["v2_f1_macro"] = v2_f1

    metrics["canonicalization_recovery_accuracy"] = (
        v3_acc - raw_acc
    )
    metrics["canonicalization_recovery_f1_macro"] = (
        v3_f1 - raw_f1
    )

    metrics["delta_v3_v2_accuracy"] = (
        v3_acc - v2_acc
    )
    metrics["delta_v3_v2_f1_macro"] = (
        v3_f1 - v2_f1
    )

    raw_gap_acc = (
        CLEAN_BASELINES[model_name]["accuracy"]
        - raw_acc
    )
    raw_gap_f1 = (
        CLEAN_BASELINES[model_name]["f1_macro"]
        - raw_f1
    )

    metrics["gap_recovery_ratio_accuracy"] = (
        metrics["canonicalization_recovery_accuracy"]
        / raw_gap_acc
        if raw_gap_acc != 0
        else float("nan")
    )

    metrics["gap_recovery_ratio_f1_macro"] = (
        metrics["canonicalization_recovery_f1_macro"]
        / raw_gap_f1
        if raw_gap_f1 != 0
        else float("nan")
    )

    return metrics


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def dataframe_to_markdown_safe(
    df: pd.DataFrame,
) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return (
            "```\n"
            + df.to_string(index=False)
            + "\n```"
        )


def write_v3_report(
    summary: pd.DataFrame,
    output_path: Path,
    canonicalizer_target: str,
) -> None:

    cols = [
        "model",
        "clean_accuracy",
        "raw_accuracy",
        "v2_accuracy",
        "accuracy",
        "canonicalization_recovery_accuracy",
        "delta_v3_v2_accuracy",
        "generalization_gap_accuracy",
        "gap_recovery_ratio_accuracy",
        "raw_f1_macro",
        "v2_f1_macro",
        "f1_macro",
        "canonicalization_recovery_f1_macro",
        "delta_v3_v2_f1_macro",
    ]

    lines = [
        "# V2B + Canonicalizer V3",
        "",
        "**Dataset:** V2B PRODUCT_ONLY v1 — FROZEN",
        "",
        "**Representation:** Canonicalizer V3",
        "",
        f"**Canonicalizer target:** `{canonicalizer_target}`",
        "",
        "## RAW → V2 → V3 comparison",
        "",
        dataframe_to_markdown_safe(
            summary[cols]
        ),
        "",
        "## Core quantities",
        "",
        "`Recovery_V3 = Performance_V3 - Performance_RAW`",
        "",
        "`Delta_V3_V2 = Performance_V3 - Performance_V2`",
        "",
        "`Gap Recovery Ratio = Recovery_V3 / (Clean - RAW)`",
        "",
        "A positive Delta_V3_V2 indicates that V3 improves downstream "
        "generalization over V2 on the same frozen external dataset.",
        "",
        "A negative Delta_V3_V2 is also a valid experimental result: "
        "visual canonicalization improvements do not necessarily imply "
        "better classifier performance.",
        "",
        "No V2B sample was removed, relabeled or replaced based on model "
        "predictions.",
    ]

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------

def log_v3_run(
    model_name: str,
    metrics: dict[str, float],
    model_path: Path,
    manifest_path: Path,
    output_dir: Path,
    manifest_sha256: str | None,
    canonicalizer_target: str,
) -> None:

    run_name = (
        f"{model_name}_domain_shift_v2b_canonical_v3"
    )

    with mlflow.start_run(
        run_name=run_name
    ):
        mlflow.set_tags(
            {
                "model": model_name,
                "stage": "external_real_v2b",
                "dataset": "V2B_PRODUCT_ONLY_v1",
                "representation": "canonicalized",
                "canonicalizer": "v3",
                "evaluation_type": (
                    "external_domain_generalization"
                ),
                "dataset_status": "frozen",
            }
        )

        mlflow.log_params(
            {
                "model_path": str(model_path),
                "manifest_path": str(manifest_path),
                "n_classes": 10,
                "canonicalization": True,
                "canonicalizer_version": "v3",
                "canonicalizer_target": canonicalizer_target,
                "manifest_sha256": (
                    manifest_sha256 or "unknown"
                ),
            }
        )

        numeric_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(
                value,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            )
            and np.isfinite(value)
        }

        mlflow.log_metrics(
            {
                key: float(value)
                for key, value
                in numeric_metrics.items()
            }
        )

        model_dir = output_dir / model_name

        if model_dir.exists():
            mlflow.log_artifacts(
                str(model_dir),
                artifact_path=(
                    f"v2b_canonical_v3/{model_name}"
                ),
            )


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:

    root = Path(args.project_root).resolve()

    manifest_path = root / args.manifest
    freeze_report_path = root / args.freeze_report
    raw_summary_path = root / args.raw_summary
    v2_summary_path = root / args.v2_summary
    output_dir = root / args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Frozen V2B manifest not found: "
            f"{manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path,
        dtype=str,
    ).fillna("")

    validate_frozen_manifest(manifest)

    print(
        "[OK] Frozen V2B manifest integrity validated."
    )
    print(
        f"[INFO] Samples: {len(manifest)}"
    )

    raw_summary = load_reference_summary(
        raw_summary_path,
        "V2B RAW",
    )
    print(
        "[OK] Frozen V2B RAW reference loaded."
    )

    v2_summary = load_reference_summary(
        v2_summary_path,
        "V2B Canonicalizer V2",
    )
    print(
        "[OK] Frozen V2B V2 reference loaded."
    )

    canonicalizer = import_callable(
        args.canonicalizer_target
    )

    print(
        "[OK] Canonicalizer V3 loaded: "
        f"{args.canonicalizer_target}"
    )

    manifest_sha256 = None

    if freeze_report_path.exists():
        try:
            freeze_report = json.loads(
                freeze_report_path.read_text(
                    encoding="utf-8"
                )
            )
            manifest_sha256 = (
                freeze_report
                .get("integrity", {})
                .get("manifest_sha256")
            )
        except Exception:
            pass

    X_v3, y_true, source_ids = (
        load_v2b_canonicalized_v3(
            manifest=manifest,
            hf_dataset_name=args.hf_dataset,
            canonicalizer=canonicalizer,
            canonicalizer_input=(
                args.canonicalizer_input
            ),
        )
    )

    if args.save_canonicalized_npz:
        np.savez_compressed(
            output_dir
            / "v2b_canonicalized_v3_inputs.npz",
            images=X_v3,
            labels=y_true,
            source_ids=np.asarray(source_ids),
        )

    specs = [
        ModelSpec(
            name="svm",
            model_path=root / args.svm_model,
            preprocess=preprocess_svm,
            kind="sklearn",
        ),
        ModelSpec(
            name="mlp",
            model_path=root / args.mlp_model,
            preprocess=preprocess_mlp,
            kind="keras",
        ),
        ModelSpec(
            name="cnn",
            model_path=root / args.cnn_model,
            preprocess=preprocess_cnn,
            kind="keras",
        ),
    ]

    mlflow.set_tracking_uri(
        args.tracking_uri
    )
    mlflow.set_experiment(
        args.experiment_name
    )

    summary_rows = []
    all_per_class = []

    for spec in specs:

        print()
        print("=" * 72)
        print(
            f"[MODEL] {spec.name.upper()} "
            "— CANONICALIZER V3"
        )
        print("=" * 72)

        model_out = (
            output_dir / spec.name
        )
        model_out.mkdir(
            parents=True,
            exist_ok=True,
        )

        model = load_model(spec)

        X_model = spec.preprocess(
            X_v3.copy()
        )

        print(
            f"[INFO] Preprocessed shape: "
            f"{X_model.shape}"
        )

        start = time.perf_counter()

        proba = predict_probabilities(
            model=model,
            X=X_model,
            kind=spec.kind,
        )

        inference_seconds = (
            time.perf_counter() - start
        )

        pred_df = build_prediction_frame(
            model_name=spec.name,
            source_ids=source_ids,
            y_true=y_true,
            proba=proba,
        )

        metrics, per_class, report = (
            evaluate_predictions(
                model_name=spec.name,
                pred_df=pred_df,
            )
        )

        metrics["inference_seconds"] = (
            float(inference_seconds)
        )
        metrics["inference_ms_per_sample"] = (
            float(
                1000.0
                * inference_seconds
                / len(pred_df)
            )
        )

        metrics = add_comparison_metrics(
            model_name=spec.name,
            metrics=metrics,
            raw_summary=raw_summary,
            v2_summary=v2_summary,
        )

        pred_df.to_csv(
            model_out / "predictions.csv",
            index=False,
        )

        per_class.to_csv(
            model_out / "per_class_metrics.csv",
            index=False,
        )

        (
            model_out / "metrics.json"
        ).write_text(
            json.dumps(
                metrics,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        (
            model_out
            / "classification_report.json"
        ).write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        save_confusion_matrix(
            model_name=f"{spec.name} / V3",
            y_true=pred_df[
                "y_true"
            ].to_numpy(),
            y_pred=pred_df[
                "y_pred"
            ].to_numpy(),
            output_path=(
                model_out
                / "confusion_matrix.png"
            ),
        )

        summary_rows.append(
            {
                "model": spec.name,
                **metrics,
            }
        )
        all_per_class.append(
            per_class
        )

        log_v3_run(
            model_name=spec.name,
            metrics=metrics,
            model_path=spec.model_path,
            manifest_path=manifest_path,
            output_dir=output_dir,
            manifest_sha256=manifest_sha256,
            canonicalizer_target=(
                args.canonicalizer_target
            ),
        )

        print(
            f"[RESULT] "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"f1_macro={metrics['f1_macro']:.4f} | "
            f"recovery_vs_raw_acc="
            f"{metrics['canonicalization_recovery_accuracy']:+.4f} | "
            f"delta_v3_v2_acc="
            f"{metrics['delta_v3_v2_accuracy']:+.4f} | "
            f"delta_v3_v2_f1="
            f"{metrics['delta_v3_v2_f1_macro']:+.4f}"
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        output_dir
        / "v2b_canonical_v3_summary.csv",
        index=False,
    )

    pd.concat(
        all_per_class,
        ignore_index=True,
    ).to_csv(
        output_dir
        / "v2b_canonical_v3_per_class_metrics.csv",
        index=False,
    )

    write_v3_report(
        summary=summary,
        output_path=(
            output_dir
            / "v2b_canonical_v3_report.md"
        ),
        canonicalizer_target=(
            args.canonicalizer_target
        ),
    )

    with mlflow.start_run(
        run_name="v2b_canonical_v3_comparison"
    ):
        mlflow.set_tags(
            {
                "stage": "external_real_v2b",
                "dataset": "V2B_PRODUCT_ONLY_v1",
                "representation": "canonicalized",
                "canonicalizer": "v3",
                "run_type": "comparison",
            }
        )

        mlflow.log_param(
            "canonicalizer_target",
            args.canonicalizer_target,
        )

        for _, row in summary.iterrows():

            model_name = row["model"]

            mlflow.log_metric(
                f"{model_name}_accuracy",
                float(row["accuracy"]),
            )
            mlflow.log_metric(
                f"{model_name}_f1_macro",
                float(row["f1_macro"]),
            )
            mlflow.log_metric(
                f"{model_name}_recovery_vs_raw_accuracy",
                float(
                    row[
                        "canonicalization_recovery_accuracy"
                    ]
                ),
            )
            mlflow.log_metric(
                f"{model_name}_delta_v3_v2_accuracy",
                float(
                    row[
                        "delta_v3_v2_accuracy"
                    ]
                ),
            )
            mlflow.log_metric(
                f"{model_name}_delta_v3_v2_f1_macro",
                float(
                    row[
                        "delta_v3_v2_f1_macro"
                    ]
                ),
            )

        mlflow.log_artifacts(
            str(output_dir),
            artifact_path=(
                "v2b_canonical_v3_comparison"
            ),
        )

    print()
    print(
        "[OK] V2B Canonicalizer V3 "
        "experiment completed."
    )
    print(
        f"[OK] Results: {output_dir}"
    )
    print()

    print(
        summary[
            [
                "model",
                "clean_accuracy",
                "raw_accuracy",
                "v2_accuracy",
                "accuracy",
                "canonicalization_recovery_accuracy",
                "delta_v3_v2_accuracy",
                "generalization_gap_accuracy",
                "gap_recovery_ratio_accuracy",
                "raw_f1_macro",
                "v2_f1_macro",
                "f1_macro",
                "canonicalization_recovery_f1_macro",
                "delta_v3_v2_f1_macro",
            ]
        ].to_string(index=False)
    )


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen V2B PRODUCT_ONLY v1 "
            "after Canonicalizer V3."
        )
    )

    parser.add_argument(
        "--project-root",
        default=".",
    )

    parser.add_argument(
        "--manifest",
        default=(
            "artifacts/v2b/"
            "v2b_evaluation_manifest_v1.csv"
        ),
    )

    parser.add_argument(
        "--freeze-report",
        default=(
            "artifacts/v2b/"
            "v2b_freeze_report_v1.json"
        ),
    )

    parser.add_argument(
        "--raw-summary",
        default=(
            "artifacts/domain_shift_v2b/raw/"
            "v2b_raw_summary.csv"
        ),
    )

    parser.add_argument(
        "--v2-summary",
        default=(
            "artifacts/domain_shift_v2b/"
            "canonical_v2/"
            "v2b_canonical_v2_summary.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/domain_shift_v2b/"
            "canonical_v3"
        ),
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
            "Frozen V3 callable as "
            "package.module:function_name"
        ),
    )

    parser.add_argument(
        "--canonicalizer-input",
        choices=[
            "numpy",
            "pil",
        ],
        default="numpy",
        help=(
            "Input expected by V3. "
            "Default: grayscale 2D numpy uint8."
        ),
    )

    parser.add_argument(
        "--save-canonicalized-npz",
        action="store_true",
        help=(
            "Save exact canonicalized V3 inputs "
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
