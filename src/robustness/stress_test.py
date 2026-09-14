

from pathlib import Path
import json

import cv2
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

from src.data.loader import load_fashion_mnist

from src.preprocessing.svm import preprocess_svm
from src.preprocessing.mlp import preprocess_mlp
from src.preprocessing.cnn import preprocess_cnn

from src.robustness.plots import (
    generate_robustness_plots,
)

from src.robustness.report import (
    generate_robustness_report,
)




# =========================================================
# Configuration
# =========================================================

ROBUSTNESS_DIR = ARTIFACTS_DIR / "robustness"

ROBUSTNESS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_STATE = 42


# =========================================================
# Severity Configuration
# =========================================================

SEVERITY_LEVELS = {

    "noise": [
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
    ],

    "rotation": [
        5,
        10,
        20,
        30,
        45,
    ],

    "blur": [
        3,
        5,
        7,
        9,
    ],

    "contrast": [
        0.90,
        0.70,
        0.50,
        0.30,
        0.15,
    ],

    "brightness": [
        0.30,
        0.50,
        0.70,
        1.30,
        1.60,
    ],

    "occlusion": [
        4,
        6,
        8,
        10,
        14,
    ],
}


# =========================================================
# Corruption Functions
# =========================================================

def add_noise(
    images: np.ndarray,
    severity: float,
) -> np.ndarray:
    """
    Adiciona ruído Gaussiano.

    severity é expressa na escala normalizada [0, 1].

    As imagens de entrada permanecem na escala original
    Fashion-MNIST [0, 255].
    """

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    images_float = (
        images.astype(np.float32)
        / 255.0
    )

    noise = rng.normal(
        loc=0.0,
        scale=severity,
        size=images.shape,
    )

    corrupted = np.clip(
        images_float + noise,
        0.0,
        1.0,
    )

    return (
        corrupted * 255.0
    ).astype(np.uint8)


def rotate(
    images: np.ndarray,
    angle: float,
) -> np.ndarray:
    """
    Rotaciona as imagens mantendo tamanho 28x28.
    """

    center = (
        images.shape[2] / 2,
        images.shape[1] / 2,
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    rotated = [

        cv2.warpAffine(
            image,
            matrix,
            (28, 28),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        for image in images
    ]

    return np.asarray(
        rotated,
        dtype=np.uint8,
    )


def blur(
    images: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    """
    Aplica Gaussian Blur.

    kernel_size deve ser ímpar.
    """

    if kernel_size % 2 == 0:

        raise ValueError(
            "kernel_size deve ser um número ímpar."
        )

    blurred = [

        cv2.GaussianBlur(
            image,
            (kernel_size, kernel_size),
            0,
        )

        for image in images
    ]

    return np.asarray(
        blurred,
        dtype=np.uint8,
    )


def contrast(
    images: np.ndarray,
    factor: float,
) -> np.ndarray:
    """
    Modifica o contraste das imagens.

    factor < 1 reduz contraste.
    """

    images_float = (
        images.astype(np.float32)
        / 255.0
    )

    corrupted = (
        (images_float - 0.5)
        * factor
        + 0.5
    )

    corrupted = np.clip(
        corrupted,
        0.0,
        1.0,
    )

    return (
        corrupted * 255.0
    ).astype(np.uint8)


def brightness(
    images: np.ndarray,
    factor: float,
) -> np.ndarray:
    """
    Modifica o brilho.

    factor < 1 escurece.
    factor > 1 clareia.
    """

    images_float = (
        images.astype(np.float32)
        / 255.0
    )

    corrupted = np.clip(
        images_float * factor,
        0.0,
        1.0,
    )

    return (
        corrupted * 255.0
    ).astype(np.uint8)


def occlusion(
    images: np.ndarray,
    size: int,
) -> np.ndarray:
    """
    Cria uma oclusão quadrada central.

    size representa o tamanho do quadrado removido.
    """

    corrupted = images.copy()

    height = images.shape[1]
    width = images.shape[2]

    start_y = (
        height - size
    ) // 2

    start_x = (
        width - size
    ) // 2

    end_y = (
        start_y + size
    )

    end_x = (
        start_x + size
    )

    corrupted[
        :,
        start_y:end_y,
        start_x:end_x,
    ] = 0

    return corrupted


# =========================================================
# Scenario Generator
# =========================================================

def generate_stress_scenarios(
    images: np.ndarray,
) -> list[dict]:
    """
    Gera todos os cenários do stress test.

    Returns
    -------
    list[dict]

    Cada cenário contém:

    corruption
    severity
    images
    """

    scenarios = []

    # -----------------------------------------------------
    # Clean baseline
    # -----------------------------------------------------

    scenarios.append({
        "corruption": "clean",
        "severity": 0,
        "images": images,
    })

    # -----------------------------------------------------
    # Noise
    # -----------------------------------------------------

    for severity in SEVERITY_LEVELS[
        "noise"
    ]:

        scenarios.append({
            "corruption": "noise",
            "severity": severity,
            "images": add_noise(
                images,
                severity=severity,
            ),
        })

    # -----------------------------------------------------
    # Rotation
    # -----------------------------------------------------

    for angle in SEVERITY_LEVELS[
        "rotation"
    ]:

        scenarios.append({
            "corruption": "rotation",
            "severity": angle,
            "images": rotate(
                images,
                angle=angle,
            ),
        })

    # -----------------------------------------------------
    # Blur
    # -----------------------------------------------------

    for kernel_size in SEVERITY_LEVELS[
        "blur"
    ]:

        scenarios.append({
            "corruption": "blur",
            "severity": kernel_size,
            "images": blur(
                images,
                kernel_size=kernel_size,
            ),
        })

    # -----------------------------------------------------
    # Contrast
    # -----------------------------------------------------

    for factor in SEVERITY_LEVELS[
        "contrast"
    ]:

        scenarios.append({
            "corruption": "contrast",
            "severity": factor,
            "images": contrast(
                images,
                factor=factor,
            ),
        })

    # -----------------------------------------------------
    # Brightness
    # -----------------------------------------------------

    for factor in SEVERITY_LEVELS[
        "brightness"
    ]:

        scenarios.append({
            "corruption": "brightness",
            "severity": factor,
            "images": brightness(
                images,
                factor=factor,
            ),
        })

    # -----------------------------------------------------
    # Occlusion
    # -----------------------------------------------------

    for size in SEVERITY_LEVELS[
        "occlusion"
    ]:

        scenarios.append({
            "corruption": "occlusion",
            "severity": size,
            "images": occlusion(
                images,
                size=size,
            ),
        })

    return scenarios


# =========================================================
# Model Loading
# =========================================================

def load_models() -> dict:
    """
    Carrega os três modelos persistidos.
    """

    models = {}

    svm_path = (
        MODELS_DIR / "svm.joblib"
    )

    mlp_path = (
        MODELS_DIR / "mlp.keras"
    )

    cnn_path = (
        MODELS_DIR / "cnn.keras"
    )

    # -----------------------------------------------------
    # SVM
    # -----------------------------------------------------

    if svm_path.exists():

        models["svm"] = joblib.load(
            svm_path
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

    # -----------------------------------------------------
    # MLP
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # CNN
    # -----------------------------------------------------

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


# =========================================================
# Preprocessing Router
# =========================================================

def preprocess_for_model(
    model_name: str,
    images: np.ndarray,
) -> np.ndarray:
    """
    Usa o preprocessing oficial de cada arquitetura.
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
        f"Modelo não suportado: "
        f"{model_name}"
    )


# =========================================================
# Prediction Router
# =========================================================

def predict_model(
    model_name: str,
    model,
    x: np.ndarray,
) -> np.ndarray:
    """
    Realiza a inferência conforme o framework.
    """

    if model_name == "svm":

        return model.predict(x)

    probabilities = model.predict(
        x,
        verbose=0,
    )

    return np.argmax(
        probabilities,
        axis=1,
    )


# =========================================================
# Evaluation
# =========================================================

def evaluate(
    model_name: str,
    model,
    images: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """
    Executa preprocessing, inferência e métricas.
    """

    x = preprocess_for_model(
        model_name,
        images,
    )

    y_pred = predict_model(
        model_name,
        model,
        x,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(
            accuracy
        ),
        "f1_macro": float(
            f1
        ),
    }


# =========================================================
# Summary
# =========================================================

def build_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Constrói resumo consolidado excluindo cenário clean.
    """

    corrupted = results[
        results["corruption"] != "clean"
    ]

    summary = (
        corrupted
        .groupby("model")
        .agg(
            mean_corrupted_accuracy=(
                "accuracy",
                "mean",
            ),
            min_corrupted_accuracy=(
                "accuracy",
                "min",
            ),
            mean_corrupted_f1=(
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
            mean_f1_drop=(
                "f1_drop",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "mean_accuracy_drop"
        )
    )

    return summary


# =========================================================
# Corruption Summary
# =========================================================

def build_corruption_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Consolida resultado médio por modelo e corrupção.
    """

    corrupted = results[
        results["corruption"] != "clean"
    ]

    return (
        corrupted
        .groupby([
            "model",
            "corruption",
        ])
        .agg(
            mean_accuracy=(
                "accuracy",
                "mean",
            ),
            min_accuracy=(
                "accuracy",
                "min",
            ),
            mean_f1=(
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
        )
        .reset_index()
    )


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    _, (
        x_test,
        y_test,
    ) = load_fashion_mnist()

    print(
        "\n========================================"
    )

    print(
        "FASHION-MNIST ROBUSTNESS STRESS TEST V2"
    )

    print(
        "========================================"
    )

    print(
        f"\nTest original: "
        f"{x_test.shape}"
    )

    # -----------------------------------------------------
    # Models
    # -----------------------------------------------------

    models = load_models()

    if not models:

        raise RuntimeError(
            "Nenhum modelo disponível."
        )

    # -----------------------------------------------------
    # Scenarios
    # -----------------------------------------------------

    scenarios = (
        generate_stress_scenarios(
            x_test
        )
    )

    print(
        f"\nCenários gerados: "
        f"{len(scenarios)}"
    )

    print(
        f"Modelos: "
        f"{len(models)}"
    )

    print(
        f"Avaliações totais: "
        f"{len(scenarios) * len(models)}"
    )

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    rows = []

    for model_name, model in (
        models.items()
    ):

        print(
            "\n========================================"
        )

        print(
            f"Modelo: {model_name.upper()}"
        )

        print(
            "========================================"
        )

        # -------------------------------------------------
        # Clean baseline
        # -------------------------------------------------

        clean_metrics = evaluate(
            model_name=model_name,
            model=model,
            images=x_test,
            y_true=y_test,
        )

        clean_accuracy = (
            clean_metrics["accuracy"]
        )

        clean_f1 = (
            clean_metrics["f1_macro"]
        )

        print(
            f"\nBaseline clean | "
            f"accuracy={clean_accuracy:.4f} | "
            f"f1={clean_f1:.4f}"
        )

        # -------------------------------------------------
        # Scenarios
        # -------------------------------------------------

        for scenario in scenarios:

            corruption_name = (
                scenario["corruption"]
            )

            severity = (
                scenario["severity"]
            )

            images = (
                scenario["images"]
            )

            if corruption_name == "clean":

                metrics = clean_metrics

            else:

                metrics = evaluate(
                    model_name=model_name,
                    model=model,
                    images=images,
                    y_true=y_test,
                )

            accuracy_drop = (
                clean_accuracy
                - metrics["accuracy"]
            )

            f1_drop = (
                clean_f1
                - metrics["f1_macro"]
            )

            rows.append({
                "model": model_name,
                "corruption":
                    corruption_name,
                "severity":
                    severity,
                "accuracy":
                    metrics["accuracy"],
                "f1_macro":
                    metrics["f1_macro"],
                "accuracy_drop":
                    accuracy_drop,
                "f1_drop":
                    f1_drop,
            })

            print(
                f"{corruption_name:12s} | "
                f"severity={severity!s:5s} | "
                f"acc={metrics['accuracy']:.4f} | "
                f"f1={metrics['f1_macro']:.4f} | "
                f"drop={accuracy_drop:.4f}"
            )

    # =====================================================
    # DataFrame
    # =====================================================

    results = pd.DataFrame(
        rows
    )

    # =====================================================
    # Summary
    # =====================================================

    summary = build_summary(
        results
    )

    corruption_summary = (
        build_corruption_summary(
            results
        )
    )

    # =====================================================
    # Persistence - CSV
    # =====================================================

    results_path = (
        ROBUSTNESS_DIR
        / "robustness_results.csv"
    )

    summary_path = (
        ROBUSTNESS_DIR
        / "robustness_summary.csv"
    )

    corruption_summary_path = (
        ROBUSTNESS_DIR
        / "corruption_summary.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    corruption_summary.to_csv(
        corruption_summary_path,
        index=False,
    )

    # =====================================================
    # Persistence - JSON
    # =====================================================

    json_path = (
        ROBUSTNESS_DIR
        / "robustness_results.json"
    )

    records = (
        results
        .to_dict(
            orient="records"
        )
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            records,
            file,
            indent=4,
        )


    # =========================================================
    # Visual Evidence
    # =========================================================
    generate_robustness_plots(
        results=results,
        output_dir=ROBUSTNESS_DIR,
    )

    # =========================================================
    # Markdown Report
    # =========================================================
    generate_robustness_report(
        results=results,
        summary=summary,
        corruption_summary=corruption_summary,
        output_dir=ROBUSTNESS_DIR,
    )


    # =====================================================
    # Terminal Summary
    # =====================================================
    print("\n\n========================================")
    print("ROBUSTNESS SUMMARY")
    print("========================================")
    print(summary.to_string(index=False))
    print("\n\n========================================")
    print("CORRUPTION SUMMARY")
    print("========================================")
    print(corruption_summary.to_string(index=False))

    # =====================================================
    # Output
    # =====================================================
    print("\n\nArquivos gerados:")
    print(f"- {results_path}")
    print(f"- {summary_path}")
    print(f"- {corruption_summary_path}")
    print(f"- {json_path}")


    print("\n\nArquivos gerados:")
    print(f"- {ROBUSTNESS_DIR / 'robustness_report.md'}")
    print(f"- {ROBUSTNESS_DIR / 'robustness_heatmap.png'}")
    print(f"- {ROBUSTNESS_DIR / 'clean_vs_worst_case.png'}")
    print(f"- {ROBUSTNESS_DIR / 'mean_accuracy_drop.png'}")


if __name__ == "__main__":
    main()