from pathlib import Path

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
# Corruptions
# =========================================================

def rotate(
    images: np.ndarray,
    angle: float = 20.0,
) -> np.ndarray:
    """
    Rotaciona imagens preservando o formato 28x28.

    Parameters
    ----------
    images : np.ndarray
        Imagens originais no intervalo [0, 255].

    angle : float
        Ângulo de rotação em graus.

    Returns
    -------
    np.ndarray
        Imagens rotacionadas.
    """

    matrix = cv2.getRotationMatrix2D(
        (14, 14),
        angle,
        1.0,
    )

    rotated = [
        cv2.warpAffine(
            image,
            matrix,
            (28, 28),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        for image in images
    ]

    return np.asarray(
        rotated,
        dtype=np.uint8,
    )


def add_noise(
    images: np.ndarray,
    severity: float = 0.15,
) -> np.ndarray:
    """
    Adiciona ruído Gaussiano.

    severity é expressa na escala normalizada [0,1],
    embora as imagens de entrada permaneçam em [0,255].
    """

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    images_float = (
        images.astype(np.float32) / 255.0
    )

    noise = rng.normal(
        loc=0.0,
        scale=severity,
        size=images.shape,
    )

    noisy = np.clip(
        images_float + noise,
        0.0,
        1.0,
    )

    return (
        noisy * 255.0
    ).astype(np.uint8)


def blur(
    images: np.ndarray,
) -> np.ndarray:
    """
    Aplica Gaussian Blur 3x3.
    """

    blurred = [
        cv2.GaussianBlur(
            image,
            (3, 3),
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
    factor: float = 0.5,
) -> np.ndarray:
    """
    Reduz o contraste das imagens.

    factor < 1 reduz contraste.
    factor > 1 aumenta contraste.
    """

    images_float = (
        images.astype(np.float32) / 255.0
    )

    adjusted = (
        (images_float - 0.5) * factor
        + 0.5
    )

    adjusted = np.clip(
        adjusted,
        0.0,
        1.0,
    )

    return (
        adjusted * 255.0
    ).astype(np.uint8)


# =========================================================
# Model loading
# =========================================================

def load_models() -> dict:
    """
    Carrega os modelos persistidos localmente.
    """

    models = {}

    svm_path = MODELS_DIR / "svm.joblib"

    mlp_path = MODELS_DIR / "mlp.keras"

    cnn_path = MODELS_DIR / "cnn.keras"

    if svm_path.exists():

        models["svm"] = joblib.load(
            svm_path
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

    else:

        print(
            f"[WARNING] CNN não encontrado: "
            f"{cnn_path}"
        )

    return models


# =========================================================
# Preprocessing
# =========================================================

def preprocess_for_model(
    model_name: str,
    images: np.ndarray,
) -> np.ndarray:
    """
    Aplica o preprocessing oficial de cada modelo.
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
        f"Modelo desconhecido: {model_name}"
    )


# =========================================================
# Prediction
# =========================================================

def predict_model(
    model_name: str,
    model,
    x: np.ndarray,
) -> np.ndarray:
    """
    Executa predição considerando diferenças entre
    Scikit-Learn e TensorFlow/Keras.
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

def evaluate_model(
    model_name: str,
    model,
    images: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Avalia um modelo em determinada versão do dataset.
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

    return {
        "accuracy": accuracy_score(
            y_test,
            y_pred,
        ),
        "f1_macro": f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    _, (x_test, y_test) = (
        load_fashion_mnist()
    )

    print(
        f"Test original: {x_test.shape}"
    )

    # IMPORTANT:
    # x_test continua em sua representação original.
    #
    # As corrupções acontecem antes do preprocessing.

    transforms = {
        "clean": x_test,
        "noise": add_noise(
            x_test,
            severity=0.15,
        ),
        "rotation": rotate(
            x_test,
            angle=20,
        ),
        "blur": blur(
            x_test,
        ),
        "contrast": contrast(
            x_test,
            factor=0.5,
        ),
    }

    # -----------------------------------------------------
    # Models
    # -----------------------------------------------------

    models = load_models()

    if not models:

        raise RuntimeError(
            "Nenhum modelo disponível para "
            "o stress test."
        )

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    rows = []

    for model_name, model in models.items():

        print(
            f"\nModelo: {model_name.upper()}"
        )

        model_results = {}

        for transform_name, images in (
            transforms.items()
        ):

            metrics = evaluate_model(
                model_name=model_name,
                model=model,
                images=images,
                y_test=y_test,
            )

            model_results[
                transform_name
            ] = metrics

        # ---------------------------------------------
        # Clean baseline
        # ---------------------------------------------

        clean_accuracy = (
            model_results["clean"][
                "accuracy"
            ]
        )

        clean_f1 = (
            model_results["clean"][
                "f1_macro"
            ]
        )

        # ---------------------------------------------
        # Degradation
        # ---------------------------------------------

        for transform_name, metrics in (
            model_results.items()
        ):

            accuracy_drop = (
                clean_accuracy
                - metrics["accuracy"]
            )

            f1_drop = (
                clean_f1
                - metrics["f1_macro"]
            )

            row = {
                "model": model_name,
                "transform": transform_name,
                "accuracy":
                    metrics["accuracy"],
                "f1_macro":
                    metrics["f1_macro"],
                "accuracy_drop":
                    accuracy_drop,
                "f1_drop":
                    f1_drop,
            }

            rows.append(row)

            print(
                f"{transform_name:10s} | "
                f"accuracy="
                f"{metrics['accuracy']:.4f} | "
                f"f1="
                f"{metrics['f1_macro']:.4f} | "
                f"drop="
                f"{accuracy_drop:.4f}"
            )

    # -----------------------------------------------------
    # DataFrame
    # -----------------------------------------------------

    results = pd.DataFrame(rows)

    # -----------------------------------------------------
    # Ranking robustness
    # -----------------------------------------------------

    corruption_results = results[
        results["transform"] != "clean"
    ]

    robustness_summary = (
        corruption_results
        .groupby("model")
        .agg(
            mean_corrupted_accuracy=(
                "accuracy",
                "mean",
            ),
            mean_corrupted_f1=(
                "f1_macro",
                "mean",
            ),
            mean_accuracy_drop=(
                "accuracy_drop",
                "mean",
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

    # -----------------------------------------------------
    # Persistence
    # -----------------------------------------------------

    results_path = (
        ROBUSTNESS_DIR
        / "robustness_results.csv"
    )

    summary_path = (
        ROBUSTNESS_DIR
        / "robustness_summary.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    robustness_summary.to_csv(
        summary_path,
        index=False,
    )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "ROBUSTNESS SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        robustness_summary.to_string(
            index=False
        )
    )

    print(
        "\nArquivos gerados:"
    )

    print(
        results_path
    )

    print(
        summary_path
    )


if __name__ == "__main__":
    main()