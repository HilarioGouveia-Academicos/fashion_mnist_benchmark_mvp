import base64
import io

import joblib
import numpy as np

from PIL import Image
from tensorflow import keras

from config.settings import (
    CLASS_NAMES,
    MODELS_DIR,
)

from src.domain_shift.canonicalizer import (
    canonicalize_image,
)

from src.preprocessing.canonicalizer_v3_candidate import (
    canonicalize_image_v3,
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


PREPROCESSING_ALIASES = {
    "canonicalized": "canonicalized_v2",
    "canonicalized_v2": "canonicalized_v2",
    "canonicalized_v3": "canonicalized_v3",
    "raw": "raw",
}

PREPROCESSING_LABELS = {
    "raw": "RAW",
    "canonicalized_v2": "Canonicalized V2",
    "canonicalized_v3": "Canonicalized V3 (GrabCut)",
}


class Predictor:

    def __init__(
        self,
        model_name: str,
        preprocessing_mode: str = "canonicalized_v2",
        use_canonicalization: bool | None = None,
    ):
        self.model_name = model_name

        # Backward compatibility with the previous serving contract.
        if use_canonicalization is not None:
            preprocessing_mode = (
                "canonicalized_v2"
                if use_canonicalization
                else "raw"
            )

        normalized_mode = PREPROCESSING_ALIASES.get(
            preprocessing_mode
        )

        if normalized_mode is None:
            raise ValueError(
                "Preprocessing inválido. Use 'raw', "
                "'canonicalized_v2' ou 'canonicalized_v3'."
            )

        self.preprocessing_mode = normalized_mode

        if model_name == "svm":
            self.model = joblib.load(
                MODELS_DIR / "svm.joblib"
            )

        elif model_name in {"mlp", "cnn"}:
            self.model = keras.models.load_model(
                MODELS_DIR / f"{model_name}.keras", compile=False
            )

        else:
            raise ValueError("Modelo inválido.")

    def _prepare_image(
        self,
        image: Image.Image,
        input_representation: str = "original",
    ) -> np.ndarray:
        """Create the exact 28x28 uint8 representation sent to model preprocessing.

        A representation produced by CV Capture Lab may already contain the
        selected canonicalization. In that case, the transformation is not
        applied again.
        """

        prepared_alias = {
            "raw_28x28": "raw",
            "canonicalized_v2": "canonicalized_v2",
            "canonicalized_v3": "canonicalized_v3",
        }.get(input_representation)

        if prepared_alias is not None:
            if prepared_alias != self.preprocessing_mode:
                raise ValueError(
                    "Representação preparada incompatível com o preprocessing "
                    f"solicitado: {input_representation} -> {self.preprocessing_mode}."
                )

            prepared = np.asarray(
                image.convert("L")
            ).astype(np.uint8)

            if prepared.shape != (28, 28):
                raise ValueError(
                    "Representações preparadas devem possuir tamanho 28x28."
                )

            return prepared

        if input_representation != "original":
            raise ValueError(
                f"input_representation inválida: {input_representation}."
            )

        if self.preprocessing_mode == "raw":
            resized = image.convert("L").resize(
                (28, 28)
            )
            return np.asarray(resized).astype(
                np.uint8
            )

        if self.preprocessing_mode == "canonicalized_v2":
            grayscale = np.asarray(
                image.convert("L")
            ).astype(np.uint8)

            return canonicalize_image(
                grayscale
            )

        # Use grayscale as the neutral interchange representation.
        # V3 converts it internally to BGR before applying GrabCut.
        grayscale = np.asarray(
            image.convert("L")
        ).astype(np.uint8)

        result = canonicalize_image_v3(
            grayscale
        )

        return result.canonical.astype(
            np.uint8
        )

    def _preprocess_array(
        self,
        prepared_image: np.ndarray,
    ) -> np.ndarray:
        batch = prepared_image[
            np.newaxis,
            ...
        ]

        if self.model_name == "svm":
            return preprocess_svm(batch)

        if self.model_name == "mlp":
            return preprocess_mlp(batch)

        return preprocess_cnn(batch)

    def preprocess(
        self,
        image: Image.Image,
        input_representation: str = "original",
    ) -> np.ndarray:
        prepared = self._prepare_image(
            image,
            input_representation=input_representation,
        )
        return self._preprocess_array(prepared)

    @staticmethod
    def _preview_to_base64(
        image: np.ndarray,
    ) -> str:
        """Encode the model-input representation as a lossless PNG preview."""

        preview = Image.fromarray(
            np.clip(image, 0, 255).astype(np.uint8),
            mode="L",
        )

        buffer = io.BytesIO()
        preview.save(buffer, format="PNG")

        return base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")

    def _calculate_uncertainty(
        self,
        probabilities: np.ndarray,
    ) -> dict:
        eps = 1e-12

        probabilities_safe = np.clip(
            probabilities,
            eps,
            1.0,
        )

        entropy = float(
            -np.sum(
                probabilities_safe
                * np.log(probabilities_safe)
            )
        )

        normalized_entropy = float(
            entropy / np.log(
                len(probabilities_safe)
            )
        )

        sorted_probabilities = np.sort(
            probabilities
        )[::-1]

        prediction_margin = float(
            sorted_probabilities[0]
            - sorted_probabilities[1]
        )

        return {
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "prediction_margin": prediction_margin,
        }

    @staticmethod
    def _confidence_level(
        confidence: float,
    ) -> str:
        if confidence >= 0.90:
            return "high_confidence"

        if confidence >= 0.70:
            return "moderate_confidence"

        return "low_confidence"

    def predict(
        self,
        image: Image.Image,
        top_k: int = 3,
        input_representation: str = "original",
    ) -> dict:
        original_size = list(image.size)

        prepared_image = self._prepare_image(
            image,
            input_representation=input_representation,
        )

        x = self._preprocess_array(
            prepared_image
        )

        if self.model_name == "svm":
            probabilities = self.model.predict_proba(
                x
            )[0]
        else:
            probabilities = self.model.predict(
                x,
                verbose=0,
            )[0]

        # Preserve descending order explicitly for every consumer.
        indices = np.argsort(
            probabilities
        )[::-1][:top_k]

        confidence = float(
            probabilities[indices[0]]
        )

        uncertainty = self._calculate_uncertainty(
            probabilities
        )

        return {
            "model": self.model_name,
            "prediction": CLASS_NAMES[
                int(indices[0])
            ],
            "confidence": confidence,
            "top_k": [
                {
                    "rank": rank,
                    "class": CLASS_NAMES[int(i)],
                    "probability": float(
                        probabilities[i]
                    ),
                }
                for rank, i in enumerate(
                    indices,
                    start=1,
                )
            ],
            "uncertainty": uncertainty,
            "input": {
                "original_size": original_size,
                "model_input_size": [28, 28],
                "preprocessing": self.preprocessing_mode,
                "preprocessing_label": PREPROCESSING_LABELS[
                    self.preprocessing_mode
                ],
                "canonicalized": (
                    self.preprocessing_mode != "raw"
                ),
                "canonicalization_version": (
                    "v2"
                    if self.preprocessing_mode == "canonicalized_v2"
                    else (
                        "v3"
                        if self.preprocessing_mode == "canonicalized_v3"
                        else None
                    )
                ),
                "input_representation": input_representation,
                "preprocessing_already_applied": (
                    input_representation != "original"
                ),
                "processed_preview_png_base64": self._preview_to_base64(
                    prepared_image
                ),
            },
            "reliability": {
                "level": self._confidence_level(
                    confidence
                ),
                "domain_warning": True,
                "message": (
                    "External image: model confidence "
                    "does not guarantee correctness."
                ),
            },
        }
