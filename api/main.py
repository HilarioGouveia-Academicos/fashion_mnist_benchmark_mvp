from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from config.settings import MODELS_DIR
from src.inference.predictor import Predictor

app = FastAPI(
    title="Fashion-MNIST ML API",
    version="0.2.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    available = []
    for name, filename in (
        ("svm", "svm.joblib"),
        ("mlp", "mlp.keras"),
        ("cnn", "cnn.keras"),
    ):
        if (MODELS_DIR / filename).exists():
            available.append(name)
    return {"available_models": available}


@app.post("/predict")
async def predict(
    model: str = "cnn",
    preprocessing: str = "canonicalized_v2",
    top_k: int = 3,
    file: UploadFile = File(...),
):
    if model not in {"svm", "mlp", "cnn"}:
        raise HTTPException(
            400,
            "Modelo inválido.",
        )

    valid_preprocessing = {
        "raw",
        "canonicalized",      # backward-compatible alias for V2
        "canonicalized_v2",
        "canonicalized_v3",
    }

    if preprocessing not in valid_preprocessing:
        raise HTTPException(
            400,
            "Preprocessing inválido. Use 'raw', "
            "'canonicalized_v2' ou 'canonicalized_v3'.",
        )

    if not 1 <= top_k <= 10:
        raise HTTPException(
            400,
            "top_k deve estar entre 1 e 10.",
        )

    try:
        contents = await file.read()
        image = Image.open(
            BytesIO(contents)
        )

        predictor = Predictor(
            model_name=model,
            preprocessing_mode=preprocessing,
        )

        return predictor.predict(
            image,
            top_k=top_k,
        )

    except FileNotFoundError:
        raise HTTPException(
            404,
            f"Modelo '{model}' não foi treinado.",
        )

    except Exception as exc:
        raise HTTPException(
            400,
            str(exc),
        )
