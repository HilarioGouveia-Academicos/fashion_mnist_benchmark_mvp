from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ROOT_DIR / "models"

EXPERIMENT_NAME = "Fashion-MNIST-Benchmark"
# MLFLOW_TRACKING_URI = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

RANDOM_STATE = 42

SVM_TRAIN_SAMPLES = 12000
MLP_EPOCHS = 5
CNN_EPOCHS = 5
BATCH_SIZE = 128

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

for directory in (ARTIFACTS_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
