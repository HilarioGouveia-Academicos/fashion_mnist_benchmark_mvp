import numpy as np

from src.data.loader import normalize_images, flatten_images


def preprocess_mlp(images: np.ndarray) -> np.ndarray:
    """
    Aplica o pré-processamento padrão utilizado pelo modelo MLP.

    Etapas
    ------
    1. Normaliza os pixels para o intervalo [0, 1].
    2. Converte imagens 28x28 em vetores de 784 features.

    Parameters
    ----------
    images : np.ndarray
        Imagens no formato (n_samples, 28, 28).

    Returns
    -------
    np.ndarray
        Dados pré-processados no formato (n_samples, 784).
    """
    normalized_images = normalize_images(images)

    return flatten_images(normalized_images)