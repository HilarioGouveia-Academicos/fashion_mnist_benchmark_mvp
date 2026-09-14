import numpy as np

from src.data.loader import normalize_images


def preprocess_cnn(images: np.ndarray) -> np.ndarray:
    """
    Aplica o pré-processamento padrão utilizado pela CNN.

    Etapas
    ------
    1. Normaliza pixels para o intervalo [0, 1].
    2. Adiciona a dimensão do canal:
       (N, 28, 28) -> (N, 28, 28, 1)

    Parameters
    ----------
    images : np.ndarray
        Imagens no formato (n_samples, 28, 28).

    Returns
    -------
    np.ndarray
        Imagens no formato (n_samples, 28, 28, 1).
    """

    normalized_images = normalize_images(images)

    return normalized_images[..., np.newaxis]