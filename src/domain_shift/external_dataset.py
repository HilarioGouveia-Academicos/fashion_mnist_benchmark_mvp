from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.domain_shift.canonicalizer import canonicalize_external_image


CLASS_MAPPING = {
    "tshirt_top": 0,
    "trouser": 1,
    "pullover": 2,
    "dress": 3,
    "coat": 4,
    "sandal": 5,
    "shirt": 6,
    "sneaker": 7,
    "bag": 8,
    "ankle_boot": 9,
}


CLASS_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
}


def load_external_image(
    image_path: Path,
) -> np.ndarray:
    """
    Carrega uma imagem externa e a converte para
    o formato espacial do Fashion-MNIST.

    Saída:
        shape=(28, 28)
        dtype=uint8
        range=[0, 255]

    Importante:
        Não normaliza.
        Não inverte.
        Não remove fundo.

    O preprocessing específico de cada modelo
    será aplicado posteriormente.
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(
            f"Não foi possível carregar: {image_path}"
        )

    image = cv2.resize(
        image,
        (28, 28),
        interpolation=cv2.INTER_AREA,
    )

    return image.astype(
        np.uint8
    )


def load_external_dataset(
    root_dir: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    """
    Carrega dataset externo organizado por
    diretórios correspondentes às classes.

    Retorna:
        X
        y
        metadata
    """

    if not root_dir.exists():
        raise FileNotFoundError(
            f"Diretório não encontrado: {root_dir}"
        )

    images = []
    labels = []
    metadata_rows = []

    for folder_name, class_id in (
        CLASS_MAPPING.items()
    ):

        class_dir = (
            root_dir
            / folder_name
        )

        if not class_dir.exists():

            print(
                f"[WARNING] Classe ausente: "
                f"{folder_name}"
            )

            continue

        image_paths = sorted(
            [
                path
                for path in class_dir.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in SUPPORTED_EXTENSIONS
                )
            ]
        )

        for image_path in image_paths:

            try:

                image = load_external_image(
                    image_path
                )

            except Exception as exc:

                print(
                    f"[WARNING] Ignorando "
                    f"{image_path}: {exc}"
                )

                continue

            images.append(
                image
            )

            labels.append(
                class_id
            )

            metadata_rows.append(
                {
                    "file": str(
                        image_path
                    ),
                    "class_id": (
                        class_id
                    ),
                    "class_name": (
                        CLASS_NAMES[
                            class_id
                        ]
                    ),
                }
            )

    if not images:

        raise RuntimeError(
            "Nenhuma imagem externa válida foi encontrada."
        )

    X = np.stack(
        images
    )

    y = np.asarray(
        labels,
        dtype=np.int64,
    )

    metadata = pd.DataFrame(
        metadata_rows
    )

    return (
        X,
        y,
        metadata,
    )



def load_canonicalized_dataset(
    metadata: pd.DataFrame,
) -> np.ndarray:
    """
    Gera o dataset canonicalizado usando
    exatamente os mesmos arquivos do V2A Raw.
    """

    images = []

    for _, row in metadata.iterrows():

        image_path = Path(
            row["file"]
        )

        image = (
            canonicalize_external_image(
                image_path
            )
        )

        images.append(
            image
        )

    return np.stack(
        images
    )