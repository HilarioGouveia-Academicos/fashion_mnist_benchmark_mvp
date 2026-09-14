from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from src.domain_shift.canonicalizer import (
    canonicalize_external_image,
)


# ============================================================
# CONFIG
# ============================================================

EXTERNAL_DIR = Path(
    "data/external"
)

OUTPUT_DIR = Path(
    "artifacts/domain_shift_v2_1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "canonicalization_examples.png"
)

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
}


# ============================================================
# DISCOVER IMAGES
# ============================================================

def find_external_images() -> list[Path]:

    files = [
        path
        for path in EXTERNAL_DIR.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ]

    return sorted(
        files
    )


# ============================================================
# VISUAL QA
# ============================================================

def main() -> None:

    files = find_external_images()

    if not files:

        raise FileNotFoundError(
            f"Nenhuma imagem encontrada em "
            f"{EXTERNAL_DIR.resolve()}"
        )

    print(
        f"[OK] Imagens encontradas: "
        f"{len(files)}"
    )

    print(
        f"[OK] Diretório de saída: "
        f"{OUTPUT_DIR.resolve()}"
    )

    # Para o piloto atual temos 40 imagens.
    # Vamos visualizar todas para que o QA
    # cubra as 10 classes.
    files = files[:40]

    rows = len(files)

    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(
            7,
            rows * 2.2,
        ),
    )

    # Caso exista somente uma imagem.
    if rows == 1:
        axes = [
            axes
        ]

    for row, image_path in enumerate(
        files
    ):

        original = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if original is None:

            raise ValueError(
                f"Não foi possível carregar: "
                f"{image_path}"
            )

        canonical = (
            canonicalize_external_image(
                image_path
            )
        )

        # ------------------------------
        # ORIGINAL
        # ------------------------------

        axes[row][0].imshow(
            original,
            cmap="gray",
            vmin=0,
            vmax=255,
        )

        axes[row][0].set_title(
            f"{image_path.parent.name}\n"
            f"{image_path.name}",
            fontsize=8,
        )

        axes[row][0].axis(
            "off"
        )

        # ------------------------------
        # CANONICAL
        # ------------------------------

        axes[row][1].imshow(
            canonical,
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )

        axes[row][1].set_title(
            "Canonicalized",
            fontsize=8,
        )

        axes[row][1].axis(
            "off"
        )

    fig.suptitle(
        "Domain Shift V2A.1 — "
        "Canonicalization Visual QA",
        fontsize=14,
        y=1.0,
    )

    plt.tight_layout()

    fig.savefig(
        OUTPUT_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"[OK] Visual QA gerado:"
    )

    print(
        OUTPUT_PATH.resolve()
    )


if __name__ == "__main__":
    main()