import cv2
import numpy as np


def invert_images(images: np.ndarray) -> np.ndarray:
    """
    Inverte as intensidades.

    Preto -> branco
    Branco -> preto
    """
    return 255 - images


def gray_background(
    images: np.ndarray,
    threshold: int = 20,
    background_value: int = 64,
) -> np.ndarray:
    """
    Substitui pixels próximos do preto por um fundo cinza.

    Simula imagens provenientes de outra fonte onde o fundo
    não é completamente preto.
    """
    shifted = images.copy()

    shifted[
        shifted <= threshold
    ] = background_value

    return shifted


def resample_images(
    images: np.ndarray,
    intermediate_size: int = 20,
) -> np.ndarray:
    """
    Reduz a imagem e depois retorna para 28x28.

    Simula diferenças de resolução, interpolação
    ou dispositivo de aquisição.
    """

    output = np.empty_like(images)

    for index, image in enumerate(images):

        smaller = cv2.resize(
            image,
            (
                intermediate_size,
                intermediate_size,
            ),
            interpolation=cv2.INTER_AREA,
        )

        restored = cv2.resize(
            smaller,
            (28, 28),
            interpolation=cv2.INTER_LINEAR,
        )

        output[index] = restored

    return output


def threshold_images(
    images: np.ndarray,
    threshold: int = 100,
) -> np.ndarray:
    """
    Converte as imagens para um estilo de alto contraste.

    Simula uma fonte visual com processamento diferente
    do Fashion-MNIST original.
    """

    _, output = cv2.threshold(
        images,
        threshold,
        255,
        cv2.THRESH_BINARY,
    )

    return output


def centered_resize(
    images: np.ndarray,
    content_size: int = 24,
) -> np.ndarray:
    """
    Redimensiona o objeto para uma região menor
    e o centraliza novamente em 28x28.

    Simula diferenças de enquadramento/escala.
    """

    output = np.zeros_like(images)

    offset = (
        28 - content_size
    ) // 2

    for index, image in enumerate(images):

        resized = cv2.resize(
            image,
            (
                content_size,
                content_size,
            ),
            interpolation=cv2.INTER_AREA,
        )

        canvas = np.zeros(
            (28, 28),
            dtype=np.uint8,
        )

        canvas[
            offset:offset + content_size,
            offset:offset + content_size,
        ] = resized

        output[index] = canvas

    return output


def generate_domain_shift_scenarios(
    images: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Gera os cenários controlados de domain shift.
    """

    return {
        "clean": images,
        "inverted": invert_images(
            images
        ),
        "gray_background": gray_background(
            images
        ),
        "resampled_20": resample_images(
            images,
            intermediate_size=20,
        ),
        "thresholded": threshold_images(
            images,
            threshold=100,
        ),
        "centered_resize_24": centered_resize(
            images,
            content_size=24,
        ),
    }