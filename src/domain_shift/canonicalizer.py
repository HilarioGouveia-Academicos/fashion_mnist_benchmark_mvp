from pathlib import Path

import cv2
import numpy as np


TARGET_SIZE = 28
CONTENT_SIZE = 22

MIN_BACKGROUND_TOLERANCE = 18
MAX_BACKGROUND_TOLERANCE = 55
MIN_COMPONENT_RATIO = 0.002


def _get_border_pixels(
    image: np.ndarray,
) -> np.ndarray:

    top = image[0, :]
    bottom = image[-1, :]
    left = image[:, 0]
    right = image[:, -1]

    return np.concatenate(
        [
            top,
            bottom,
            left,
            right,
        ]
    )


def _estimate_background(
    image: np.ndarray,
) -> tuple[float, float]:

    border = _get_border_pixels(
        image
    ).astype(np.float32)

    background_value = float(
        np.median(border)
    )

    mad = np.median(
        np.abs(
            border - background_value
        )
    )

    robust_std = float(
        1.4826 * mad
    )

    return (
        background_value,
        robust_std,
    )


def _calculate_background_tolerance(
    background_std: float,
) -> float:

    tolerance = (
        MIN_BACKGROUND_TOLERANCE
        + (1.5 * background_std)
    )

    return float(
        np.clip(
            tolerance,
            MIN_BACKGROUND_TOLERANCE,
            MAX_BACKGROUND_TOLERANCE,
        )
    )


def _build_background_candidate(
    image: np.ndarray,
    background_value: float,
    tolerance: float,
) -> np.ndarray:

    difference = np.abs(
        image.astype(np.float32)
        - background_value
    )

    return (
        difference <= tolerance
    ).astype(np.uint8) * 255


def _keep_border_connected_background(
    background_candidate: np.ndarray,
) -> np.ndarray:

    height, width = (
        background_candidate.shape
    )

    padded = cv2.copyMakeBorder(
        background_candidate,
        1,
        1,
        1,
        1,
        cv2.BORDER_CONSTANT,
        value=255,
    )

    flood_mask = np.zeros(
        (
            padded.shape[0] + 2,
            padded.shape[1] + 2,
        ),
        dtype=np.uint8,
    )

    flooded = padded.copy()

    cv2.floodFill(
        flooded,
        flood_mask,
        seedPoint=(0, 0),
        newVal=128,
    )

    connected_background = (
        flooded == 128
    ).astype(np.uint8) * 255

    return connected_background[
        1:height + 1,
        1:width + 1,
    ]


def _build_foreground_mask(
    image: np.ndarray,
    background_value: float,
    tolerance: float,
) -> np.ndarray:

    background_candidate = (
        _build_background_candidate(
            image,
            background_value,
            tolerance,
        )
    )

    connected_background = (
        _keep_border_connected_background(
            background_candidate
        )
    )

    return cv2.bitwise_not(
        connected_background
    )


def _remove_small_components(
    mask: np.ndarray,
) -> np.ndarray:

    number_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    output = np.zeros_like(
        mask
    )

    image_area = (
        mask.shape[0]
        * mask.shape[1]
    )

    minimum_area = max(
        8,
        int(
            image_area
            * MIN_COMPONENT_RATIO
        ),
    )

    for label in range(
        1,
        number_labels,
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA,
        ]

        if area >= minimum_area:

            output[
                labels == label
            ] = 255

    return output


def _refine_foreground_mask(
    mask: np.ndarray,
) -> np.ndarray:

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    refined = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    return _remove_small_components(
        refined
    )


def _crop_foreground(
    image: np.ndarray,
    mask: np.ndarray,
    margin_ratio: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:

    coordinates = cv2.findNonZero(
        mask
    )

    if coordinates is None:

        return (
            image,
            mask,
        )

    x, y, w, h = cv2.boundingRect(
        coordinates
    )

    margin = max(
        1,
        int(
            max(w, h)
            * margin_ratio
        ),
    )

    x1 = max(
        0,
        x - margin,
    )

    y1 = max(
        0,
        y - margin,
    )

    x2 = min(
        image.shape[1],
        x + w + margin,
    )

    y2 = min(
        image.shape[0],
        y + h + margin,
    )

    return (
        image[
            y1:y2,
            x1:x2,
        ],
        mask[
            y1:y2,
            x1:x2,
        ],
    )


def _normalize_polarity(
    image: np.ndarray,
    mask: np.ndarray,
    background_value: float,
) -> np.ndarray:

    output = image.copy()

    if background_value > 127:

        output = (
            255 - output
        )

    output[
        mask == 0
    ] = 0

    return output


def _ensure_visible_foreground(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:

    output = image.copy()

    foreground = output[
        mask > 0
    ]

    if foreground.size == 0:

        return output

    mean_foreground = float(
        foreground.mean()
    )

    if mean_foreground < 60:

        scale = min(
            2.0,
            100.0
            / max(
                mean_foreground,
                1.0,
            ),
        )

        adjusted = (
            output.astype(np.float32)
            * scale
        )

        output = np.clip(
            adjusted,
            0,
            255,
        ).astype(np.uint8)

        output[
            mask == 0
        ] = 0

    return output


def _resize_preserving_aspect_ratio(
    image: np.ndarray,
    content_size: int,
) -> np.ndarray:

    height, width = (
        image.shape
    )

    if height == 0 or width == 0:

        return np.zeros(
            (
                content_size,
                content_size,
            ),
            dtype=np.uint8,
        )

    scale = min(
        content_size / width,
        content_size / height,
    )

    new_width = max(
        1,
        int(
            round(
                width * scale
            )
        ),
    )

    new_height = max(
        1,
        int(
            round(
                height * scale
            )
        ),
    )

    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_LINEAR
    )

    return cv2.resize(
        image,
        (
            new_width,
            new_height,
        ),
        interpolation=interpolation,
    )


def _center_on_canvas(
    image: np.ndarray,
    target_size: int = TARGET_SIZE,
    content_size: int = CONTENT_SIZE,
) -> np.ndarray:

    resized = (
        _resize_preserving_aspect_ratio(
            image,
            content_size,
        )
    )

    height, width = (
        resized.shape
    )

    canvas = np.zeros(
        (
            target_size,
            target_size,
        ),
        dtype=np.uint8,
    )

    x_offset = (
        target_size - width
    ) // 2

    y_offset = (
        target_size - height
    ) // 2

    canvas[
        y_offset:y_offset + height,
        x_offset:x_offset + width,
    ] = resized

    return canvas


def canonicalize_image(
    image: np.ndarray,
) -> np.ndarray:

    if image.ndim != 2:

        raise ValueError(
            "canonicalize_image espera "
            "uma imagem grayscale 2D."
        )

    if image.dtype != np.uint8:

        image = np.clip(
            image,
            0,
            255,
        ).astype(np.uint8)

    (
        background_value,
        background_std,
    ) = _estimate_background(
        image
    )

    tolerance = (
        _calculate_background_tolerance(
            background_std
        )
    )

    foreground_mask = (
        _build_foreground_mask(
            image,
            background_value,
            tolerance,
        )
    )

    foreground_mask = (
        _refine_foreground_mask(
            foreground_mask
        )
    )

    (
        cropped_image,
        cropped_mask,
    ) = _crop_foreground(
        image,
        foreground_mask,
    )

    normalized = (
        _normalize_polarity(
            cropped_image,
            cropped_mask,
            background_value,
        )
    )

    normalized = (
        _ensure_visible_foreground(
            normalized,
            cropped_mask,
        )
    )

    return _center_on_canvas(
        normalized
    )


def canonicalize_external_image(
    image_path: Path,
) -> np.ndarray:

    image_path = Path(
        image_path
    )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:

        raise ValueError(
            f"Não foi possível carregar "
            f"a imagem: {image_path}"
        )

    return canonicalize_image(
        image
    )