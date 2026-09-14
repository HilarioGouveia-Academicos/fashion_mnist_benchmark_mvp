
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


TARGET_SIZE = 28
CONTENT_SIZE = 22
VERSION = "v3-candidate-grabcut-02-focused-qa"


@dataclass(frozen=True)
class CanonicalizationResult:
    canonical: np.ndarray
    mask: np.ndarray
    roi: np.ndarray


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image deve ser np.ndarray")

    if image.ndim not in (2, 3):
        raise ValueError("Imagem deve ser grayscale, BGR ou BGRA.")

    if image.size == 0:
        raise ValueError("Imagem vazia.")


def _to_bgr(image: np.ndarray) -> np.ndarray:
    _validate_image(image)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 3:
        return image.copy()

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    raise ValueError("Imagem 3D deve ter 3 ou 4 canais.")


def _grabcut_mask(
    bgr: np.ndarray,
    border_ratio: float = 0.04,
    iterations: int = 5,
) -> np.ndarray:
    """
    Segmentação candidata para o QA focal.

    Motivo da V3:
    reduzir dependência da diferença de intensidade foreground/background
    observada no V2, especialmente em casos low-contrast.
    """
    h, w = bgr.shape[:2]

    if h < 8 or w < 8:
        raise ValueError("Imagem pequena demais para GrabCut.")

    mx = max(2, int(round(w * border_ratio)))
    my = max(2, int(round(h * border_ratio)))

    rect_w = w - 2 * mx
    rect_h = h - 2 * my

    if rect_w <= 1 or rect_h <= 1:
        raise ValueError("Retângulo inicial inválido para GrabCut.")

    rect = (mx, my, rect_w, rect_h)

    gc_mask = np.zeros((h, w), dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(
        bgr,
        gc_mask,
        rect,
        bg_model,
        fg_model,
        iterations,
        cv2.GC_INIT_WITH_RECT,
    )

    return np.where(
        (gc_mask == cv2.GC_FGD)
        | (gc_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)


def _refine_mask(mask: np.ndarray) -> np.ndarray:
    """
    Refino conservador:
    - fecha pequenas lacunas;
    - remove somente componentes muito pequenos/periféricos;
    - evita assumir que o maior componente é sempre o objeto inteiro.
    """
    h, w = mask.shape

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

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        refined,
        connectivity=8,
    )

    if n <= 1:
        return refined

    cleaned = np.zeros_like(refined)

    min_area = max(
        8,
        int(h * w * 0.0015),
    )

    cx = w / 2.0
    cy = h / 2.0

    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        ccx, ccy = centroids[label]

        dx = (ccx - cx) / max(w / 2.0, 1.0)
        dy = (ccy - cy) / max(h / 2.0, 1.0)
        center_distance = dx * dx + dy * dy

        if area >= min_area and center_distance < 1.25:
            cleaned[labels == label] = 255

    if cv2.countNonZero(cleaned) < min_area:
        return refined

    return cleaned


def _crop_with_margin(
    image: np.ndarray,
    mask: np.ndarray,
    margin_ratio: float = 0.06,
) -> tuple[np.ndarray, np.ndarray]:
    coords = cv2.findNonZero(mask)

    if coords is None:
        return image.copy(), mask.copy()

    x, y, w, h = cv2.boundingRect(coords)
    margin = max(
        2,
        int(round(max(w, h) * margin_ratio)),
    )

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image.shape[1], x + w + margin)
    y2 = min(image.shape[0], y + h + margin)

    return (
        image[y1:y2, x1:x2].copy(),
        mask[y1:y2, x1:x2].copy(),
    )


def _to_visible_foreground(
    roi_bgr: np.ndarray,
    roi_mask: np.ndarray,
) -> np.ndarray:
    """
    Produz objeto claro em fundo preto.

    Importante:
    esta etapa só atua depois da máscara. Ela não deve compensar
    uma segmentação semanticamente errada.
    """
    gray = cv2.cvtColor(
        roi_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    foreground_values = gray[roi_mask > 0]

    output = np.zeros_like(gray)

    if foreground_values.size == 0:
        return output

    candidate = gray.copy()

    if float(foreground_values.mean()) < 128.0:
        candidate = 255 - candidate

    values = candidate[roi_mask > 0]

    p5, p95 = np.percentile(
        values,
        [5, 95],
    )

    dynamic_range = float(p95 - p5)

    if dynamic_range > 20:
        normalized = (
            (candidate.astype(np.float32) - p5)
            / dynamic_range
        )

        # Evita saturar todo o objeto em branco.
        normalized = normalized * 175.0 + 80.0

        candidate = np.clip(
            normalized,
            0,
            255,
        ).astype(np.uint8)
    else:
        # Contraste interno muito baixo:
        # preserva forma sem inventar textura.
        candidate = np.maximum(
            candidate,
            110,
        ).astype(np.uint8)

    output[roi_mask > 0] = candidate[roi_mask > 0]
    return output


def _center_on_canvas(
    image: np.ndarray,
    content_size: int = CONTENT_SIZE,
    target_size: int = TARGET_SIZE,
) -> np.ndarray:
    h, w = image.shape

    if h == 0 or w == 0:
        return np.zeros(
            (target_size, target_size),
            dtype=np.uint8,
        )

    scale = min(
        content_size / w,
        content_size / h,
    )

    new_w = max(
        1,
        int(round(w * scale)),
    )

    new_h = max(
        1,
        int(round(h * scale)),
    )

    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_LINEAR
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=interpolation,
    )

    canvas = np.zeros(
        (target_size, target_size),
        dtype=np.uint8,
    )

    x = (target_size - new_w) // 2
    y = (target_size - new_h) // 2

    canvas[
        y:y + new_h,
        x:x + new_w,
    ] = resized

    return canvas


def canonicalize_image_v3(
    image: np.ndarray,
) -> CanonicalizationResult:
    """
    V3 candidate para QA focal.

    Pipeline:
        image
          ↓
        GrabCut
          ↓
        conservative mask refinement
          ↓
        ROI / crop
          ↓
        visible foreground
          ↓
        resize preserving aspect ratio
          ↓
        center on 28×28

    Não aprovado para V2B sem Visual QA.
    """
    bgr = _to_bgr(image)

    mask = _grabcut_mask(bgr)
    mask = _refine_mask(mask)

    roi_bgr, roi_mask = _crop_with_margin(
        bgr,
        mask,
    )

    normalized = _to_visible_foreground(
        roi_bgr,
        roi_mask,
    )

    canonical = _center_on_canvas(
        normalized,
    )

    return CanonicalizationResult(
        canonical=canonical,
        mask=mask,
        roi=roi_bgr,
    )
