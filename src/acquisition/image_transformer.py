from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.domain_shift.canonicalizer import canonicalize_image
from src.preprocessing.canonicalizer_v3_candidate import (
    VERSION as V3_VERSION,
    canonicalize_image_v3,
)


V2_VERSION = "v2"


@dataclass(frozen=True)
class ImageRepresentation:
    key: str
    label: str
    image: Image.Image | None
    png_bytes: bytes
    metadata: dict[str, Any] = field(default_factory=dict)
    mask: Image.Image | None = None
    roi: Image.Image | None = None
    error: str | None = None


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_original(image: Image.Image) -> Image.Image:
    """Normalize EXIF orientation while preserving an RGB original preview."""
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def _gray_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.uint8)


def _mask_diagnostics(mask: np.ndarray) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=np.uint8)
    height, width = binary.shape[:2]
    foreground = binary > 0
    foreground_pixels = int(foreground.sum())
    total_pixels = int(height * width)

    diagnostics: dict[str, Any] = {
        "foreground_occupancy": (
            foreground_pixels / total_pixels if total_pixels else 0.0
        ),
        "background_ratio": (
            1.0 - (foreground_pixels / total_pixels) if total_pixels else 1.0
        ),
        "grabcut_status": "SUCCESS" if foreground_pixels else "EMPTY_MASK",
    }

    coordinates = cv2.findNonZero((foreground.astype(np.uint8) * 255))
    if coordinates is not None:
        x, y, w, h = cv2.boundingRect(coordinates)
        diagnostics.update(
            {
                "bounding_box": [int(x), int(y), int(w), int(h)],
                "object_center_normalized": [
                    float((x + (w / 2.0)) / max(width, 1)),
                    float((y + (h / 2.0)) / max(height, 1)),
                ],
                "width_height_ratio": float(w / max(h, 1)),
            }
        )

    return diagnostics


def build_representations(image: Image.Image) -> dict[str, ImageRepresentation]:
    """Build all CV Capture Lab representations from one immutable capture."""
    original = normalize_original(image)
    original_bytes = _png_bytes(original)
    captured_at = datetime.now(timezone.utc).isoformat()
    original_size = list(original.size)

    common = {
        "source": "cv_capture_lab",
        "acquisition": "camera",
        "captured_at": captured_at,
        "original_size": original_size,
        "original_sha256": _sha256(original_bytes),
    }

    representations: dict[str, ImageRepresentation] = {
        "original": ImageRepresentation(
            key="original",
            label="Original Capture",
            image=original,
            png_bytes=original_bytes,
            metadata={
                **common,
                "representation": "original",
                "canonicalization_applied": False,
                "canonicalization_version": None,
                "output_size": original_size,
                "output_sha256": _sha256(original_bytes),
            },
        )
    }

    gray = _gray_array(original)

    raw_array = np.asarray(
        Image.fromarray(gray, mode="L").resize((28, 28)),
        dtype=np.uint8,
    )
    raw_image = Image.fromarray(raw_array, mode="L")
    raw_bytes = _png_bytes(raw_image)
    representations["raw_28x28"] = ImageRepresentation(
        key="raw_28x28",
        label="RAW 28×28",
        image=raw_image,
        png_bytes=raw_bytes,
        metadata={
            **common,
            "representation": "raw_28x28",
            "canonicalization_applied": False,
            "canonicalization_version": None,
            "output_size": [28, 28],
            "output_sha256": _sha256(raw_bytes),
        },
    )

    try:
        v2_array = canonicalize_image(gray)
        v2_image = Image.fromarray(v2_array.astype(np.uint8), mode="L")
        v2_bytes = _png_bytes(v2_image)
        representations["canonicalized_v2"] = ImageRepresentation(
            key="canonicalized_v2",
            label="Canonicalized V2",
            image=v2_image,
            png_bytes=v2_bytes,
            metadata={
                **common,
                "representation": "canonicalized_v2",
                "canonicalization_applied": True,
                "canonicalization_version": V2_VERSION,
                "output_size": [28, 28],
                "output_sha256": _sha256(v2_bytes),
                "status": "SUCCESS",
            },
        )
    except Exception as exc:
        representations["canonicalized_v2"] = ImageRepresentation(
            key="canonicalized_v2",
            label="Canonicalized V2",
            image=None,
            png_bytes=b"",
            metadata={
                **common,
                "representation": "canonicalized_v2",
                "canonicalization_applied": False,
                "canonicalization_version": V2_VERSION,
                "status": "FAILED",
            },
            error=str(exc),
        )

    try:
        v3_result = canonicalize_image_v3(gray)
        v3_image = Image.fromarray(v3_result.canonical.astype(np.uint8), mode="L")
        v3_bytes = _png_bytes(v3_image)
        mask_image = Image.fromarray(v3_result.mask.astype(np.uint8), mode="L")

        roi_array = np.asarray(v3_result.roi)
        if roi_array.ndim == 3:
            roi_rgb = cv2.cvtColor(roi_array, cv2.COLOR_BGR2RGB)
            roi_image = Image.fromarray(roi_rgb, mode="RGB")
        else:
            roi_image = Image.fromarray(roi_array.astype(np.uint8), mode="L")

        diagnostics = _mask_diagnostics(v3_result.mask)
        representations["canonicalized_v3"] = ImageRepresentation(
            key="canonicalized_v3",
            label="Canonicalized V3 (GrabCut)",
            image=v3_image,
            png_bytes=v3_bytes,
            mask=mask_image,
            roi=roi_image,
            metadata={
                **common,
                "representation": "canonicalized_v3",
                "canonicalization_applied": True,
                "canonicalization_version": V3_VERSION,
                "cv_method": "grabcut",
                "output_size": [28, 28],
                "output_sha256": _sha256(v3_bytes),
                "status": "SUCCESS",
                "diagnostics": diagnostics,
            },
        )
    except Exception as exc:
        representations["canonicalized_v3"] = ImageRepresentation(
            key="canonicalized_v3",
            label="Canonicalized V3 (GrabCut)",
            image=None,
            png_bytes=b"",
            metadata={
                **common,
                "representation": "canonicalized_v3",
                "canonicalization_applied": False,
                "canonicalization_version": V3_VERSION,
                "cv_method": "grabcut",
                "status": "FAILED",
            },
            error=str(exc),
        )

    return representations
