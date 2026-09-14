
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from PIL import Image

ALLOWED_PRESENTATION_TYPES_V2B_1 = {"PRODUCT_ONLY", "MANNEQUIN"}

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def dhash(image: Image.Image, hash_size: int = 8) -> str:
    gray = image.convert("L").resize((hash_size + 1, hash_size))
    arr = np.asarray(gray)
    diff = arr[:, 1:] > arr[:, :-1]
    return f"{sum((1 << i) for i, v in enumerate(diff.flatten()) if v):016x}"

def validate_pil(image: Image.Image) -> tuple[bool, str]:
    try:
        image.load()
        return (image.width >= 2 and image.height >= 2), ""
    except Exception as exc:
        return False, f"decode_error:{type(exc).__name__}"

def resolution_status(width: int, height: int) -> str:
    m = min(width, height)
    if m < 64:
        return "CRITICAL_LOW_RESOLUTION"
    if m < 128:
        return "LOW_RESOLUTION_WARNING"
    return "ADEQUATE"

def image_quality(image: Image.Image) -> dict:
    arr = np.asarray(image.convert("L"), dtype=np.float32)
    w, h = image.size
    rs = resolution_status(w, h)
    contrast = float(arr.std())
    return {
        "width": w,
        "height": h,
        "resolution_status": rs,
        "critical_low_resolution": rs == "CRITICAL_LOW_RESOLUTION",
        "low_resolution_warning": rs == "LOW_RESOLUTION_WARNING",
        "low_contrast_global": contrast < 18.0,
        "contrast_std": round(contrast, 3),
    }

def exact_duplicate_groups(df: pd.DataFrame) -> pd.Series:
    counts = df["sha256"].value_counts()
    return df["sha256"].map(lambda x: "EXACT_DUPLICATE" if counts.get(x, 0) > 1 else "UNIQUE")

def presentation_allowed_for_v2b_1(presentation_type: str, human_present: str) -> bool:
    return presentation_type in ALLOWED_PRESENTATION_TYPES_V2B_1 and human_present == "NO"

def final_eligibility(row: pd.Series) -> bool:
    return (
        row.get("file_integrity") == "PASS"
        and row.get("exact_duplicate_status") == "UNIQUE"
        and row.get("semantic_status") == "VALID"
        and row.get("composition_status") == "CLEAN"
        and row.get("human_review_status") == "ACCEPT"
        and presentation_allowed_for_v2b_1(
            str(row.get("presentation_type", "")),
            str(row.get("human_present", "")),
        )
    )
