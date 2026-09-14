from __future__ import annotations

import io, sys
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Reuse the same quality utilities already used in V2B.
from src.v2b.quality import (
    sha256_bytes,
    dhash,
    image_quality,
    validate_pil,
    exact_duplicate_groups,
)

# ---------------------------------------------------------------------
# EAS v1 - External Adaptation Set candidate manifest
# ---------------------------------------------------------------------
#
# Purpose:
#   Build a NEW candidate population for domain adaptation.
#
# Critical rules:
#   1) V2B PRODUCT_ONLY v1 remains evaluation-only.
#   2) No source_id, dataset_index or exact image hash from V2B may enter EAS.
#   3) No model prediction is used during candidate selection.
#   4) Candidate generation is deterministic.
#   5) This file is NOT the final training set. Human/data curation comes next.
#
# The candidate pool is intentionally larger than the final adaptation set.
# We DO NOT force 60 PRODUCT_ONLY samples per class at generation time because
# V2B already showed that presentation type is class-dependent; forcing a fixed
# final count before curation could lead to selection engineering.
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

V2B_FROZEN_MANIFEST = (
    PROJECT_ROOT / "artifacts" / "v2b" / "v2b_evaluation_manifest_v1.csv"
)

OUT_DIR = PROJECT_ROOT / "artifacts" / "domain_adaptation" / "eas_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MANIFEST = OUT_DIR / "eas_candidate_manifest.csv"
OUT_REPORT = OUT_DIR / "eas_candidate_manifest_report.json"

DATASET_NAME = "ashraq/fashion-product-images-small"
SPLIT = "train"
SEED = 20260911

# Broad candidate pool. These are candidates for curation, NOT guaranteed
# PRODUCT_ONLY/valid samples.
TARGET_CANDIDATES_PER_CLASS = 200
ANKLE_BOOT_CANDIDATE_TARGET = 300

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

# Mapping kept compatible with the V2B study.
ARTICLE_TYPE_MAPPING = {
    "Tshirts": 0,
    "Trousers": 1,
    "Sweaters": 2,
    "Sweatshirts": 2,
    "Dresses": 3,
    "Jackets": 4,
    "Sandals": 5,
    "Shirts": 6,
    "Sports Shoes": 7,
    "Handbags": 8,
}

# Class 9 remains a visual-review class. Generic footwear must NOT be
# automatically assumed to be an ankle boot.
ANKLE_BOOT_CANDIDATE_ARTICLE_TYPES = {
    "Casual Shoes",
    "Formal Shoes",
    "Heels",
    "Flats",
}


def norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_v2b_exclusion_sets(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen V2B manifest not found: {path}\n"
            "EAS generation is blocked because overlap cannot be validated."
        )

    v2b = pd.read_csv(path, dtype=str).fillna("")

    required = {"dataset_index", "source_id", "sha256"}
    missing = required.difference(v2b.columns)
    if missing:
        raise ValueError(
            f"V2B manifest missing required exclusion fields: {sorted(missing)}"
        )

    return {
        "dataset_index": set(v2b["dataset_index"].astype(str)),
        "source_id": set(v2b["source_id"].astype(str)),
        "sha256": set(v2b["sha256"].astype(str)),
    }


def candidate_class(article_type: str) -> tuple[int | None, str]:
    if article_type in ARTICLE_TYPE_MAPPING:
        class_id = ARTICLE_TYPE_MAPPING[article_type]
        return class_id, "MAPPED"

    if article_type in ANKLE_BOOT_CANDIDATE_ARTICLE_TYPES:
        return 9, "VISUAL_REVIEW_REQUIRED"

    return None, "NOT_MAPPED"


def encode_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def build_manifest() -> pd.DataFrame:
    exclusions = load_v2b_exclusion_sets(V2B_FROZEN_MANIFEST)

    ds = load_dataset(DATASET_NAME, split=SPLIT)
    fingerprint = norm(getattr(ds, "_fingerprint", ""))

    # First collect eligible source indices by mapped class while excluding
    # known V2B identities before sampling.
    pools: dict[int, list[int]] = {i: [] for i in range(10)}

    for idx in range(len(ds)):
        row = ds[idx]

        dataset_index = str(idx)
        source_id = norm(row.get("id", idx))

        # Cheap exclusion before reading/re-encoding image.
        if dataset_index in exclusions["dataset_index"]:
            continue
        if source_id in exclusions["source_id"]:
            continue

        article_type = norm(row.get("articleType", ""))
        class_id, _ = candidate_class(article_type)

        if class_id is not None:
            pools[class_id].append(idx)

    rng = np.random.default_rng(SEED)
    records: list[dict[str, Any]] = []

    for class_id in range(10):
        indices = np.asarray(pools[class_id], dtype=int)
        rng.shuffle(indices)

        target = (
            ANKLE_BOOT_CANDIDATE_TARGET
            if class_id == 9
            else TARGET_CANDIDATES_PER_CLASS
        )

        selected_count = 0

        for idx in indices:
            if selected_count >= target:
                break

            row = ds[int(idx)]
            image = row["image"].convert("RGB")
            raw = encode_png(image)
            image_sha256 = sha256_bytes(raw)

            # Strong third overlap check: exact bytes after deterministic PNG
            # encoding. Any collision with the frozen evaluation set is rejected.
            if image_sha256 in exclusions["sha256"]:
                continue

            ok, err = validate_pil(image)
            quality = image_quality(image)

            article_type = norm(row.get("articleType", ""))
            mapped_class_id, mapping_status = candidate_class(article_type)

            if mapped_class_id != class_id:
                continue

            records.append(
                {
                    "dataset_index": int(idx),
                    "source_dataset": DATASET_NAME,
                    "dataset_fingerprint": fingerprint,
                    "source_id": norm(row.get("id", idx)),
                    "original_article_type": article_type,
                    "product_display_name": norm(
                        row.get("productDisplayName", "")
                    ),
                    "mapped_class_id": class_id,
                    "mapped_class_name": CLASS_NAMES[class_id],
                    "mapping_status": mapping_status,
                    "sha256": image_sha256,
                    "dhash": dhash(image),
                    "file_integrity": "PASS" if ok else "FAIL",
                    "integrity_notes": norm(err),
                    **quality,
                    "exact_duplicate_status": "",
                    "quality_status": (
                        "REVIEW"
                        if quality["critical_low_resolution"]
                        or quality["low_contrast_global"]
                        else "PASS"
                    ),
                    # EAS-specific controls
                    "v2b_dataset_index_overlap": "NO",
                    "v2b_source_id_overlap": "NO",
                    "v2b_sha256_overlap": "NO",
                    "semantic_status": "UNREVIEWED",
                    "presentation_type": "UNREVIEWED",
                    "human_present": "UNREVIEWED",
                    "composition_status": "UNREVIEWED",
                    "human_review_status": "UNREVIEWED",
                    "review_notes": "",
                    "eas_split": "UNASSIGNED",
                    "canonicalizer_version": "v2",
                    "final_status": "CANDIDATE",
                }
            )
            selected_count += 1

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No EAS candidates were generated.")

    df["exact_duplicate_status"] = exact_duplicate_groups(df)

    # Block any accidental duplicate identity inside EAS.
    if df["source_id"].duplicated().any():
        duplicated = df.loc[df["source_id"].duplicated(False), "source_id"].tolist()
        raise RuntimeError(
            f"Duplicate source_id found inside EAS candidate pool: {duplicated[:10]}"
        )

    if df["dataset_index"].duplicated().any():
        raise RuntimeError("Duplicate dataset_index found inside EAS candidate pool.")

    # Final overlap assertions against frozen V2B.
    assert not set(df["source_id"].astype(str)) & exclusions["source_id"]
    assert not set(df["dataset_index"].astype(str)) & exclusions["dataset_index"]
    assert not set(df["sha256"].astype(str)) & exclusions["sha256"]

    return df.sort_values(
        ["mapped_class_id", "dataset_index"],
        kind="stable",
    ).reset_index(drop=True)


def save_report(df: pd.DataFrame) -> None:
    counts = (
        df.groupby(["mapped_class_id", "mapped_class_name"])
        .size()
        .rename("candidate_count")
        .reset_index()
    )

    report = {
        "dataset_name": DATASET_NAME,
        "split": SPLIT,
        "seed": SEED,
        "v2b_exclusion_manifest": str(V2B_FROZEN_MANIFEST),
        "candidate_manifest": str(OUT_MANIFEST),
        "candidate_count": int(len(df)),
        "target_candidates_per_class": TARGET_CANDIDATES_PER_CLASS,
        "ankle_boot_candidate_target": ANKLE_BOOT_CANDIDATE_TARGET,
        "class_distribution": counts.to_dict(orient="records"),
        "overlap_checks": {
            "dataset_index": 0,
            "source_id": 0,
            "sha256": 0,
        },
        "status": "CANDIDATE_POOL_ONLY_NOT_CURATED_NOT_FROZEN",
        "notes": [
            "V2B remains evaluation-only.",
            "No predictions are used during EAS candidate generation.",
            "PRODUCT_ONLY eligibility is determined during human/data curation.",
            "A fixed final count per class is not forced before curation.",
        ],
    }

    OUT_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    df = build_manifest()
    df.to_csv(OUT_MANIFEST, index=False)
    save_report(df)

    print("\nEAS v1 candidate manifest generated")
    print("=" * 60)
    print(
        df.groupby(["mapped_class_id", "mapped_class_name"])
        .size()
        .rename("n")
        .to_string()
    )
    print("-" * 60)
    print(f"Total candidates : {len(df)}")
    print("V2B overlaps     : 0 (dataset_index/source_id/sha256)")
    print(f"Manifest         : {OUT_MANIFEST}")
    print(f"Report           : {OUT_REPORT}")
    print("Status           : CANDIDATE / NOT CURATED / NOT FROZEN")


if __name__ == "__main__":
    main()
