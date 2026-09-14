
from __future__ import annotations
import io, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v2b.quality import sha256_bytes, dhash, image_quality, validate_pil, exact_duplicate_groups

CFG = json.loads((PROJECT_ROOT / "config/v2b_mapping.json").read_text(encoding="utf-8"))
ART = PROJECT_ROOT / "artifacts/v2b"
ART.mkdir(parents=True, exist_ok=True)
MANIFEST = ART / "v2b_candidate_manifest.csv"

ds = load_dataset(CFG["dataset"], split="train")
fingerprint = getattr(ds, "_fingerprint", "")
rng = np.random.default_rng(CFG["seed"])
pool_n = CFG["target_per_class"] * CFG["candidate_multiplier"]

candidate_indices = {i: [] for i in range(10)}
for idx in range(len(ds)):
    row = ds[idx]
    article = str(row.get("articleType", ""))
    if article in CFG["mapping"]:
        candidate_indices[CFG["mapping"][article]["class_id"]].append(idx)
    elif article in CFG["ankle_boot_candidate_article_types"]:
        candidate_indices[9].append(idx)

records = []
for class_id, indices in candidate_indices.items():
    indices = np.asarray(indices, dtype=int)
    rng.shuffle(indices)
    n = max(pool_n, 300) if class_id == 9 else pool_n
    for idx in indices[:n]:
        row = ds[int(idx)]
        image = row["image"].convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        raw = buf.getvalue()
        ok, err = validate_pil(image)
        q = image_quality(image)
        article = str(row.get("articleType", ""))

        if class_id == 9:
            class_name = "Ankle boot"
            mapping_status = "VISUAL_REVIEW_REQUIRED"
        else:
            class_name = CFG["mapping"][article]["class_name"]
            mapping_status = "REVIEW_REQUIRED" if CFG["mapping"][article]["review"] else "MAPPED"

        records.append({
            "dataset_index": int(idx),
            "source_dataset": CFG["dataset"],
            "dataset_fingerprint": fingerprint,
            "source_id": str(row.get("id", idx)),
            "original_article_type": article,
            "product_display_name": str(row.get("productDisplayName", "")),
            "mapped_class_id": class_id,
            "mapped_class_name": class_name,
            "mapping_status": mapping_status,
            "sha256": sha256_bytes(raw),
            "dhash": dhash(image),
            "file_integrity": "PASS" if ok else "FAIL",
            "integrity_notes": err,
            **q,
            "exact_duplicate_status": "",
            "quality_status": "REVIEW" if q["critical_low_resolution"] or q["low_contrast_global"] else "PASS",
            "semantic_status": "UNREVIEWED",
            "presentation_type": "UNREVIEWED",
            "human_present": "UNREVIEWED",
            "composition_status": "UNREVIEWED",
            "human_review_status": "UNREVIEWED",
            "review_notes": "",
            "final_status": "CANDIDATE",
        })

df = pd.DataFrame(records)
df["exact_duplicate_status"] = exact_duplicate_groups(df)

# Preserve previous human curation when regenerating.
if MANIFEST.exists():
    previous = pd.read_csv(MANIFEST).fillna("")
    keep = ["dataset_index", "semantic_status", "presentation_type", "human_present",
            "composition_status", "human_review_status", "review_notes", "final_status"]
    previous = previous[[c for c in keep if c in previous.columns]]
    df = df.merge(previous, on="dataset_index", how="left", suffixes=("", "_old"))
    for col in keep[1:]:
        old = f"{col}_old"
        if old in df.columns:
            mask = df[old].astype(str).ne("")
            df.loc[mask, col] = df.loc[mask, old]
            df.drop(columns=[old], inplace=True)

df.to_csv(MANIFEST, index=False)
print(df.groupby(["mapped_class_id","mapped_class_name"]).size())
print(f"Manifest: {MANIFEST}")
