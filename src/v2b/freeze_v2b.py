
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v2b.quality import final_eligibility

MANIFEST = PROJECT_ROOT / "artifacts/v2b/v2b_candidate_manifest.csv"
CFG = json.loads((PROJECT_ROOT / "config/v2b_mapping.json").read_text(encoding="utf-8"))
OUT = PROJECT_ROOT / "data/v2b/frozen"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST).fillna("")
df["eligible_now"] = df.apply(final_eligibility, axis=1)

target = CFG["target_per_class"]
rng = np.random.default_rng(CFG["seed"])
ds = load_dataset(CFG["dataset"], split="train")

selected = []
for class_id in range(10):
    pool = df[(df.mapped_class_id==class_id) & (df.eligible_now)].copy()
    if len(pool) < target:
        raise RuntimeError(f"Classe {class_id}: {len(pool)} elegíveis; target={target}.")
    chosen = rng.choice(pool.index.to_numpy(), size=target, replace=False)
    selected.append(pool.loc[chosen])

final = pd.concat(selected).sort_values(["mapped_class_id","dataset_index"]).reset_index(drop=True)

for _, rec in final.iterrows():
    cls = f"{int(rec.mapped_class_id)}_{rec.mapped_class_name.replace('/','_').replace(' ','_')}"
    d = OUT / cls
    d.mkdir(parents=True, exist_ok=True)
    ds[int(rec.dataset_index)]["image"].convert("RGB").save(d / f"{rec.source_id}.png")

manifest_bytes = final.drop(columns=["eligible_now"]).to_csv(index=False).encode("utf-8")
digest = hashlib.sha256(manifest_bytes).hexdigest()
(PROJECT_ROOT / "artifacts/v2b/v2b_frozen_manifest.csv").write_bytes(manifest_bytes)
(PROJECT_ROOT / "artifacts/v2b/v2b_freeze.json").write_text(json.dumps({
    "version":"V2B-1.0",
    "domain":"Real External Product Domain",
    "target_per_class":target,
    "total_images":len(final),
    "manifest_sha256":digest,
    "resolution_is_blocking":False,
    "allowed_presentations":["PRODUCT_ONLY","MANNEQUIN"],
    "human_present_required":"NO",
    "worn_by_person":"excluded from main V2B-1.0; retained as context-domain candidate"
}, indent=2), encoding="utf-8")

print(f"V2B frozen: {len(final)} images")
print(f"Manifest SHA-256: {digest}")
