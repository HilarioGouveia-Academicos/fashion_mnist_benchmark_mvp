from pathlib import Path
import pandas as pd

REQUIRED_CLASSES = {str(i) for i in range(10)}

def validate_master_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"source_id","dataset_index","mapped_class_id","mapped_class_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest sem colunas obrigatórias: {sorted(missing)}")
    if df["source_id"].duplicated().any():
        raise ValueError("Manifest contém source_id duplicado.")
    bad_idx = pd.to_numeric(df["dataset_index"], errors="coerce").isna()
    if bad_idx.any():
        raise ValueError(f"Manifest contém {int(bad_idx.sum())} dataset_index inválido(s).")
    classes = set(df["mapped_class_id"].astype(str))
    missing_classes = REQUIRED_CLASSES - classes
    if missing_classes:
        raise ValueError(f"Manifest incompleto; classes ausentes: {sorted(missing_classes)}")
    return df
