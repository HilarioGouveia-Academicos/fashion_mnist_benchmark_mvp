
from __future__ import annotations
from pathlib import Path
import pandas as pd

EXPECTED_CLASSES = {str(i) for i in range(10)}

def validate_master(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"source_id","dataset_index","mapped_class_id","mapped_class_name",
                "file_integrity","exact_duplicate_status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Master sem colunas obrigatórias: {sorted(missing)}")
    if df["source_id"].eq("").any():
        raise ValueError("Master contém source_id vazio.")
    if df["source_id"].duplicated().any():
        raise ValueError(f"Master contém {int(df['source_id'].duplicated().sum())} source_id duplicado(s).")
    bad_idx = pd.to_numeric(df["dataset_index"], errors="coerce").isna()
    if bad_idx.any():
        raise ValueError(f"Master contém {int(bad_idx.sum())} dataset_index inválido(s).")
    classes = set(df["mapped_class_id"])
    missing_classes = EXPECTED_CLASSES - classes
    if missing_classes:
        raise ValueError(f"Master incompleto; classes ausentes: {sorted(missing_classes)}")
    return df

def validate_state(state: pd.DataFrame, master: pd.DataFrame) -> None:
    if state["source_id"].eq("").any():
        raise ValueError("Curation state contém source_id vazio.")
    if state["source_id"].duplicated().any():
        raise ValueError("Curation state contém source_id duplicado.")
    unknown = set(state["source_id"]) - set(master["source_id"])
    if unknown:
        raise ValueError(f"Curation state contém {len(unknown)} source_id fora do master.")
