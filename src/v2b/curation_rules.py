
from __future__ import annotations
import pandas as pd

MAIN_PRESENTATIONS = {"PRODUCT_ONLY", "MANNEQUIN"}

def classify_final_status(row: pd.Series) -> str:
    decision = str(row.get("human_review_status", ""))
    if decision == "INVALID_SAMPLE":
        return "EXCLUDED"
    if decision == "NEEDS_REVIEW":
        return "REVIEW"

    valid = (
        decision == "VALID_SAMPLE"
        and str(row.get("semantic_status", "")) == "VALID"
        and str(row.get("composition_status", "")) == "CLEAN"
        and str(row.get("file_integrity", "")) == "PASS"
        and str(row.get("exact_duplicate_status", "")) == "UNIQUE"
    )
    if not valid:
        return "CANDIDATE"

    presentation = str(row.get("presentation_type", ""))
    human = str(row.get("human_present", ""))
    if presentation in MAIN_PRESENTATIONS and human == "NO":
        return "ELIGIBLE"
    if presentation == "WORN_BY_PERSON" and human == "YES":
        return "CONTEXT_DOMAIN"
    return "VALID_OUTSIDE_MAIN_DOMAIN"
