from __future__ import annotations

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


import hashlib
import io
import importlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st

from src.domain_shift.canonicalizer import canonicalize_image


DEFAULT_INPUT_DIR = "data/external"
DEFAULT_OUTPUT_DIR = "artifacts/canonicalization_visual_qa"
CANONICALIZER_VERSION = "v2"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

STATUS_OPTIONS = ["UNREVIEWED", "PASS", "PASS_WITH_LIMITATIONS", "FAIL"]
ISSUE_OPTIONS = [
    "foreground_cut", "foreground_missing_parts", "background_residue",
    "object_too_small", "object_too_large", "off_center",
    "aspect_ratio_distortion", "polarity_problem", "contrast_problem",
    "mask_noise", "thin_details_lost", "other",
]


def load_project_canonicalizer():
    """Ajuste esta função se o módulo/função do Canonicalizer V2 for diferente."""
    candidates = [
        ("src.preprocessing.canonicalizer", "canonicalize_image"),
        ("src.preprocessing.canonicalization", "canonicalize_image"),
        ("src.preprocessing.image_canonicalizer", "canonicalize_image"),
        ("src.domain_shift.canonicalizer", "canonicalize_image"),
    ]
    errors = []
    for module_name, function_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, function_name), f"{module_name}.{function_name}"
        except Exception as exc:
            errors.append(f"{module_name}.{function_name}: {exc}")
    return None, errors


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_gray(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("L")


def ensure_pil_gray(obj) -> Image.Image:
    if isinstance(obj, Image.Image):
        return obj.convert("L")
    arr = np.asarray(obj)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] == 1 else np.mean(arr[..., :3], axis=-1)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def run_canonicalizer(fn, original: Image.Image) -> Tuple[Image.Image, Image.Image]:
    errors = []
    for input_obj in (original, np.asarray(original)):
        try:
            result = fn(input_obj)
            if isinstance(result, dict):
                canonical = result.get("canonicalized") or result.get("image") or result.get("output")
                mask = result.get("mask")
            elif isinstance(result, tuple):
                canonical = result[0]
                mask = result[1] if len(result) > 1 else None
            else:
                canonical, mask = result, None
            canonical_img = ensure_pil_gray(canonical)
            if mask is None:
                mask_arr = (np.asarray(canonical_img) > 0).astype(np.uint8) * 255
                mask_img = Image.fromarray(mask_arr, mode="L")
            else:
                mask_img = ensure_pil_gray(mask)
            return canonical_img, mask_img
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("Canonicalizer falhou: " + " | ".join(errors))


def class_from_path(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
        if len(rel.parts) > 1:
            return rel.parts[-2]
    except Exception:
        pass
    return path.parent.name


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: str(p).lower(),
    )


def load_reviews(csv_path: Path) -> Dict[str, dict]:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path).fillna("")
    if "image_sha256" not in df.columns:
        return {}
    return {str(row["image_sha256"]): row.to_dict() for _, row in df.iterrows()}


def save_reviews(csv_path: Path, reviews: Dict[str, dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if reviews:
        pd.DataFrame(list(reviews.values())).to_csv(csv_path, index=False)


def make_triptych(original, mask, canonical, title, status, notes):
    size = 320
    header = 70
    footer = 70
    canvas = Image.new("RGB", (size * 3, header + size + footer), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill="black")
    draw.text((10, 34), f"Status: {status}", fill="black")
    for i, (label, img, nearest) in enumerate([
        ("Original", original, False),
        ("Mask / Foreground", mask, True),
        ("Canonicalized", canonical, True),
    ]):
        tmp = Image.new("L", (size, size), 30)
        src = img.copy().convert("L")
        resample = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
        src.thumbnail((size, size), resample=resample)
        tmp.paste(src, ((size - src.width)//2, (size - src.height)//2))
        canvas.paste(tmp.convert("RGB"), (i*size, header))
        draw.text((i*size + 8, header + 8), label, fill="white")
    draw.text((10, header + size + 10), f"Notes: {(notes or '')[:180]}", fill="black")
    return canvas


def build_export_zip(reviews: Dict[str, dict], rendered: Dict[str, dict]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        df = pd.DataFrame(list(reviews.values()))
        z.writestr("canonicalization_visual_qa.csv", df.to_csv(index=False).encode("utf-8"))
        z.writestr("canonicalization_visual_qa.json", json.dumps({
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "canonicalizer_version": CANONICALIZER_VERSION,
            "reviews": list(reviews.values()),
        }, ensure_ascii=False, indent=2).encode("utf-8"))
        for h, payload in rendered.items():
            if h not in reviews:
                continue
            r = reviews[h]
            triptych = make_triptych(
                payload["original"], payload["mask"], payload["canonical"],
                r.get("filename", h), r.get("review_status", "UNREVIEWED"), r.get("review_notes", "")
            )
            img_buf = io.BytesIO()
            triptych.save(img_buf, format="PNG")
            z.writestr(f"triptychs/{Path(r.get('filename', h)).stem}_{h[:8]}.png", img_buf.getvalue())
    buf.seek(0)
    return buf.getvalue()


st.set_page_config(page_title="Canonicalization Visual QA", page_icon="🔎", layout="wide")
st.title("🔎 Canonicalization Visual QA")
st.caption("Original × Mask/Foreground × Canonicalized × Human Review")

with st.sidebar:
    st.header("Configuração")
    input_dir = Path(st.text_input("Diretório das imagens", DEFAULT_INPUT_DIR))
    output_dir = Path(st.text_input("Diretório dos artefatos", DEFAULT_OUTPUT_DIR))
    page_size = st.selectbox("Imagens por página", [5, 10, 20], index=1)

review_csv = output_dir / "canonicalization_visual_qa.csv"
fn, canonicalizer_info = load_project_canonicalizer()
if fn is None:
    st.error("Não encontrei automaticamente o Canonicalizer V2. Ajuste `load_project_canonicalizer()`.")
    with st.expander("Tentativas de import"):
        for e in canonicalizer_info:
            st.code(e)
    st.stop()

st.success(f"Canonicalizer carregado: `{canonicalizer_info}`")
all_images = list_images(input_dir)
if not all_images:
    st.warning(f"Nenhuma imagem encontrada em `{input_dir}`")
    st.stop()

if "qa_reviews" not in st.session_state:
    st.session_state.qa_reviews = load_reviews(review_csv)
reviews = st.session_state.qa_reviews

classes = sorted({class_from_path(p, input_dir) for p in all_images})
selected_classes = st.multiselect("Filtrar classes", classes, default=classes)
review_filter = st.selectbox("Filtrar por avaliação", ["ALL"] + STATUS_OPTIONS)

def status_for(path: Path):
    return reviews.get(sha256_file(path), {}).get("review_status", "UNREVIEWED")

filtered = [p for p in all_images if class_from_path(p, input_dir) in selected_classes]
if review_filter != "ALL":
    filtered = [p for p in filtered if status_for(p) == review_filter]

max_page = max(1, (len(filtered) + page_size - 1) // page_size)
page = st.number_input("Página", 1, max_page, 1, 1)
page_images = filtered[(page-1)*page_size : page*page_size]

reviewed_count = sum(status_for(p) != "UNREVIEWED" for p in all_images)
a,b,c,d = st.columns(4)
a.metric("Imagens", len(all_images))
b.metric("Filtradas", len(filtered))
c.metric("Avaliadas", reviewed_count)
d.metric("Pendentes", len(all_images)-reviewed_count)
st.divider()

rendered = {}
for path in page_images:
    h = sha256_file(path)
    original = load_gray(path)
    try:
        canonical, mask = run_canonicalizer(fn, original)
        err = ""
    except Exception as exc:
        canonical = Image.new("L", (28,28), 0)
        mask = Image.new("L", original.size, 0)
        err = str(exc)
    rendered[h] = {"original": original, "mask": mask, "canonical": canonical}
    existing = reviews.get(h, {})

    st.subheader(path.name)
    st.caption(f"Classe/pasta: `{class_from_path(path, input_dir)}`")
    c1,c2,c3,c4 = st.columns([1,1,1,1.15])
    with c1:
        st.markdown("**Original**")
        st.image(original, use_container_width=True)
        st.caption(f"{original.width}×{original.height}")
    with c2:
        st.markdown("**Mask / Foreground**")
        st.image(mask, use_container_width=True)
        st.caption(f"{mask.width}×{mask.height}")
    with c3:
        st.markdown("**Canonicalized**")
        st.image(canonical.resize((280,280), Image.Resampling.NEAREST), use_container_width=True)
        st.caption(f"{canonical.width}×{canonical.height}")
    with c4:
        st.markdown("**Avaliação**")
        if err:
            st.error(err)
        sk, ik, nk = f"status_{h}", f"issues_{h}", f"notes_{h}"
        if sk not in st.session_state:
            st.session_state[sk] = existing.get("review_status", "UNREVIEWED")
        if ik not in st.session_state:
            st.session_state[ik] = [x for x in str(existing.get("issue_tags", "")).split("|") if x]
        if nk not in st.session_state:
            st.session_state[nk] = existing.get("review_notes", "")
        status = st.selectbox("Status", STATUS_OPTIONS, key=sk)
        issues = st.multiselect("Problemas observados", ISSUE_OPTIONS, key=ik)
        notes = st.text_area("Notas", key=nk, height=100)
        if st.button("Salvar avaliação", key=f"save_{h}", use_container_width=True):
            reviews[h] = {
                "image_sha256": h,
                "filename": path.name,
                "relative_path": str(path.relative_to(input_dir)),
                "class_name": class_from_path(path, input_dir),
                "canonicalizer_version": CANONICALIZER_VERSION,
                "canonicalizer_callable": canonicalizer_info,
                "review_status": status,
                "issue_tags": "|".join(issues),
                "review_notes": notes.strip(),
                "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            st.session_state.qa_reviews = reviews
            save_reviews(review_csv, reviews)
            st.success("Salvo")
    st.divider()

st.header("Resumo e exportação")
if reviews:
    df = pd.DataFrame(list(reviews.values()))
    summary = df["review_status"].value_counts(dropna=False).rename_axis("status").reset_index(name="count")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Exportar CSV", df.to_csv(index=False).encode("utf-8"), "canonicalization_visual_qa.csv", "text/csv", use_container_width=True)
    st.download_button("⬇️ Exportar JSON", json.dumps(list(reviews.values()), ensure_ascii=False, indent=2).encode("utf-8"), "canonicalization_visual_qa.json", "application/json", use_container_width=True)
    zip_bytes = build_export_zip(reviews, rendered)
    st.download_button("📦 Exportar pacote para análise", zip_bytes, "canonicalization_visual_qa_export.zip", "application/zip", use_container_width=True)
    st.info("Para eu também analisar visualmente depois, envie o ZIP exportado. Ele inclui CSV/JSON e triptychs das imagens renderizadas nesta sessão/página.")
else:
    st.info("Ainda não há avaliações salvas.")
