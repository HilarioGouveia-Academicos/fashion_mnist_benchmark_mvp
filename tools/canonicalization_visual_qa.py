
from __future__ import annotations

import hashlib
import importlib
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIG
# ============================================================

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "qa_low_contrast"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "canonicalization_focused_qa"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

SAMPLE_ROLE_OPTIONS = [
    "LOW_CONTRAST_TARGET",
    "HIGH_CONTRAST_CONTROL",
    "EXCLUDE_COMPOSITION_ARTIFACT",
    "UNDECIDED",
]

STATUS_OPTIONS = [
    "UNREVIEWED",
    "PASS",
    "PASS_WITH_LIMITATIONS",
    "FAIL",
]

WINNER_OPTIONS = [
    "UNDECIDED",
    "V2",
    "V3",
    "TIE",
    "BOTH_FAIL",
]

ISSUE_OPTIONS = [
    "under_segmentation",
    "over_segmentation",
    "foreground_cut",
    "foreground_missing_parts",
    "background_residue",
    "semantic_distortion",
    "thin_details_lost",
    "off_center",
    "aspect_ratio_problem",
    "polarity_problem",
    "contrast_problem",
    "composition_artifact",
    "other",
]


# ============================================================
# CANONICALIZER ADAPTERS
# ============================================================

def load_v2_module():
    candidates = [
        "src.preprocessing.canonicalizer",
        "src.preprocessing.canonicalization",
        "src.domain_shift.canonicalizer",
    ]

    errors = []
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "canonicalize_image"):
                return module, module_name
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")

    raise ImportError(
        "Não foi possível localizar o Canonicalizer V2.\n"
        + "\n".join(errors)
    )


def load_v3_module():
    candidates = [
        "src.preprocessing.canonicalizer_v3_candidate",
        "src.preprocessing.canonicalizer_v3_candidate_refactored",
    ]

    errors = []
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "canonicalize_image_v3"):
                return module, module_name
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")

    raise ImportError(
        "Não foi possível localizar o Canonicalizer V3 candidate.\n"
        + "\n".join(errors)
    )


V2_MODULE, V2_MODULE_NAME = load_v2_module()
V3_MODULE, V3_MODULE_NAME = load_v3_module()


def build_v2_mask(gray: np.ndarray) -> np.ndarray:
    """
    Reconstrói a máscara do V2 usando as próprias funções internas do módulo.

    Suporta as duas variantes que usamos no projeto:
    - estimate_background -> build_foreground_mask
    - estimate_background + background_std -> calculate_tolerance
      -> build_foreground_mask -> refine_foreground_mask
    """
    estimate = getattr(V2_MODULE, "_estimate_background", None)
    build = getattr(V2_MODULE, "_build_foreground_mask", None)
    refine = getattr(V2_MODULE, "_refine_foreground_mask", None)
    tolerance_fn = getattr(V2_MODULE, "_calculate_background_tolerance", None)

    if estimate is None or build is None:
        raise RuntimeError(
            "O V2 não expõe _estimate_background/_build_foreground_mask. "
            "A comparação visual da máscara V2 exige essas funções."
        )

    estimated = estimate(gray)

    if isinstance(estimated, tuple):
        background_value = estimated[0]
        background_std = estimated[1] if len(estimated) > 1 else None
    else:
        background_value = estimated
        background_std = None

    if tolerance_fn is not None and background_std is not None:
        tolerance = tolerance_fn(background_std)
        mask = build(gray, background_value, tolerance)
    else:
        # Usa o default do próprio V2 quando o terceiro argumento não é exigido.
        try:
            mask = build(gray, background_value)
        except TypeError:
            mask = build(gray, background_value, 25)

    if refine is not None:
        mask = refine(mask)

    return np.asarray(mask, dtype=np.uint8)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        [
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: str(p).lower(),
    )


def class_from_path(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
        return rel.parts[-2] if len(rel.parts) > 1 else path.parent.name
    except Exception:
        return path.parent.name


@st.cache_data(show_spinner=False)
def process_image(path_str: str):
    path = Path(path_str)

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if bgr is None or gray is None:
        raise ValueError(f"Não foi possível abrir {path}")

    original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # V2
    v2_mask = build_v2_mask(gray)
    v2_canonical = V2_MODULE.canonicalize_image(gray)

    # V3
    v3_result = V3_MODULE.canonicalize_image_v3(bgr)
    v3_mask = np.asarray(v3_result.mask, dtype=np.uint8)
    v3_canonical = np.asarray(v3_result.canonical, dtype=np.uint8)

    return (
        original_rgb,
        v2_mask,
        v2_canonical,
        v3_mask,
        v3_canonical,
    )


def load_reviews(path: Path) -> dict:
    if not path.exists():
        return {}

    df = pd.read_csv(path).fillna("")
    if "image_sha256" not in df.columns:
        return {}

    return {
        str(row["image_sha256"]): row.to_dict()
        for _, row in df.iterrows()
    }


def save_reviews(path: Path, reviews: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if reviews:
        pd.DataFrame(list(reviews.values())).to_csv(path, index=False)


def image_png_bytes(array: np.ndarray) -> bytes:
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_export_zip(reviews: dict, payloads: dict) -> bytes:
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        df = pd.DataFrame(list(reviews.values()))

        z.writestr(
            "canonicalization_focused_qa.csv",
            df.to_csv(index=False).encode("utf-8"),
        )

        z.writestr(
            "canonicalization_focused_qa.json",
            json.dumps(
                {
                    "exported_at_utc": datetime.now(timezone.utc).isoformat(),
                    "v2_module": V2_MODULE_NAME,
                    "v3_module": V3_MODULE_NAME,
                    "reviews": list(reviews.values()),
                },
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
        )

        for image_hash, payload in payloads.items():
            if image_hash not in reviews:
                continue

            name = Path(reviews[image_hash]["filename"]).stem

            z.writestr(
                f"visuals/{name}_original.png",
                image_png_bytes(payload["original"]),
            )
            z.writestr(
                f"visuals/{name}_mask_v2.png",
                image_png_bytes(payload["v2_mask"]),
            )
            z.writestr(
                f"visuals/{name}_canonical_v2.png",
                image_png_bytes(payload["v2_canonical"]),
            )
            z.writestr(
                f"visuals/{name}_mask_v3.png",
                image_png_bytes(payload["v3_mask"]),
            )
            z.writestr(
                f"visuals/{name}_canonical_v3.png",
                image_png_bytes(payload["v3_canonical"]),
            )

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# UI
# ============================================================

st.set_page_config(
    page_title="Focused Canonicalization QA",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Focused Canonicalization QA — V2 × V3")
st.caption(
    "Teste direcionado ao fenômeno de baixo contraste foreground/background, "
    "com controles e exclusão explícita de imagens com artefatos de composição."
)

st.info(
    "Objetivo: testar o método de extração/canonicalização, não o classificador. "
    "Use LOW_CONTRAST_TARGET para casos como camiseta/fundo de tonalidade semelhante; "
    "HIGH_CONTRAST_CONTROL para verificar regressões; e EXCLUDE_COMPOSITION_ARTIFACT "
    "para imagens cuja composição já esteja enviesada."
)

with st.sidebar:
    input_dir = Path(
        st.text_input(
            "Diretório das imagens",
            str(DEFAULT_INPUT_DIR),
        )
    )

    output_dir = Path(
        st.text_input(
            "Diretório dos artefatos",
            str(DEFAULT_OUTPUT_DIR),
        )
    )

    page_size = st.selectbox(
        "Imagens por página",
        [3, 5, 10],
        index=1,
    )


images = list_images(input_dir)
if not images:
    st.warning(
        f"Nenhuma imagem encontrada em `{input_dir}`.\n\n"
        "Crie uma pequena base focal (aprox. 12–20 imagens), em vez de reutilizar "
        "automaticamente as 40 imagens sintéticas."
    )
    st.stop()


review_csv = output_dir / "canonicalization_focused_qa.csv"
reviews = load_reviews(review_csv)

if "focused_reviews" not in st.session_state:
    st.session_state.focused_reviews = reviews

reviews = st.session_state.focused_reviews


classes = sorted({class_from_path(p, input_dir) for p in images})
selected_classes = st.multiselect(
    "Classes",
    options=classes,
    default=classes,
)

filtered = [
    p for p in images
    if class_from_path(p, input_dir) in selected_classes
]


total = len(filtered)
max_page = max(1, (total + page_size - 1) // page_size)

page = st.number_input(
    "Página",
    min_value=1,
    max_value=max_page,
    value=1,
    step=1,
)

start = (page - 1) * page_size
end = min(start + page_size, total)
page_images = filtered[start:end]


reviewed = sum(
    1
    for p in images
    if reviews.get(sha256_file(p), {}).get("v3_status", "UNREVIEWED")
    != "UNREVIEWED"
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Amostras", len(images))
m2.metric("Revisadas", reviewed)
m3.metric("Pendentes", len(images) - reviewed)
m4.metric("Comparação", "V2 × V3")


payloads = {}

for image_path in page_images:
    image_hash = sha256_file(image_path)
    class_name = class_from_path(image_path, input_dir)
    existing = reviews.get(image_hash, {})

    try:
        (
            original,
            v2_mask,
            v2_canonical,
            v3_mask,
            v3_canonical,
        ) = process_image(str(image_path))
    except Exception as exc:
        st.error(f"{image_path.name}: {exc}")
        continue

    payloads[image_hash] = {
        "original": original,
        "v2_mask": v2_mask,
        "v2_canonical": v2_canonical,
        "v3_mask": v3_mask,
        "v3_canonical": v3_canonical,
    }

    st.divider()
    st.subheader(image_path.name)

    # 5 visual columns
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown("**Original**")
        st.image(original, use_container_width=True)
        st.caption(f"Classe/pasta: `{class_name}`")

    with c2:
        st.markdown("**Mask V2**")
        st.image(v2_mask, use_container_width=True)
        st.caption(f"coverage: {(v2_mask > 0).mean():.1%}")

    with c3:
        st.markdown("**Canonical V2**")
        st.image(
            Image.fromarray(v2_canonical).resize(
                (280, 280),
                Image.Resampling.NEAREST,
            ),
            use_container_width=True,
        )

    with c4:
        st.markdown("**Mask V3**")
        st.image(v3_mask, use_container_width=True)
        st.caption(f"coverage: {(v3_mask > 0).mean():.1%}")

    with c5:
        st.markdown("**Canonical V3**")
        st.image(
            Image.fromarray(v3_canonical).resize(
                (280, 280),
                Image.Resampling.NEAREST,
            ),
            use_container_width=True,
        )

    st.markdown("#### Avaliação metodológica")

    a1, a2, a3, a4 = st.columns([1.1, 1, 1, 1.4])

    role_key = f"role_{image_hash}"
    v2_key = f"v2_{image_hash}"
    v3_key = f"v3_{image_hash}"
    winner_key = f"winner_{image_hash}"
    issues_key = f"issues_{image_hash}"
    notes_key = f"notes_{image_hash}"

    if role_key not in st.session_state:
        st.session_state[role_key] = existing.get(
            "sample_role",
            "UNDECIDED",
        )

    if v2_key not in st.session_state:
        st.session_state[v2_key] = existing.get(
            "v2_status",
            "UNREVIEWED",
        )

    if v3_key not in st.session_state:
        st.session_state[v3_key] = existing.get(
            "v3_status",
            "UNREVIEWED",
        )

    if winner_key not in st.session_state:
        st.session_state[winner_key] = existing.get(
            "winner",
            "UNDECIDED",
        )

    if issues_key not in st.session_state:
        st.session_state[issues_key] = [
            x
            for x in str(existing.get("issue_tags", "")).split("|")
            if x
        ]

    if notes_key not in st.session_state:
        st.session_state[notes_key] = existing.get(
            "review_notes",
            "",
        )

    with a1:
        sample_role = st.selectbox(
            "Papel da amostra",
            SAMPLE_ROLE_OPTIONS,
            key=role_key,
        )

    with a2:
        v2_status = st.selectbox(
            "V2",
            STATUS_OPTIONS,
            key=v2_key,
        )

    with a3:
        v3_status = st.selectbox(
            "V3",
            STATUS_OPTIONS,
            key=v3_key,
        )

    with a4:
        winner = st.selectbox(
            "Resultado comparativo",
            WINNER_OPTIONS,
            key=winner_key,
        )

    issues = st.multiselect(
        "Problemas observados",
        ISSUE_OPTIONS,
        key=issues_key,
    )

    notes = st.text_area(
        "Notas",
        key=notes_key,
        height=85,
        placeholder=(
            "Ex.: V2 perde o corpo da camiseta; V3 preserva a silhueta. "
            "Ou: imagem possui fragmento lateral e deve ser excluída do QA focal."
        ),
    )

    if st.button(
        "Salvar avaliação",
        key=f"save_{image_hash}",
        use_container_width=True,
    ):
        record = {
            "image_sha256": image_hash,
            "filename": image_path.name,
            "relative_path": str(image_path.relative_to(input_dir)),
            "class_name": class_name,
            "sample_role": sample_role,
            "v2_status": v2_status,
            "v3_status": v3_status,
            "winner": winner,
            "issue_tags": "|".join(issues),
            "review_notes": notes.strip(),
            "v2_mask_coverage": float((v2_mask > 0).mean()),
            "v3_mask_coverage": float((v3_mask > 0).mean()),
            "v2_module": V2_MODULE_NAME,
            "v3_module": V3_MODULE_NAME,
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        reviews[image_hash] = record
        st.session_state.focused_reviews = reviews
        save_reviews(review_csv, reviews)
        st.success("Avaliação salva.")


st.divider()
st.header("Resumo")

if reviews:
    df = pd.DataFrame(list(reviews.values()))

    s1, s2 = st.columns(2)

    with s1:
        st.markdown("**Papéis das amostras**")
        st.dataframe(
            df["sample_role"]
            .value_counts()
            .rename_axis("sample_role")
            .reset_index(name="count"),
            hide_index=True,
            use_container_width=True,
        )

    with s2:
        st.markdown("**Resultado V2 × V3**")
        st.dataframe(
            df["winner"]
            .value_counts()
            .rename_axis("winner")
            .reset_index(name="count"),
            hide_index=True,
            use_container_width=True,
        )

    # Resultado focal: ignora composição enviesada.
    focal = df[
        df["sample_role"].isin(
            ["LOW_CONTRAST_TARGET", "HIGH_CONTRAST_CONTROL"]
        )
    ].copy()

    if not focal.empty:
        st.markdown("**QA focal válido (sem artefatos de composição)**")

        v2_pass = focal["v2_status"].isin(
            ["PASS", "PASS_WITH_LIMITATIONS"]
        ).mean()

        v3_pass = focal["v3_status"].isin(
            ["PASS", "PASS_WITH_LIMITATIONS"]
        ).mean()

        q1, q2, q3 = st.columns(3)
        q1.metric("V2 pass rate", f"{v2_pass:.1%}")
        q2.metric("V3 pass rate", f"{v3_pass:.1%}")
        q3.metric("N focal", len(focal))

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Exportar CSV",
        df.to_csv(index=False).encode("utf-8"),
        "canonicalization_focused_qa.csv",
        "text/csv",
        use_container_width=True,
    )

    export_zip = build_export_zip(
        reviews,
        payloads,
    )

    st.download_button(
        "📦 Exportar pacote para análise",
        export_zip,
        "canonicalization_focused_qa_export.zip",
        "application/zip",
        use_container_width=True,
    )

else:
    st.info("Ainda não há avaliações salvas.")
