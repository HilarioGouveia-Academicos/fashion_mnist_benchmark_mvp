from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from datasets import load_dataset

# ---------------------------------------------------------------------
# EAS v1 — External Adaptation Set Curator
#
# Architectural rule:
#
#   eas_candidate_manifest.csv   [READ ONLY]
#             +
#   eas_curation_state.csv       [MUTABLE]
#             |
#             v
#        JOIN source_id
#             |
#             v
#      curated snapshot
#
# The candidate manifest is NEVER overwritten by this UI.
# Model predictions are intentionally absent from the curation process.
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EAS_DIR = PROJECT_ROOT / "artifacts" / "domain_adaptation" / "eas_v1"

MASTER_MANIFEST = EAS_DIR / "eas_candidate_manifest.csv"
STATE_PATH = EAS_DIR / "eas_curation_state.csv"
SNAPSHOT_PATH = EAS_DIR / "eas_curated_snapshot.csv"

SEMANTIC = ["UNREVIEWED", "VALID", "AMBIGUOUS", "WRONG_LABEL"]

PRESENTATION = [
    "UNREVIEWED",
    "PRODUCT_ONLY",
    "WORN_BY_PERSON",
    "MANNEQUIN",
    "FLAT_LAY",
    "OTHER",
]

HUMAN = ["UNREVIEWED", "NO", "YES"]

COMPOSITION = [
    "UNREVIEWED",
    "CLEAN",
    "REVIEW",
    "SECONDARY_OBJECT",
    "FRAMING_ARTIFACT",
]

DECISION = [
    "UNREVIEWED",
    "VALID_SAMPLE",
    "INVALID_SAMPLE",
    "NEEDS_REVIEW",
]

ADAPTATION_VALUE = [
    "UNREVIEWED",
    "STANDARD",
    "USEFUL_VARIATION",
    "REDUNDANT",
]

STATE_COLUMNS = [
    "source_id",
    "semantic_status",
    "presentation_type",
    "human_present",
    "composition_status",
    "human_review_status",
    "adaptation_value",
    "review_notes",
]

FINAL_STATUSES = [
    "ALL",
    "CANDIDATE",
    "ADAPTATION_ELIGIBLE",
    "CONTEXT_DOMAIN",
    "VALID_OUTSIDE_ADAPTATION_DOMAIN",
    "REVIEW",
    "EXCLUDED",
]


def option_index(options: list[str], value: object) -> int:
    value = str(value)
    return options.index(value) if value in options else 0


def as_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def classify_final_status(row: pd.Series) -> str:
    """
    Classify an EAS candidate without using model outputs.

    ADAPTATION_ELIGIBLE requires:
      - human decision VALID_SAMPLE
      - semantically VALID
      - CLEAN composition
      - PRODUCT_ONLY presentation
      - NO human present
      - file integrity PASS
      - exact duplicate status UNIQUE
      - zero overlap with frozen V2B by dataset_index/source_id/sha256
    """
    semantic = as_text(row.get("semantic_status", "UNREVIEWED"))
    presentation = as_text(row.get("presentation_type", "UNREVIEWED"))
    human = as_text(row.get("human_present", "UNREVIEWED"))
    composition = as_text(row.get("composition_status", "UNREVIEWED"))
    decision = as_text(row.get("human_review_status", "UNREVIEWED"))

    integrity = as_text(row.get("file_integrity", ""))
    duplicate = as_text(row.get("exact_duplicate_status", ""))

    overlap_dataset_index = as_text(
        row.get("v2b_dataset_index_overlap", "NO")
    )
    overlap_source_id = as_text(
        row.get("v2b_source_id_overlap", "NO")
    )
    overlap_sha256 = as_text(
        row.get("v2b_sha256_overlap", "NO")
    )

    if decision == "INVALID_SAMPLE":
        return "EXCLUDED"

    if decision == "NEEDS_REVIEW":
        return "REVIEW"

    no_v2b_overlap = (
        overlap_dataset_index == "NO"
        and overlap_source_id == "NO"
        and overlap_sha256 == "NO"
    )

    base_valid = (
        decision == "VALID_SAMPLE"
        and semantic == "VALID"
        and composition == "CLEAN"
        and integrity == "PASS"
        and duplicate == "UNIQUE"
        and no_v2b_overlap
    )

    if not base_valid:
        return "CANDIDATE"

    if presentation == "PRODUCT_ONLY" and human == "NO":
        return "ADAPTATION_ELIGIBLE"

    if presentation == "WORN_BY_PERSON" and human == "YES":
        return "CONTEXT_DOMAIN"

    return "VALID_OUTSIDE_ADAPTATION_DOMAIN"


def validate_master(master: pd.DataFrame) -> None:
    required = {
        "dataset_index",
        "source_id",
        "source_dataset",
        "mapped_class_id",
        "mapped_class_name",
        "original_article_type",
        "product_display_name",
        "file_integrity",
        "exact_duplicate_status",
        "v2b_dataset_index_overlap",
        "v2b_source_id_overlap",
        "v2b_sha256_overlap",
    }

    missing = required.difference(master.columns)
    if missing:
        raise ValueError(
            f"eas_candidate_manifest.csv missing required columns: "
            f"{sorted(missing)}"
        )

    if master["source_id"].astype(str).duplicated().any():
        raise ValueError(
            "Master manifest contains duplicate source_id values. "
            "Curation is blocked."
        )

    overlap_cols = [
        "v2b_dataset_index_overlap",
        "v2b_source_id_overlap",
        "v2b_sha256_overlap",
    ]
    for col in overlap_cols:
        bad = master[col].astype(str).ne("NO")
        if bad.any():
            raise ValueError(
                f"Frozen V2B overlap detected in master manifest: "
                f"{col} has {int(bad.sum())} non-NO rows."
            )


def initial_state(master: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "semantic_status": "UNREVIEWED",
        "presentation_type": "UNREVIEWED",
        "human_present": "UNREVIEWED",
        "composition_status": "UNREVIEWED",
        "human_review_status": "UNREVIEWED",
        "adaptation_value": "UNREVIEWED",
        "review_notes": "",
    }

    state = pd.DataFrame({"source_id": master["source_id"].astype(str)})

    for col, default in defaults.items():
        if col in master.columns:
            values = master[col].fillna("").astype(str)
            state[col] = values.where(values.ne(""), default)
        else:
            state[col] = default

    return state[STATE_COLUMNS]


def migrate_state(state: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    if "source_id" not in state.columns:
        raise ValueError("eas_curation_state.csv has no source_id column.")

    state = state.copy()
    state["source_id"] = state["source_id"].astype(str)

    if state["source_id"].duplicated().any():
        raise ValueError(
            "eas_curation_state.csv contains duplicate source_id values."
        )

    defaults = {
        "semantic_status": "UNREVIEWED",
        "presentation_type": "UNREVIEWED",
        "human_present": "UNREVIEWED",
        "composition_status": "UNREVIEWED",
        "human_review_status": "UNREVIEWED",
        "adaptation_value": "UNREVIEWED",
        "review_notes": "",
    }

    for col, default in defaults.items():
        if col not in state.columns:
            state[col] = default
        state[col] = state[col].fillna("").astype(str)
        state.loc[state[col].eq(""), col] = default

    # Migrate legacy decision labels if they appear.
    state["human_review_status"] = state["human_review_status"].replace(
        {
            "ACCEPT": "VALID_SAMPLE",
            "REJECT": "INVALID_SAMPLE",
            "REVIEW": "NEEDS_REVIEW",
        }
    )

    # Keep state aligned with current master population.
    master_ids = pd.DataFrame(
        {"source_id": master["source_id"].astype(str)}
    )

    state = master_ids.merge(
        state[STATE_COLUMNS],
        on="source_id",
        how="left",
        validate="one_to_one",
    )

    for col, default in defaults.items():
        state[col] = state[col].fillna(default)

    return state[STATE_COLUMNS]


def load_or_create_state(master: pd.DataFrame) -> pd.DataFrame:
    if STATE_PATH.exists():
        state = pd.read_csv(STATE_PATH, dtype=str).fillna("")
        state = migrate_state(state, master)
    else:
        state = initial_state(master)
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        state.to_csv(STATE_PATH, index=False)

    return state


def save_state(state: pd.DataFrame) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state[STATE_COLUMNS].to_csv(STATE_PATH, index=False)


def build_snapshot(
    master: pd.DataFrame,
    state: pd.DataFrame,
) -> pd.DataFrame:
    snapshot = master.drop(
        columns=[
            c for c in STATE_COLUMNS if c != "source_id" and c in master.columns
        ],
        errors="ignore",
    ).merge(
        state[STATE_COLUMNS],
        on="source_id",
        how="left",
        validate="one_to_one",
    )

    snapshot["final_status"] = snapshot.apply(
        classify_final_status,
        axis=1,
    )

    return snapshot


@st.cache_resource(show_spinner="Carregando dataset externo...")
def get_dataset(dataset_name: str):
    return load_dataset(dataset_name, split="train")


st.set_page_config(
    page_title="EAS v1 Curator",
    page_icon="🧹",
    layout="wide",
)

st.title("🧹 EAS v1 — External Adaptation Set Curator")
st.caption(
    "Candidate Manifest [READ ONLY] + Curation State [MUTABLE]"
)

st.info(
    "Objetivo: selecionar dados válidos para adaptação da CNN sem tocar "
    "na V2B congelada. ADAPTATION_ELIGIBLE exige PRODUCT_ONLY + "
    "human_present=NO + VALID + CLEAN."
)

st.warning(
    "Não use predições de SVM/MLP/CNN durante a curadoria. "
    "A curadoria deve avaliar o dado, não escolher amostras em função "
    "do desempenho do modelo."
)

if not MASTER_MANIFEST.exists():
    st.error(f"Master manifest não encontrado: {MASTER_MANIFEST}")
    st.stop()

try:
    master = pd.read_csv(
        MASTER_MANIFEST,
        dtype={"source_id": str},
    ).fillna("")
    validate_master(master)
    state = load_or_create_state(master)
    snapshot = build_snapshot(master, state)
except Exception as exc:
    st.exception(exc)
    st.stop()

dataset_names = snapshot["source_dataset"].dropna().astype(str).unique()
if len(dataset_names) != 1:
    st.error(
        "O EAS v1 esperava exatamente uma source_dataset, mas encontrou: "
        f"{dataset_names.tolist()}"
    )
    st.stop()

dataset_name = dataset_names[0]
ds = get_dataset(dataset_name)

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------

classes = sorted(snapshot["mapped_class_id"].astype(int).unique())

class_id = st.sidebar.selectbox(
    "Classe",
    classes,
    format_func=lambda x: (
        f"{x} — "
        f"{snapshot.loc[snapshot.mapped_class_id.astype(int) == x, 'mapped_class_name'].iloc[0]}"
    ),
)

status_filter = st.sidebar.selectbox(
    "Status final",
    FINAL_STATUSES,
)

review_filter = st.sidebar.selectbox(
    "Revisão humana",
    [
        "ALL",
        "UNREVIEWED",
        "VALID_SAMPLE",
        "INVALID_SAMPLE",
        "NEEDS_REVIEW",
    ],
)

adaptation_filter = st.sidebar.selectbox(
    "Valor para adaptação",
    ["ALL"] + ADAPTATION_VALUE,
)

page_size = st.sidebar.selectbox(
    "Por página",
    [5, 10, 20, 30],
    index=1,
)

st.sidebar.divider()
st.sidebar.caption(
    f"Master: `{MASTER_MANIFEST.name}`"
)
st.sidebar.caption(
    f"State: `{STATE_PATH.name}`"
)
st.sidebar.caption(
    "V2B overlap contract: **0 / 0 / 0**"
)

# ---------------------------------------------------------------------
# Filter view
# ---------------------------------------------------------------------

view = snapshot[
    snapshot["mapped_class_id"].astype(int) == int(class_id)
].copy()

if status_filter != "ALL":
    view = view[view["final_status"] == status_filter]

if review_filter != "ALL":
    view = view[
        view["human_review_status"] == review_filter
    ]

if adaptation_filter != "ALL":
    view = view[
        view["adaptation_value"] == adaptation_filter
    ]

view = view.sort_values("dataset_index", kind="stable")

page_max = max(1, (len(view) + page_size - 1) // page_size)
page = st.number_input(
    "Página",
    min_value=1,
    max_value=page_max,
    value=1,
    step=1,
)

page_view = view.iloc[
    (int(page) - 1) * page_size : int(page) * page_size
]

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

class_df = snapshot[
    snapshot["mapped_class_id"].astype(int) == int(class_id)
]

m1, m2, m3, m4, m5, m6 = st.columns(6)

m1.metric("Candidates", len(class_df))
m2.metric(
    "Reviewed",
    int(
        (
            class_df["human_review_status"]
            != "UNREVIEWED"
        ).sum()
    ),
)
m3.metric(
    "Valid samples",
    int(
        (
            class_df["human_review_status"]
            == "VALID_SAMPLE"
        ).sum()
    ),
)
m4.metric(
    "Adaptation eligible",
    int(
        (
            class_df["final_status"]
            == "ADAPTATION_ELIGIBLE"
        ).sum()
    ),
)
m5.metric(
    "Useful variation",
    int(
        (
            class_df["adaptation_value"]
            == "USEFUL_VARIATION"
        ).sum()
    ),
)
m6.metric(
    "Excluded",
    int(
        (
            class_df["final_status"]
            == "EXCLUDED"
        ).sum()
    ),
)

st.caption(
    f"Exibindo {len(page_view)} de {len(view)} amostras após filtros."
)

# ---------------------------------------------------------------------
# Curation cards
# ---------------------------------------------------------------------

for _, rec in page_view.iterrows():
    source_id = str(rec["source_id"])
    dataset_index = int(rec["dataset_index"])

    try:
        image = ds[dataset_index]["image"].convert("RGB")
    except Exception as exc:
        st.error(
            f"Falha ao carregar dataset_index={dataset_index}: {exc}"
        )
        continue

    st.divider()
    c1, c2, c3 = st.columns([1.10, 0.95, 1.35])

    with c1:
        st.image(
            image,
            use_container_width=True,
        )

    with c2:
        st.markdown(
            f"### {rec['mapped_class_id']} — "
            f"{rec['mapped_class_name']}"
        )
        st.caption(
            f"source_id: `{source_id}` | "
            f"dataset_index: `{dataset_index}`"
        )
        st.caption(
            f"articleType: `{rec['original_article_type']}`"
        )
        st.write(as_text(rec["product_display_name"]))

        st.write(
            f"**Integridade:** {as_text(rec['file_integrity'])}"
        )
        st.write(
            f"**Duplicata exata:** "
            f"{as_text(rec['exact_duplicate_status'])}"
        )

        if "width" in rec.index and "height" in rec.index:
            st.write(
                f"**Resolução:** "
                f"{as_text(rec['width'])}×{as_text(rec['height'])}"
            )

        if "resolution_status" in rec.index:
            st.write(
                f"**Resolution status:** "
                f"{as_text(rec['resolution_status'])}"
            )

        if "quality_status" in rec.index:
            st.write(
                f"**Quality status:** "
                f"{as_text(rec['quality_status'])}"
            )

        st.markdown("#### V2B overlap")
        st.write(
            "dataset_index: "
            f"**{as_text(rec['v2b_dataset_index_overlap'])}**"
        )
        st.write(
            "source_id: "
            f"**{as_text(rec['v2b_source_id_overlap'])}**"
        )
        st.write(
            "sha256: "
            f"**{as_text(rec['v2b_sha256_overlap'])}**"
        )

        status = classify_final_status(rec)

        st.markdown("#### Status")
        if status == "ADAPTATION_ELIGIBLE":
            st.success("ADAPTATION_ELIGIBLE")
        elif status == "CONTEXT_DOMAIN":
            st.info(
                "CONTEXT_DOMAIN — válido, fora do EAS principal"
            )
        elif status == "VALID_OUTSIDE_ADAPTATION_DOMAIN":
            st.info(
                "VALID_OUTSIDE_ADAPTATION_DOMAIN"
            )
        elif status == "EXCLUDED":
            st.error("EXCLUDED")
        elif status == "REVIEW":
            st.warning("REVIEW")
        else:
            st.caption("CANDIDATE")

    with c3:
        sem = st.selectbox(
            "Semântica",
            SEMANTIC,
            index=option_index(
                SEMANTIC,
                rec["semantic_status"],
            ),
            key=f"sem_{source_id}",
        )

        pres = st.selectbox(
            "Presentation type",
            PRESENTATION,
            index=option_index(
                PRESENTATION,
                rec["presentation_type"],
            ),
            key=f"pres_{source_id}",
        )

        hum = st.selectbox(
            "Human present",
            HUMAN,
            index=option_index(
                HUMAN,
                rec["human_present"],
            ),
            key=f"hum_{source_id}",
        )

        comp = st.selectbox(
            "Composição",
            COMPOSITION,
            index=option_index(
                COMPOSITION,
                rec["composition_status"],
            ),
            key=f"comp_{source_id}",
            help=(
                "Em WORN_BY_PERSON, a pessoa que veste o item não é "
                "SECONDARY_OBJECT apenas por estar presente."
            ),
        )

        dec = st.selectbox(
            "Validade da amostra",
            DECISION,
            index=option_index(
                DECISION,
                rec["human_review_status"],
            ),
            key=f"dec_{source_id}",
            help=(
                "VALID_SAMPLE significa que o dado é válido. "
                "A elegibilidade para adaptação é calculada separadamente."
            ),
        )

        value = st.selectbox(
            "Valor para adaptação",
            ADAPTATION_VALUE,
            index=option_index(
                ADAPTATION_VALUE,
                rec["adaptation_value"],
            ),
            key=f"value_{source_id}",
            help=(
                "STANDARD = exemplo normal; USEFUL_VARIATION = variação "
                "válida e informativa; REDUNDANT = exemplo válido, mas "
                "visualmente repetitivo. Não use erro/acerto do modelo."
            ),
        )

        notes = st.text_area(
            "Notas",
            value=as_text(rec["review_notes"]),
            key=f"notes_{source_id}",
            height=90,
        )

        preview = rec.copy()
        preview["semantic_status"] = sem
        preview["presentation_type"] = pres
        preview["human_present"] = hum
        preview["composition_status"] = comp
        preview["human_review_status"] = dec
        preview["adaptation_value"] = value

        predicted = classify_final_status(preview)
        st.caption(
            f"Resultado previsto: **{predicted}**"
        )

        if pres == "WORN_BY_PERSON" and hum == "YES":
            st.caption(
                "A pessoa faz parte da apresentação. "
                "Esse caso pertence ao contextual domain, não ao "
                "PRODUCT_ONLY principal."
            )

        if (
            dec == "VALID_SAMPLE"
            and sem == "VALID"
            and comp == "CLEAN"
            and pres == "PRODUCT_ONLY"
            and hum == "NO"
        ):
            st.caption(
                "✓ Estrutura humana compatível com "
                "ADAPTATION_ELIGIBLE; integridade/duplicatas/overlap "
                "também precisam passar."
            )

        if st.button(
            "Salvar avaliação",
            key=f"save_{source_id}",
            use_container_width=True,
        ):
            mask = state["source_id"].astype(str) == source_id

            if int(mask.sum()) != 1:
                st.error(
                    "Estado inconsistente: source_id não é único."
                )
            else:
                state.loc[
                    mask,
                    [
                        "semantic_status",
                        "presentation_type",
                        "human_present",
                        "composition_status",
                        "human_review_status",
                        "adaptation_value",
                        "review_notes",
                    ],
                ] = [
                    sem,
                    pres,
                    hum,
                    comp,
                    dec,
                    value,
                    notes,
                ]

                save_state(state)

                refreshed = build_snapshot(
                    master,
                    state,
                )
                saved_status = refreshed.loc[
                    refreshed["source_id"].astype(str) == source_id,
                    "final_status",
                ].iloc[0]

                st.success(
                    f"Salvo em eas_curation_state.csv: {saved_status}"
                )

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

st.divider()
st.subheader("Resumo por classe")

snapshot = build_snapshot(master, state)

summary = (
    snapshot.groupby(
        ["mapped_class_id", "mapped_class_name"]
    )
    .agg(
        candidates=("dataset_index", "size"),
        reviewed=(
            "human_review_status",
            lambda s: int((s != "UNREVIEWED").sum()),
        ),
        valid_samples=(
            "human_review_status",
            lambda s: int((s == "VALID_SAMPLE").sum()),
        ),
        adaptation_eligible=(
            "final_status",
            lambda s: int((s == "ADAPTATION_ELIGIBLE").sum()),
        ),
        useful_variation=(
            "adaptation_value",
            lambda s: int((s == "USEFUL_VARIATION").sum()),
        ),
        redundant=(
            "adaptation_value",
            lambda s: int((s == "REDUNDANT").sum()),
        ),
        context_domain=(
            "final_status",
            lambda s: int((s == "CONTEXT_DOMAIN").sum()),
        ),
        outside_domain=(
            "final_status",
            lambda s: int(
                (s == "VALID_OUTSIDE_ADAPTATION_DOMAIN").sum()
            ),
        ),
        needs_review=(
            "final_status",
            lambda s: int((s == "REVIEW").sum()),
        ),
        excluded=(
            "final_status",
            lambda s: int((s == "EXCLUDED").sum()),
        ),
    )
    .reset_index()
)

st.dataframe(
    summary,
    hide_index=True,
    use_container_width=True,
)

total_reviewed = int(
    (snapshot["human_review_status"] != "UNREVIEWED").sum()
)
total_eligible = int(
    (snapshot["final_status"] == "ADAPTATION_ELIGIBLE").sum()
)

progress = total_reviewed / max(len(snapshot), 1)

st.progress(
    progress,
    text=(
        f"Curadoria: {total_reviewed}/{len(snapshot)} "
        f"({progress:.1%}) | "
        f"ADAPTATION_ELIGIBLE: {total_eligible}"
    ),
)

# ---------------------------------------------------------------------
# Snapshot / downloads
# ---------------------------------------------------------------------

st.divider()
st.subheader("Exportação e checkpoint")

st.caption(
    "O master nunca é alterado. O estado humano fica em "
    "`eas_curation_state.csv`. O snapshot é uma visão derivada."
)

ec1, ec2, ec3 = st.columns(3)

with ec1:
    st.download_button(
        "⬇️ Baixar curation state",
        state.to_csv(index=False).encode("utf-8"),
        "eas_curation_state.csv",
        "text/csv",
        use_container_width=True,
    )

with ec2:
    st.download_button(
        "⬇️ Baixar curated snapshot",
        snapshot.to_csv(index=False).encode("utf-8"),
        "eas_curated_snapshot.csv",
        "text/csv",
        use_container_width=True,
    )

with ec3:
    if st.button(
        "💾 Salvar snapshot no projeto",
        use_container_width=True,
    ):
        SNAPSHOT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        snapshot.to_csv(
            SNAPSHOT_PATH,
            index=False,
        )
        st.success(
            f"Snapshot salvo: {SNAPSHOT_PATH}"
        )

st.caption(
    "EAS v1 permanece CANDIDATE / CURATION IN PROGRESS até que "
    "a seleção, o split Train/Validation e o freeze sejam executados."
)
