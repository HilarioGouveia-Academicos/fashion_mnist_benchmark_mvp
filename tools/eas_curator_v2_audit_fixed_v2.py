from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EAS_DIR = PROJECT_ROOT / "artifacts" / "domain_adaptation" / "eas_v1"

MASTER_MANIFEST = EAS_DIR / "eas_candidate_manifest.csv"
STATE_PATH = EAS_DIR / "eas_curation_state.csv"
SNAPSHOT_PATH = EAS_DIR / "eas_curated_snapshot.csv"

DECISIONS = ["UNREVIEWED", "SELECTED", "REJECTED"]
STATUS_FILTERS = [
    "ALL",
    "UNREVIEWED",
    "SELECTED",
    "REJECTED",
    "ADAPTATION_ELIGIBLE",
    "SYSTEM_BLOCKED",
]

STATE_COLUMNS = [
    "source_id",
    "curation_decision",
    "rejection_reason",
    "review_notes",
]

REJECTION_REASONS = [
    "WORN_BY_PERSON",
    "SECONDARY_OBJECT",
    "SEMANTIC_MISMATCH",
    "OTHER",
]

AUDIT_CLASS_IDS = {1, 6}  # Trouser, Shirt


def ensure_state_schema(state: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza o schema do estado de curadoria.

    Regras:
    - source_id é sempre string;
    - a decisão canônica chama-se curation_decision;
    - arquivos antigos com review_decision/human_review_status são migrados;
    - review_notes sempre existe.
    """
    state = state.copy()

    if "source_id" not in state.columns:
        raise ValueError("eas_curation_state.csv has no source_id column.")

    state["source_id"] = state["source_id"].astype(str)

    if "curation_decision" not in state.columns:
        if "review_decision" in state.columns:
            state["curation_decision"] = state["review_decision"]
        elif "human_review_status" in state.columns:
            state["curation_decision"] = (
                state["human_review_status"]
                .astype(str)
                .map(
                    {
                        "VALID_SAMPLE": "SELECTED",
                        "INVALID_SAMPLE": "REJECTED",
                        "NEEDS_REVIEW": "UNREVIEWED",
                        "UNREVIEWED": "UNREVIEWED",
                    }
                )
                .fillna("UNREVIEWED")
            )
        else:
            state["curation_decision"] = "UNREVIEWED"

    if "rejection_reason" not in state.columns:
        state["rejection_reason"] = ""

    if "rejection_reason" not in state.columns:
        state["rejection_reason"] = ""

    if "review_notes" not in state.columns:
        state["review_notes"] = ""

    state["curation_decision"] = (
        state["curation_decision"]
        .fillna("UNREVIEWED")
        .astype(str)
    )

    state.loc[
        ~state["curation_decision"].isin(DECISIONS),
        "curation_decision",
    ] = "UNREVIEWED"

    state["rejection_reason"] = (
        state["rejection_reason"]
        .fillna("")
        .astype(str)
    )
    state.loc[
        ~state["rejection_reason"].isin([""] + REJECTION_REASONS),
        "rejection_reason",
    ] = ""

    state["review_notes"] = (
        state["review_notes"]
        .fillna("")
        .astype(str)
    )

    if state["source_id"].duplicated().any():
        raise ValueError("eas_curation_state.csv contains duplicate source_id values.")

    return state[STATE_COLUMNS]



def as_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


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
            f"eas_candidate_manifest.csv missing required columns: {sorted(missing)}"
        )

    if master["source_id"].astype(str).duplicated().any():
        raise ValueError("Master manifest contains duplicate source_id values.")

    for col in [
        "v2b_dataset_index_overlap",
        "v2b_source_id_overlap",
        "v2b_sha256_overlap",
    ]:
        bad = master[col].astype(str).ne("NO")
        if bad.any():
            raise ValueError(
                f"Frozen V2B overlap detected: {col} has {int(bad.sum())} non-NO rows."
            )


def initial_state(master: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": master["source_id"].astype(str),
            "curation_decision": "UNREVIEWED",
            "rejection_reason": "",
            "review_notes": "",
        }
    )


def migrate_state(state: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    if "source_id" not in state.columns:
        raise ValueError("eas_curation_state.csv has no source_id column.")

    state = state.copy()
    state["source_id"] = state["source_id"].astype(str)

    if state["source_id"].duplicated().any():
        raise ValueError("eas_curation_state.csv contains duplicate source_id values.")

    if "curation_decision" not in state.columns:
        if "human_review_status" in state.columns:
            state["curation_decision"] = (
                state["human_review_status"]
                .astype(str)
                .map(
                    {
                        "VALID_SAMPLE": "SELECTED",
                        "INVALID_SAMPLE": "REJECTED",
                        "NEEDS_REVIEW": "UNREVIEWED",
                        "UNREVIEWED": "UNREVIEWED",
                    }
                )
                .fillna("UNREVIEWED")
            )
        else:
            state["curation_decision"] = "UNREVIEWED"

    if "rejection_reason" not in state.columns:
        state["rejection_reason"] = ""

    if "review_notes" not in state.columns:
        state["review_notes"] = ""

    state["curation_decision"] = (
        state["curation_decision"].fillna("UNREVIEWED").astype(str)
    )
    state.loc[
        ~state["curation_decision"].isin(DECISIONS),
        "curation_decision",
    ] = "UNREVIEWED"

    state["rejection_reason"] = state["rejection_reason"].fillna("").astype(str)
    state.loc[
        ~state["rejection_reason"].isin([""] + REJECTION_REASONS),
        "rejection_reason",
    ] = ""

    state["review_notes"] = state["review_notes"].fillna("").astype(str)

    master_ids = pd.DataFrame({"source_id": master["source_id"].astype(str)})

    state = master_ids.merge(
        state[STATE_COLUMNS],
        on="source_id",
        how="left",
        validate="one_to_one",
    )

    state["curation_decision"] = state["curation_decision"].fillna("UNREVIEWED")
    state["rejection_reason"] = state["rejection_reason"].fillna("")
    state["review_notes"] = state["review_notes"].fillna("")

    return state[STATE_COLUMNS]


def save_state(state: pd.DataFrame) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state[STATE_COLUMNS].to_csv(STATE_PATH, index=False)


def load_or_create_state(master: pd.DataFrame) -> pd.DataFrame:
    if STATE_PATH.exists():
        state = pd.read_csv(
            STATE_PATH,
            dtype={"source_id": str},
        ).fillna("")
        state = migrate_state(state, master)
    else:
        state = initial_state(master)

    state = ensure_state_schema(state)
    save_state(state)
    return state



def system_gate(row: pd.Series) -> tuple[bool, list[str]]:
    reasons = []

    if as_text(row.get("file_integrity", "")) != "PASS":
        reasons.append("FILE_INTEGRITY")

    if as_text(row.get("exact_duplicate_status", "")) != "UNIQUE":
        reasons.append("EXACT_DUPLICATE")

    for col, label in [
        ("v2b_dataset_index_overlap", "V2B_DATASET_INDEX"),
        ("v2b_source_id_overlap", "V2B_SOURCE_ID"),
        ("v2b_sha256_overlap", "V2B_SHA256"),
    ]:
        if as_text(row.get(col, "NO")) != "NO":
            reasons.append(label)

    return len(reasons) == 0, reasons


def derive_status(row: pd.Series) -> str:
    decision = as_text(row.get("curation_decision", "UNREVIEWED"))

    if decision == "UNREVIEWED":
        return "UNREVIEWED"

    if decision == "REJECTED":
        return "REJECTED"

    gate_ok, _ = system_gate(row)

    if decision == "SELECTED":
        return "ADAPTATION_ELIGIBLE" if gate_ok else "SYSTEM_BLOCKED"

    return "UNREVIEWED"


def build_snapshot(
    master: pd.DataFrame,
    state: pd.DataFrame,
) -> pd.DataFrame:
    master = master.copy()
    state = ensure_state_schema(state)

    # Chave de merge com o MESMO dtype nos dois lados.
    master["source_id"] = master["source_id"].astype(str)
    state["source_id"] = state["source_id"].astype(str)

    # O estado humano é a fonte oficial dessas colunas.
    state_merge = state[
        ["source_id", "curation_decision", "rejection_reason", "review_notes"]
    ].copy()

    # Evita colisões do tipo *_x / *_y caso o master tenha colunas antigas.
    master = master.drop(
        columns=[
            col
            for col in [
                "curation_decision",
                "rejection_reason",
                "review_notes",
                "final_status",
            ]
            if col in master.columns
        ],
        errors="ignore",
    )

    snapshot = master.merge(
        state_merge,
        on="source_id",
        how="left",
        validate="one_to_one",
    )

    snapshot["curation_decision"] = (
        snapshot["curation_decision"]
        .fillna("UNREVIEWED")
        .astype(str)
    )

    snapshot["rejection_reason"] = (
        snapshot["rejection_reason"]
        .fillna("")
        .astype(str)
    )

    snapshot["review_notes"] = (
        snapshot["review_notes"]
        .fillna("")
        .astype(str)
    )

    # Status derivado — não é salvo no state.
    snapshot["final_status"] = snapshot.apply(derive_status, axis=1)

    # Campos semânticos derivados para os SELECTED elegíveis.
    eligible = snapshot["final_status"].eq("ADAPTATION_ELIGIBLE")
    snapshot["semantic_status"] = ""
    snapshot["presentation_type"] = ""
    snapshot["human_present"] = ""
    snapshot["composition_status"] = ""
    snapshot["human_review_status"] = ""

    snapshot.loc[eligible, "semantic_status"] = "VALID"
    snapshot.loc[eligible, "presentation_type"] = "PRODUCT_ONLY"
    snapshot.loc[eligible, "human_present"] = "NO"
    snapshot.loc[eligible, "composition_status"] = "CLEAN"
    snapshot.loc[eligible, "human_review_status"] = "VALID_SAMPLE"

    return snapshot



def save_decision(
    state: pd.DataFrame,
    source_id: str,
    decision: str,
    notes: str = "",
) -> None:
    mask = state["source_id"].astype(str) == str(source_id)

    if int(mask.sum()) != 1:
        raise ValueError(f"source_id must be unique in state: {source_id}")

    state.loc[mask, "curation_decision"] = decision
    state.loc[mask, "review_notes"] = notes

    # Motivo só faz sentido quando a amostra continua REJECTED.
    if decision != "REJECTED":
        state.loc[mask, "rejection_reason"] = ""

    save_state(state)


def save_rejection_reason(
    state: pd.DataFrame,
    source_id: str,
    reason: str,
) -> None:
    if reason not in REJECTION_REASONS:
        raise ValueError(f"Invalid rejection reason: {reason}")

    mask = state["source_id"].astype(str) == str(source_id)

    if int(mask.sum()) != 1:
        raise ValueError(f"source_id must be unique in state: {source_id}")

    decision = as_text(state.loc[mask, "curation_decision"].iloc[0])
    if decision != "REJECTED":
        raise ValueError(
            f"rejection_reason can only be assigned to REJECTED samples: {source_id}"
        )

    state.loc[mask, "rejection_reason"] = reason
    save_state(state)


@st.cache_resource(show_spinner="Carregando dataset externo...")
def get_dataset(dataset_name: str):
    return load_dataset(dataset_name, split="train")


st.set_page_config(
    page_title="EAS v1 Curator",
    page_icon="🧹",
    layout="wide",
)

st.title("🧹 EAS v1 — Product-Only Curator")
st.caption(
    "Selecione apenas imagens que representem corretamente a classe e sejam "
    "PRODUCT_ONLY. O restante é calculado automaticamente."
)

st.info(
    "Decisão humana: SELECTED ou REJECTED. "
    "Integridade, duplicatas e overlap com V2B são gates automáticos."
)

st.warning("Não use predições dos modelos durante a curadoria.")

st.info(
    "Auditoria pós-curadoria: para Trouser e Shirt rejeitados, classifique o motivo "
    "da rejeição sem alterar a decisão original."
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
    st.error(f"Expected one source_dataset, found: {dataset_names.tolist()}")
    st.stop()

dataset_name = dataset_names[0]
ds = get_dataset(dataset_name)

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

st.sidebar.header("Filtros")

classes = sorted(snapshot["mapped_class_id"].astype(int).unique())

class_labels = {}
for cid in classes:
    name = snapshot.loc[
        snapshot["mapped_class_id"].astype(int) == cid,
        "mapped_class_name",
    ].iloc[0]
    class_labels[cid] = f"{cid} — {name}"

class_id = st.sidebar.selectbox(
    "Classe",
    classes,
    format_func=lambda x: class_labels[x],
)

class_scope = snapshot[
    snapshot["mapped_class_id"].astype(int) == int(class_id)
].copy()

article_types = sorted(
    x
    for x in class_scope["original_article_type"].astype(str).unique()
    if x
)

article_type = st.sidebar.selectbox(
    "articleType",
    ["ALL"] + article_types,
    help=(
        "Útil especialmente em Ankle boot para explorar subclasses candidatas "
        "sem assumir equivalência automática."
    ),
)

status_filter = st.sidebar.selectbox(
    "Status",
    STATUS_FILTERS,
)

page_size = st.sidebar.selectbox(
    "Itens por página",
    [12, 16, 20, 24, 32, 40],
    index=2,
)

grid_columns = st.sidebar.selectbox(
    "Colunas do grid",
    [3, 4, 5],
    index=1,
)

st.sidebar.divider()
st.sidebar.caption(f"Master: `{MASTER_MANIFEST.name}`")
st.sidebar.caption(f"State: `{STATE_PATH.name}`")
st.sidebar.caption("V2B overlap contract: **0 / 0 / 0**")

# ---------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------

view = class_scope.copy()

if article_type != "ALL":
    view = view[view["original_article_type"] == article_type]

if status_filter != "ALL":
    view = view[view["final_status"] == status_filter]

view = view.sort_values(
    ["original_article_type", "dataset_index"],
    kind="stable",
)

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Candidates", len(class_scope))
m2.metric(
    "Reviewed",
    int((class_scope["curation_decision"] != "UNREVIEWED").sum()),
)
m3.metric(
    "Selected",
    int((class_scope["curation_decision"] == "SELECTED").sum()),
)
m4.metric(
    "Eligible",
    int((class_scope["final_status"] == "ADAPTATION_ELIGIBLE").sum()),
)
m5.metric(
    "Rejected",
    int((class_scope["curation_decision"] == "REJECTED").sum()),
)

if int(class_id) == 9:
    st.info(
        "Classe 9 — Ankle boot: use `articleType` para investigar "
        "Casual Shoes, Formal Shoes, Heels, Flats etc. "
        "A decisão final continua visual e humana."
    )

# ---------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------

page_max = max(1, (len(view) + page_size - 1) // page_size)

page = st.number_input(
    "Página",
    min_value=1,
    max_value=page_max,
    value=1,
    step=1,
)

start = (int(page) - 1) * page_size
page_view = view.iloc[start : start + page_size]

st.caption(f"Exibindo {len(page_view)} de {len(view)} registros filtrados.")

# ---------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------

if page_view.empty:
    st.warning("Nenhuma imagem encontrada para os filtros atuais.")
else:
    records = list(page_view.to_dict(orient="records"))

    for start_idx in range(0, len(records), grid_columns):
        row_records = records[start_idx : start_idx + grid_columns]
        cols = st.columns(grid_columns)

        for col, rec in zip(cols, row_records):
            source_id = str(rec["source_id"])
            dataset_index = int(rec["dataset_index"])

            with col:
                try:
                    image = ds[dataset_index]["image"].convert("RGB")
                except Exception as exc:
                    st.error(f"Erro ao carregar imagem: {exc}")
                    continue

                st.image(image, use_container_width=True)

                st.markdown(f"**{rec['mapped_class_name']}**")
                st.caption(f"`{rec['original_article_type']}`")

                description = as_text(rec["product_display_name"])
                if description:
                    st.write(description)

                decision = as_text(rec["curation_decision"])
                final_status = as_text(rec["final_status"])

                if final_status == "ADAPTATION_ELIGIBLE":
                    st.success("✓ SELECTED / ELIGIBLE")
                elif final_status == "SYSTEM_BLOCKED":
                    _, reasons = system_gate(pd.Series(rec))
                    st.error("SELECTED, but blocked: " + ", ".join(reasons))
                elif decision == "REJECTED":
                    st.error("✕ REJECTED")
                else:
                    st.caption("○ UNREVIEWED")

                b1, b2 = st.columns(2)

                with b1:
                    if st.button(
                        "✅ Selecionar",
                        key=f"select_{source_id}",
                        use_container_width=True,
                    ):
                        save_decision(state, source_id, "SELECTED")
                        st.rerun()

                with b2:
                    if st.button(
                        "❌ Rejeitar",
                        key=f"reject_{source_id}",
                        use_container_width=True,
                    ):
                        save_decision(state, source_id, "REJECTED")
                        st.rerun()

                if decision != "UNREVIEWED":
                    if st.button(
                        "↩ Limpar",
                        key=f"reset_{source_id}",
                        use_container_width=True,
                    ):
                        save_decision(state, source_id, "UNREVIEWED")
                        st.rerun()

                # Auditoria pós-curadoria: somente Trouser/Shirt rejeitados.
                if (
                    decision == "REJECTED"
                    and int(rec["mapped_class_id"]) in AUDIT_CLASS_IDS
                ):
                    st.markdown("**Motivo da rejeição**")

                    current_reason = as_text(rec.get("rejection_reason", ""))
                    if current_reason:
                        st.caption(f"Atual: `{current_reason}`")
                    else:
                        st.caption("Ainda não auditado")

                    r1, r2 = st.columns(2)
                    r3, r4 = st.columns(2)

                    with r1:
                        if st.button(
                            "👤 Worn by person",
                            key=f"reason_worn_{source_id}",
                            use_container_width=True,
                        ):
                            save_rejection_reason(
                                state, source_id, "WORN_BY_PERSON"
                            )
                            st.rerun()

                    with r2:
                        if st.button(
                            "📦 Secondary object",
                            key=f"reason_secondary_{source_id}",
                            use_container_width=True,
                        ):
                            save_rejection_reason(
                                state, source_id, "SECONDARY_OBJECT"
                            )
                            st.rerun()

                    with r3:
                        if st.button(
                            "❓ Semantic mismatch",
                            key=f"reason_semantic_{source_id}",
                            use_container_width=True,
                        ):
                            save_rejection_reason(
                                state, source_id, "SEMANTIC_MISMATCH"
                            )
                            st.rerun()

                    with r4:
                        if st.button(
                            "⚪ Other",
                            key=f"reason_other_{source_id}",
                            use_container_width=True,
                        ):
                            save_rejection_reason(
                                state, source_id, "OTHER"
                            )
                            st.rerun()

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

st.divider()
st.subheader("Resumo por classe")

snapshot = build_snapshot(master, state)

summary = (
    snapshot.groupby(["mapped_class_id", "mapped_class_name"])
    .agg(
        candidates=("dataset_index", "size"),
        reviewed=(
            "curation_decision",
            lambda s: int((s != "UNREVIEWED").sum()),
        ),
        selected=(
            "curation_decision",
            lambda s: int((s == "SELECTED").sum()),
        ),
        eligible=(
            "final_status",
            lambda s: int((s == "ADAPTATION_ELIGIBLE").sum()),
        ),
        rejected=(
            "curation_decision",
            lambda s: int((s == "REJECTED").sum()),
        ),
        system_blocked=(
            "final_status",
            lambda s: int((s == "SYSTEM_BLOCKED").sum()),
        ),
    )
    .reset_index()
)

st.dataframe(summary, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------
# Rejected-sample audit: Trouser / Shirt
# ---------------------------------------------------------------------

st.subheader("Auditoria dos rejeitados — Trouser / Shirt")

audit_scope = snapshot[
    snapshot["mapped_class_id"].astype(int).isin(AUDIT_CLASS_IDS)
    & snapshot["curation_decision"].eq("REJECTED")
].copy()

audit_total = len(audit_scope)
audit_labeled = int(audit_scope["rejection_reason"].astype(str).ne("").sum())
audit_progress = audit_labeled / max(audit_total, 1)

st.progress(
    audit_progress,
    text=(
        f"Auditoria: {audit_labeled}/{audit_total} "
        f"({audit_progress:.1%})"
    ),
)

audit_summary = (
    audit_scope.assign(
        rejection_reason=audit_scope["rejection_reason"].replace("", "UNLABELED")
    )
    .groupby(["mapped_class_id", "mapped_class_name", "rejection_reason"])
    .size()
    .reset_index(name="count")
)

audit_totals = (
    audit_scope.groupby(["mapped_class_id", "mapped_class_name"])
    .size()
    .reset_index(name="class_rejected_total")
)

audit_summary = audit_summary.merge(
    audit_totals,
    on=["mapped_class_id", "mapped_class_name"],
    how="left",
)

audit_summary["share_of_rejected"] = (
    audit_summary["count"] / audit_summary["class_rejected_total"]
)

st.dataframe(
    audit_summary.sort_values(
        ["mapped_class_id", "count"],
        ascending=[True, False],
    ),
    hide_index=True,
    use_container_width=True,
    column_config={
        "share_of_rejected": st.column_config.NumberColumn(
            "share_of_rejected",
            format="%.1%%",
        )
    },
)

# ---------------------------------------------------------------------
# articleType analysis
# ---------------------------------------------------------------------

st.subheader("Distribuição por articleType")

article_summary = (
    snapshot[
        snapshot["mapped_class_id"].astype(int) == int(class_id)
    ]
    .groupby("original_article_type")
    .agg(
        candidates=("dataset_index", "size"),
        reviewed=(
            "curation_decision",
            lambda s: int((s != "UNREVIEWED").sum()),
        ),
        selected=(
            "curation_decision",
            lambda s: int((s == "SELECTED").sum()),
        ),
        eligible=(
            "final_status",
            lambda s: int((s == "ADAPTATION_ELIGIBLE").sum()),
        ),
    )
    .reset_index()
    .sort_values(
        ["eligible", "selected", "candidates"],
        ascending=False,
    )
)

st.dataframe(article_summary, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------
# Overall progress
# ---------------------------------------------------------------------

total_reviewed = int(
    (snapshot["curation_decision"] != "UNREVIEWED").sum()
)

total_eligible = int(
    (snapshot["final_status"] == "ADAPTATION_ELIGIBLE").sum()
)

progress = total_reviewed / max(len(snapshot), 1)

st.progress(
    progress,
    text=(
        f"Curadoria: {total_reviewed}/{len(snapshot)} "
        f"({progress:.1%}) | Eligible: {total_eligible}"
    ),
)

# ---------------------------------------------------------------------
# Export / checkpoint
# ---------------------------------------------------------------------

st.divider()
st.subheader("Exportação e checkpoint")

st.caption(
    "O master permanece somente leitura. "
    "O estado humano é salvo separadamente."
)

e1, e2, e3 = st.columns(3)

with e1:
    st.download_button(
        "⬇️ Baixar curation state",
        state.to_csv(index=False).encode("utf-8"),
        "eas_curation_state.csv",
        "text/csv",
        use_container_width=True,
    )

with e2:
    st.download_button(
        "⬇️ Baixar curated snapshot",
        snapshot.to_csv(index=False).encode("utf-8"),
        "eas_curated_snapshot.csv",
        "text/csv",
        use_container_width=True,
    )

with e3:
    if st.button(
        "💾 Salvar snapshot no projeto",
        use_container_width=True,
    ):
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        snapshot.to_csv(SNAPSHOT_PATH, index=False)
        st.success(f"Snapshot salvo: {SNAPSHOT_PATH}")

st.caption(
    "A curadoria principal do EAS v1 permanece preservada. "
    "A auditoria de rejeição adiciona apenas rejection_reason para Trouser/Shirt "
    "sem alterar SELECTED, REJECTED ou ADAPTATION_ELIGIBLE."
)
