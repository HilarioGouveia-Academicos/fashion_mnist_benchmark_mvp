
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
import streamlit as st
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v2b.curation_rules import classify_final_status

MANIFEST = PROJECT_ROOT / "artifacts/v2b/v2b_candidate_manifest.csv"
CONFIG = PROJECT_ROOT / "config/v2b_mapping.json"

SEMANTIC = ["UNREVIEWED","VALID","AMBIGUOUS","WRONG_LABEL"]
PRESENTATION = ["UNREVIEWED","PRODUCT_ONLY","WORN_BY_PERSON","MANNEQUIN","FLAT_LAY","OTHER"]
HUMAN = ["UNREVIEWED","NO","YES"]
COMPOSITION = ["UNREVIEWED","CLEAN","REVIEW","SECONDARY_OBJECT","FRAMING_ARTIFACT"]
DECISION = ["UNREVIEWED","VALID_SAMPLE","INVALID_SAMPLE","NEEDS_REVIEW"]

def idx(options, value):
    return options.index(value) if value in options else 0

def migrate(df):
    for col, default in {
        "semantic_status":"UNREVIEWED",
        "presentation_type":"UNREVIEWED",
        "human_present":"UNREVIEWED",
        "composition_status":"UNREVIEWED",
        "human_review_status":"UNREVIEWED",
        "review_notes":"",
        "final_status":"CANDIDATE",
    }.items():
        if col not in df.columns:
            df[col] = default
    df["human_review_status"] = df["human_review_status"].replace({
        "ACCEPT":"VALID_SAMPLE",
        "REJECT":"INVALID_SAMPLE",
        "REVIEW":"NEEDS_REVIEW",
    })
    return df

st.set_page_config(page_title="V2B Data Curation", page_icon="🧹", layout="wide")
st.title("🧹 V2B — Data Cleaning & Curation")
st.caption("Validade da amostra ≠ Elegibilidade para o benchmark principal")
st.info("Marque VALID_SAMPLE quando a imagem for válida. A ferramenta calcula depois ELIGIBLE, CONTEXT_DOMAIN ou outro status.")
st.warning("Não use predições dos modelos durante a curadoria. Resolução é apenas quality flag.")

if not MANIFEST.exists():
    st.error("Manifest não encontrado.")
    st.stop()

cfg = json.loads(CONFIG.read_text(encoding="utf-8"))

@st.cache_resource
def get_dataset():
    return load_dataset(cfg["dataset"], split="train")

ds = get_dataset()
df = migrate(pd.read_csv(MANIFEST).fillna(""))
df["final_status"] = df.apply(classify_final_status, axis=1)
df.to_csv(MANIFEST, index=False)

classes = sorted(df["mapped_class_id"].unique())
class_id = st.sidebar.selectbox(
    "Classe",
    classes,
    format_func=lambda x: f"{x} — {df[df.mapped_class_id==x].mapped_class_name.iloc[0]}"
)
status_filter = st.sidebar.selectbox(
    "Status final",
    ["ALL","CANDIDATE","ELIGIBLE","CONTEXT_DOMAIN","VALID_OUTSIDE_MAIN_DOMAIN","REVIEW","EXCLUDED"]
)
page_size = st.sidebar.selectbox("Por página", [5,10,20], index=1)

view = df[df.mapped_class_id == class_id].copy()
if status_filter != "ALL":
    view = view[view.final_status == status_filter]

page_max = max(1, (len(view)+page_size-1)//page_size)
page = st.number_input("Página", 1, page_max, 1)
view = view.iloc[(page-1)*page_size:page*page_size]

class_df = df[df.mapped_class_id == class_id]
m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Candidates", len(class_df))
m2.metric("Valid samples", int((class_df.human_review_status=="VALID_SAMPLE").sum()))
m3.metric("Eligible V2B-1.0", int((class_df.final_status=="ELIGIBLE").sum()))
m4.metric("Context domain", int((class_df.final_status=="CONTEXT_DOMAIN").sum()))
m5.metric("Excluded", int((class_df.final_status=="EXCLUDED").sum()))

for row_idx, rec in view.iterrows():
    image = ds[int(rec.dataset_index)]["image"].convert("RGB")
    st.divider()
    c1,c2,c3 = st.columns([1.15,0.95,1.25])

    with c1:
        st.image(image, use_container_width=True)

    with c2:
        st.markdown(f"**{rec.mapped_class_id} — {rec.mapped_class_name}**")
        st.caption(f"articleType: `{rec.original_article_type}`")
        st.write(rec.product_display_name)
        st.write(f"Integridade: **{rec.file_integrity}**")
        st.write(f"Duplicata exata: **{rec.exact_duplicate_status}**")
        st.write(f"Resolução: {rec.width}×{rec.height}")
        st.write(f"Resolution status: **{rec.resolution_status}**")

        status = classify_final_status(rec)
        st.markdown("### Status")
        if status == "ELIGIBLE":
            st.success("ELIGIBLE — V2B-1.0")
        elif status == "CONTEXT_DOMAIN":
            st.info("CONTEXT_DOMAIN — válido, fora do benchmark principal")
        elif status == "EXCLUDED":
            st.error("EXCLUDED")
        elif status == "REVIEW":
            st.warning("REVIEW")
        else:
            st.caption(status)

    with c3:
        sem = st.selectbox("Semântica", SEMANTIC, index=idx(SEMANTIC, rec.semantic_status), key=f"sem_{row_idx}")
        pres = st.selectbox("Presentation type", PRESENTATION, index=idx(PRESENTATION, rec.presentation_type), key=f"pres_{row_idx}")
        hum = st.selectbox("Human present", HUMAN, index=idx(HUMAN, rec.human_present), key=f"hum_{row_idx}")
        comp = st.selectbox(
            "Composição",
            COMPOSITION,
            index=idx(COMPOSITION, rec.composition_status),
            key=f"comp_{row_idx}",
            help="Em WORN_BY_PERSON, a própria pessoa não é SECONDARY_OBJECT."
        )
        dec = st.selectbox(
            "Validade da amostra",
            DECISION,
            index=idx(DECISION, rec.human_review_status),
            key=f"dec_{row_idx}",
            help="VALID_SAMPLE significa dado válido; a elegibilidade V2B é calculada separadamente."
        )
        notes = st.text_area("Notas", value=rec.review_notes, key=f"notes_{row_idx}", height=80)

        preview = rec.copy()
        preview["semantic_status"] = sem
        preview["presentation_type"] = pres
        preview["human_present"] = hum
        preview["composition_status"] = comp
        preview["human_review_status"] = dec
        predicted = classify_final_status(preview)
        st.caption(f"Resultado previsto: **{predicted}**")

        if pres == "WORN_BY_PERSON" and hum == "YES":
            st.caption("A pessoa faz parte da apresentação e não deve ser marcada como SECONDARY_OBJECT apenas por estar presente.")

        if st.button("Salvar avaliação", key=f"save_{row_idx}", use_container_width=True):
            df.loc[row_idx, ["semantic_status","presentation_type","human_present","composition_status","human_review_status","review_notes"]] = [sem,pres,hum,comp,dec,notes]
            df.loc[row_idx, "final_status"] = classify_final_status(df.loc[row_idx])
            df.to_csv(MANIFEST, index=False)
            st.success(f"Salvo: {df.loc[row_idx, 'final_status']}")

st.divider()
st.subheader("Resumo por classe")
summary = df.groupby(["mapped_class_id","mapped_class_name"]).agg(
    candidates=("dataset_index","size"),
    valid_samples=("human_review_status", lambda s: int((s=="VALID_SAMPLE").sum())),
    eligible=("final_status", lambda s: int((s=="ELIGIBLE").sum())),
    context_domain=("final_status", lambda s: int((s=="CONTEXT_DOMAIN").sum())),
    outside_main=("final_status", lambda s: int((s=="VALID_OUTSIDE_MAIN_DOMAIN").sum())),
    review=("final_status", lambda s: int((s=="REVIEW").sum())),
    excluded=("final_status", lambda s: int((s=="EXCLUDED").sum())),
).reset_index()

st.dataframe(summary, hide_index=True, use_container_width=True)
st.download_button(
    "⬇️ Exportar manifest",
    df.to_csv(index=False).encode("utf-8"),
    "v2b_candidate_manifest_curated.csv",
    "text/csv",
    use_container_width=True,
)
