
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
from src.v2b.manifest_guard import validate_master, validate_state

MASTER = PROJECT_ROOT / "artifacts/v2b/v2b_candidate_manifest_master.csv"
STATE = PROJECT_ROOT / "artifacts/v2b/v2b_curation_state.csv"
CONFIG = PROJECT_ROOT / "config/v2b_mapping.json"

SEMANTIC = ["UNREVIEWED","VALID","AMBIGUOUS","WRONG_LABEL"]
PRESENTATION = ["UNREVIEWED","PRODUCT_ONLY","WORN_BY_PERSON","MANNEQUIN","FLAT_LAY","OTHER"]
HUMAN = ["UNREVIEWED","NO","YES"]
COMPOSITION = ["UNREVIEWED","CLEAN","REVIEW","SECONDARY_OBJECT","FRAMING_ARTIFACT"]
DECISION = ["UNREVIEWED","VALID_SAMPLE","INVALID_SAMPLE","NEEDS_REVIEW"]
STATE_COLS = ["source_id","semantic_status","presentation_type","human_present",
              "composition_status","human_review_status","review_notes"]

def opt_index(options, value):
    return options.index(value) if value in options else 0

def load_joined():
    master = validate_master(MASTER)
    if STATE.exists():
        state = pd.read_csv(STATE, dtype=str).fillna("")
    else:
        state = pd.DataFrame({"source_id": master["source_id"]})
        for c in STATE_COLS[1:]:
            state[c] = "UNREVIEWED" if c != "review_notes" else ""
        state.to_csv(STATE, index=False)
    validate_state(state, master)
    # Structural columns come only from master; mutable columns only from state.
    base = master.drop(columns=[c for c in STATE_COLS[1:] if c in master.columns] + ["final_status"], errors="ignore")
    joined = base.merge(state[STATE_COLS], on="source_id", how="left", validate="one_to_one")
    joined = joined.fillna("")
    joined["final_status"] = joined.apply(classify_final_status, axis=1)
    return master, state, joined

st.set_page_config(page_title="V2B Curator V4", page_icon="🧹", layout="wide")
st.title("🧹 V2B Curator V4")
st.caption("Master READ-ONLY → Curation State → Joined View")
st.info("A população canônica nunca é gravada pela UI. Somente v2b_curation_state.csv é alterado.")

try:
    master, state, df = load_joined()
except Exception as exc:
    st.error(f"Integrity Guard bloqueou a inicialização: {exc}")
    st.stop()

cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
@st.cache_resource
def get_dataset():
    return load_dataset(cfg["dataset"], split="train")
ds = get_dataset()

classes = sorted(df["mapped_class_id"].unique(), key=int)
class_id = st.sidebar.selectbox("Classe", classes,
    format_func=lambda x: f"{x} — {df[df.mapped_class_id==x].mapped_class_name.iloc[0]}")
status_filter = st.sidebar.selectbox("Status final",
    ["ALL","CANDIDATE","ELIGIBLE","CONTEXT_DOMAIN","VALID_OUTSIDE_MAIN_DOMAIN","REVIEW","EXCLUDED"])
page_size = st.sidebar.selectbox("Por página", [5,10,20], index=1)

view = df[df.mapped_class_id == class_id].copy()
if status_filter != "ALL":
    view = view[view.final_status == status_filter]
page_max = max(1, (len(view)+page_size-1)//page_size)
page = st.number_input("Página", 1, page_max, 1)
view = view.iloc[(page-1)*page_size:page*page_size]

class_df = df[df.mapped_class_id == class_id]
a,b,c,d,e = st.columns(5)
a.metric("Candidates", len(class_df))
b.metric("Valid samples", int((class_df.human_review_status=="VALID_SAMPLE").sum()))
c.metric("Eligible V2B-1.0", int((class_df.final_status=="ELIGIBLE").sum()))
d.metric("Context domain", int((class_df.final_status=="CONTEXT_DOMAIN").sum()))
e.metric("Excluded", int((class_df.final_status=="EXCLUDED").sum()))

for row_idx, rec in view.iterrows():
    image = ds[int(rec.dataset_index)]["image"].convert("RGB")
    st.divider()
    c1,c2,c3 = st.columns([1.15,.95,1.25])
    with c1:
        st.image(image, use_container_width=True)
    with c2:
        st.markdown(f"**{rec.mapped_class_id} — {rec.mapped_class_name}**")
        st.caption(f"source_id: `{rec.source_id}` | articleType: `{rec.original_article_type}`")
        st.write(rec.product_display_name)
        st.write(f"Integridade: **{rec.file_integrity}**")
        st.write(f"Duplicata: **{rec.exact_duplicate_status}**")
        st.write(f"Resolução: {rec.width}×{rec.height}")
        st.write(f"Resolution: **{rec.resolution_status}**")
        status = rec.final_status
        if status == "ELIGIBLE": st.success(status)
        elif status == "CONTEXT_DOMAIN": st.info(status)
        elif status == "EXCLUDED": st.error(status)
        elif status == "REVIEW": st.warning(status)
        else: st.caption(status)
    with c3:
        sem = st.selectbox("Semântica", SEMANTIC, index=opt_index(SEMANTIC, rec.semantic_status), key=f"s_{rec.source_id}")
        pres = st.selectbox("Presentation type", PRESENTATION, index=opt_index(PRESENTATION, rec.presentation_type), key=f"p_{rec.source_id}")
        hum = st.selectbox("Human present", HUMAN, index=opt_index(HUMAN, rec.human_present), key=f"h_{rec.source_id}")
        comp = st.selectbox("Composição", COMPOSITION, index=opt_index(COMPOSITION, rec.composition_status), key=f"c_{rec.source_id}",
                            help="Em WORN_BY_PERSON, a pessoa não é SECONDARY_OBJECT apenas por estar presente.")
        dec = st.selectbox("Validade da amostra", DECISION, index=opt_index(DECISION, rec.human_review_status), key=f"d_{rec.source_id}")
        notes = st.text_area("Notas", value=rec.review_notes, key=f"n_{rec.source_id}", height=70)

        preview = rec.copy()
        preview["semantic_status"], preview["presentation_type"] = sem, pres
        preview["human_present"], preview["composition_status"] = hum, comp
        preview["human_review_status"] = dec
        st.caption(f"Resultado previsto: **{classify_final_status(preview)}**")

        if st.button("Salvar avaliação", key=f"save_{rec.source_id}", use_container_width=True):
            mask = state["source_id"] == rec.source_id
            if not mask.any():
                new = {"source_id": rec.source_id, "semantic_status": sem, "presentation_type": pres,
                       "human_present": hum, "composition_status": comp,
                       "human_review_status": dec, "review_notes": notes}
                state = pd.concat([state, pd.DataFrame([new])], ignore_index=True)
            else:
                state.loc[mask, ["semantic_status","presentation_type","human_present",
                                 "composition_status","human_review_status","review_notes"]] = \
                    [sem,pres,hum,comp,dec,notes]
            validate_state(state, master)
            state.to_csv(STATE, index=False)  # ONLY mutable file
            st.success(f"Estado salvo para source_id={rec.source_id}")

st.divider()
st.subheader("Resumo por classe")
summary = df.groupby(["mapped_class_id","mapped_class_name"]).agg(
    candidates=("source_id","size"),
    valid_samples=("human_review_status", lambda s:int((s=="VALID_SAMPLE").sum())),
    eligible=("final_status", lambda s:int((s=="ELIGIBLE").sum())),
    context_domain=("final_status", lambda s:int((s=="CONTEXT_DOMAIN").sum())),
    excluded=("final_status", lambda s:int((s=="EXCLUDED").sum())),
).reset_index()
st.dataframe(summary, hide_index=True, use_container_width=True)

# Export is a joined snapshot only; it never becomes the master automatically.
st.download_button("⬇️ Exportar snapshot curado",
                   df.to_csv(index=False).encode("utf-8"),
                   "v2b_curated_snapshot.csv","text/csv",use_container_width=True)
