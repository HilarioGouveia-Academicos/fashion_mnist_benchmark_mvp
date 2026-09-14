
from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


st.set_page_config(
    page_title="Fashion-MNIST Reliability Study",
    page_icon="👕",
    layout="wide",
)


from app.pages.overview import render_overview
from app.pages.dataset import render_dataset
from app.pages.model_selection import render_model_selection
from app.pages.model_selection_gate import render_model_selection_gate
from app.pages.benchmark import render_benchmark
from app.pages.robustness import render_robustness
from app.pages.domain_shift import render_domain_shift
from app.pages.inference import render_inference
from app.pages.experiment_evidence import render_experiment_evidence
from app.pages.cv_capture_lab import render_cv_capture_lab

API_URL = "http://localhost:8000"


st.title("👕 Fashion-MNIST Reliability Study")

st.caption(
    "SVM × MLP × CNN | MLflow | FastAPI | Streamlit"
)


overview_page = st.Page(
    render_overview,
    title="Overview",
    icon="🏠",
    url_path="overview",
    default=True,
)

dataset_page = st.Page(
    render_dataset,
    title="Dataset",
    icon="📦",
    url_path="dataset",
)

model_selection_page = st.Page(
    render_model_selection,
    title="Model Selection",
    icon="🧠",
    url_path="model-selection",
)

benchmark_page = st.Page(
    render_benchmark,
    title="Benchmark",
    icon="🏆",
    url_path="benchmark",
)

robustness_page = st.Page(
    render_robustness,
    title="Robustness",
    icon="🧪",
    url_path="robustness",
)

domain_shift_page = st.Page(
    render_domain_shift,
    title="Domain Shift",
    icon="🌐",
    url_path="domain-shift",
)

model_selection_gate_page = st.Page(
    render_model_selection_gate,
    title="Model Selection Gate",
    icon="🚦",
    url_path="model-selection-gate",
)

inference_page = st.Page(
    render_inference,
    title="Inference",
    icon="🔮",
    url_path="inference",
)

cv_capture_lab_page = st.Page(
    render_cv_capture_lab,
    title="CV Capture Lab",
    icon="📷",
    url_path="cv-capture-lab",
)

experiment_evidence_page = st.Page(
    render_experiment_evidence,
    title="Experiment Evidence",
    icon="🧾",
    url_path="experiment-evidence",
)


navigation = st.navigation(
    {
        "Study": [
            overview_page,
            dataset_page,
            model_selection_page,
        ],
        "Evaluation": [
            benchmark_page,
            robustness_page,
            domain_shift_page,
            model_selection_gate_page,
        ],
        "Application": [
            inference_page,
            experiment_evidence_page,
            cv_capture_lab_page,
        ],
    },
    position="sidebar",
)


navigation.run()