from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "artifacts"

BENCHMARK_PATH = ARTIFACTS / "benchmark.csv"
ROBUSTNESS_PATH = ARTIFACTS / "robustness" / "robustness_summary.csv"
DOMAIN_SHIFT_PATH = ARTIFACTS / "domain_shift" / "domain_shift_summary.csv"
V2B_RAW_PATH = ARTIFACTS / "domain_shift_v2b" / "raw" / "v2b_raw_summary.csv"
V2B_V2_PATH = ARTIFACTS / "domain_shift_v2b" / "canonical_v2" / "v2b_canonical_v2_summary.csv"


@st.cache_data
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _model_value(df: pd.DataFrame, model: str, column: str) -> float | None:
    if df.empty or "model" not in df.columns or column not in df.columns:
        return None
    rows = df[df["model"].astype(str).str.lower() == model.lower()]
    if rows.empty:
        return None
    return float(rows.iloc[0][column])


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


@st.cache_data
def _load_pitch_metrics() -> dict[str, float | None]:
    benchmark = _read_csv(BENCHMARK_PATH)
    robustness = _read_csv(ROBUSTNESS_PATH)
    domain_shift = _read_csv(DOMAIN_SHIFT_PATH)
    raw = _read_csv(V2B_RAW_PATH)
    v2 = _read_csv(V2B_V2_PATH)

    return {
        "clean": _model_value(benchmark, "cnn", "accuracy"),
        "robustness": _model_value(robustness, "cnn", "mean_corrupted_accuracy"),
        "domain_shift": _model_value(domain_shift, "cnn", "mean_shifted_accuracy"),
        "v2b_raw": _model_value(raw, "cnn", "accuracy"),
        "v2b_v2": _model_value(v2, "cnn", "accuracy"),
        "v2b_v2_f1": _model_value(v2, "cnn", "f1_macro"),
    }


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .overview-hero {
            padding: 1.55rem 1.65rem;
            border: 1px solid rgba(128,128,128,.24);
            border-radius: 18px;
            margin-bottom: 1.1rem;
            background: linear-gradient(135deg, rgba(125,125,125,.08), rgba(125,125,125,.02));
        }
        .overview-kicker {
            font-size: .78rem;
            letter-spacing: .12em;
            text-transform: uppercase;
            opacity: .7;
            margin-bottom: .35rem;
        }
        .overview-title {
            font-size: 2.1rem;
            line-height: 1.08;
            font-weight: 760;
            margin: 0 0 .35rem 0;
        }
        .overview-subtitle {
            font-size: 1.04rem;
            opacity: .78;
            max-width: 900px;
            margin-bottom: .9rem;
        }
        .badge-row {display:flex; flex-wrap:wrap; gap:.42rem; margin-top:.45rem;}
        .tech-badge, .status-badge {
            display:inline-flex;
            align-items:center;
            gap:.3rem;
            border:1px solid rgba(128,128,128,.28);
            border-radius:999px;
            padding:.25rem .55rem;
            font-size:.76rem;
            font-weight:650;
            background:rgba(128,128,128,.08);
        }
        .status-badge {font-size:.72rem; letter-spacing:.02em;}
        .pipeline-band {
            border:1px solid rgba(128,128,128,.22);
            border-radius:16px;
            padding:.8rem 1rem;
            margin:.45rem 0 .75rem 0;
            text-align:center;
            background:rgba(128,128,128,.05);
        }
        .pipeline-grid {
            display:grid;
            grid-template-columns:repeat(5,minmax(115px,1fr));
            gap:.55rem;
            align-items:stretch;
        }
        .pipe-card {
            border:1px solid rgba(128,128,128,.22);
            border-radius:14px;
            padding:.7rem .55rem;
            min-height:86px;
            display:flex;
            flex-direction:column;
            justify-content:center;
            text-align:center;
            background:rgba(128,128,128,.035);
        }
        .pipe-card b {font-size:.86rem;}
        .pipe-card span {font-size:.72rem; opacity:.68; margin-top:.2rem;}
        .gate-card {
            border:1.5px solid rgba(46, 160, 67, .5);
            background:rgba(46,160,67,.07);
        }
        .current-card {
            border:1.5px solid rgba(31,111,235,.55);
            background:rgba(31,111,235,.07);
        }
        .thesis-card {
            border:1px solid rgba(128,128,128,.25);
            border-radius:16px;
            padding:1rem 1.1rem;
            text-align:center;
            margin:.7rem 0 1rem 0;
        }
        .thesis-card .headline {font-size:1.25rem; font-weight:760;}
        .thesis-card .sub {opacity:.72; margin-top:.2rem;}
        .role-card {
            border:1px solid rgba(128,128,128,.22);
            border-radius:14px;
            padding:.85rem .9rem;
            min-height:138px;
        }
        .role-card .role-title {font-weight:760; margin-bottom:.3rem;}
        .role-card .role-copy {font-size:.85rem; opacity:.76;}
        .role-card .role-quote {font-size:.79rem; margin-top:.65rem; font-style:italic; opacity:.85;}
        .phase-line {
            display:grid;
            grid-template-columns:1.4fr .8fr;
            gap:.5rem;
            padding:.4rem 0;
            border-bottom:1px solid rgba(128,128,128,.13);
            font-size:.87rem;
        }
        .assistant-card {
            border:1px dashed rgba(128,128,128,.38);
            border-radius:16px;
            padding:1rem 1.1rem;
            margin-top:.6rem;
            background:rgba(128,128,128,.03);
        }
        @media (max-width: 900px) {
            .pipeline-grid {grid-template-columns:repeat(2,minmax(120px,1fr));}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    st.markdown(
        """
        <div class="overview-hero">
          <div class="overview-kicker">ML Engineering • Reliability Study</div>
          <div class="overview-title">Fashion-MNIST Reliability Lab</div>
          <div class="overview-subtitle">
            <b>From Benchmark Accuracy to Production Readiness.</b><br>
            Um estudo end-to-end sobre robustez, domain shift, representação visual,
            confiabilidade e evolução controlada de modelos de Machine Learning.
          </div>
          <div class="badge-row">
            <span class="tech-badge">🐍 Python</span>
            <span class="tech-badge">🧠 scikit-learn</span>
            <span class="tech-badge">◫ TensorFlow</span>
            <span class="tech-badge">👁 OpenCV</span>
            <span class="tech-badge">🧪 MLflow</span>
            <span class="tech-badge">⚡ FastAPI</span>
            <span class="tech-badge">📊 Streamlit</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_pipeline() -> None:
    st.subheader("End-to-End ML Engineering Pipeline")
    st.markdown(
        """
        <div class="pipeline-band">
          <b>🧪 MLFLOW — EXPERIMENT TRACKING & EVIDENCE LAYER</b><br>
          <span style="opacity:.7;font-size:.82rem">runs • metrics • artifacts • models • traceability</span>
        </div>
        <div class="pipeline-grid">
          <div class="pipe-card"><b>1 · DATA</b><span>Fashion-MNIST + external domains</span></div>
          <div class="pipe-card"><b>2 · EDA</b><span>quality • classes • distributions</span></div>
          <div class="pipe-card"><b>3 · MODEL SELECTION</b><span>SVM × MLP × CNN</span></div>
          <div class="pipe-card"><b>4 · CLEAN BENCHMARK</b><span>baseline performance</span></div>
          <div class="pipe-card"><b>5 · ROBUSTNESS</b><span>corruptions • stress</span></div>
          <div class="pipe-card"><b>6 · DOMAIN SHIFT</b><span>controlled distribution change</span></div>
          <div class="pipe-card"><b>7 · EXTERNAL VALIDATION</b><span>V2B frozen domain</span></div>
          <div class="pipe-card gate-card"><b>8 · MODEL SELECTION GATE ✓</b><span>CNN selected • DG-MODEL-001</span></div>
          <div class="pipe-card current-card"><b>9 · EDA ENRICHED ◉</b><span>ML → DL → CV • current phase</span></div>
          <div class="pipe-card"><b>10 · RELIABILITY & ADAPTATION</b><span>evidence-driven intervention</span></div>
          <div class="pipe-card"><b>11 · POST-ADAPTATION</b><span>retention • external gain</span></div>
          <div class="pipe-card"><b>12 · PRODUCTION READINESS</b><span>performance • reliability • traceability</span></div>
          <div class="pipe-card"><b>13 · FASTAPI</b><span>model serving</span></div>
          <div class="pipe-card"><b>14 · STREAMLIT</b><span>ML storytelling</span></div>
          <div class="pipe-card"><b>15 · MONITORING</b><span>feedback → data</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "MLflow acompanha o ciclo experimental como camada transversal; FastAPI representa a operacionalização do modelo e Streamlit comunica a narrativa e as evidências."
    )


def _render_thesis(metrics: dict[str, float | None]) -> None:
    clean = metrics["clean"]
    external = metrics["v2b_v2"]
    gap = None if clean is None or external is None else clean - external

    st.markdown(
        f"""
        <div class="thesis-card">
          <div class="headline">Clean Accuracy ≠ Production Readiness</div>
          <div class="sub">
            A CNN atingiu <b>{_pct(clean)}</b> no domínio clean, mas apenas <b>{_pct(external)}</b>
            no V2B com Canonicalizer V2{f' — um gap de <b>{gap * 100:.2f} p.p.</b>' if gap is not None else ''}.
            O benchmark foi o ponto de partida, não a conclusão.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Clean Accuracy", _pct(metrics["clean"]))
    c2.metric("Robustness", _pct(metrics["robustness"]))
    c3.metric("Controlled Shift", _pct(metrics["domain_shift"]))
    c4.metric("V2B RAW", _pct(metrics["v2b_raw"]))
    c5.metric("V2B + V2", _pct(metrics["v2b_v2"]))



def _render_current_state() -> None:
    st.subheader("Where We Are Now")
    left, right = st.columns([1.15, 1])

    with left:
        rows = [
            ("Data & baseline models", "COMPLETE"),
            ("Clean Benchmark", "VALIDATED"),
            ("Robustness V2", "VALIDATED / FROZEN"),
            ("Controlled Domain Shift V1", "VALIDATED / FROZEN"),
            ("V2B External Validation", "VALIDATED / FROZEN"),
            ("Model Selection Gate", "CNN SELECTED"),
            ("EDA Enriched", "CURRENT"),
            ("Reliability & Domain Adaptation", "NEXT"),
            ("Production Readiness", "PENDING"),
        ]
        html = "".join(
            f'<div class="phase-line"><span>{name}</span><span><span class="status-badge">{status}</span></span></div>'
            for name, status in rows
        )
        st.markdown(html, unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="role-card">
              <div class="role-title">🧠 CNN — Primary Experimental Model</div>
              <div class="role-copy">
                Selecionada após Clean Benchmark, Robustness, Controlled Domain Shift e V2B.
                SVM e MLP permanecem como comparative baselines.
              </div>
              <div class="badge-row">
                <span class="status-badge">SELECTED</span>
                <span class="status-badge">DG-MODEL-001</span>
                <span class="status-badge">NOT PRODUCTION READY</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )



def _render_architecture_roles() -> None:
    st.subheader("Architecture Roles")
    c1, c2, c3, c4 = st.columns(4)

    cards = [
        (c1, "🧪 MLflow", "Experimental Evidence", "Runs, metrics, artifacts, models and reproducibility.", "MLflow preserves the evidence."),
        (c2, "⚡ FastAPI", "Model Serving", "Exposes inference through a production-oriented API contract.", "FastAPI operationalizes the model."),
        (c3, "📊 Streamlit", "ML Storytelling", "Explains experiments, trade-offs, decisions and current project state.", "Streamlit tells the story."),
        (c4, "📐 SDD", "Engineering Reasoning", "Preserves scope, criteria, gates, assumptions and decision history.", "SDD preserves the reasoning."),
    ]

    for col, title, role, copy, quote in cards:
        with col:
            st.markdown(
                f"""
                <div class="role-card">
                  <div class="role-title">{title}</div>
                  <div class="status-badge">{role}</div>
                  <div class="role-copy" style="margin-top:.55rem">{copy}</div>
                  <div class="role-quote">{quote}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )



def _render_ai_assistant() -> None:
    st.subheader("AI Experiment Assistant")
    st.markdown(
        """
        <div class="assistant-card">
          <div class="badge-row" style="margin-top:0">
            <span class="status-badge">PLANNED</span>
            <span class="status-badge">EVIDENCE-GROUNDED</span>
            <span class="status-badge">DATA SCIENCE • ML • DL • CV</span>
          </div>
          <div style="margin-top:.65rem;font-weight:720">Transversal intelligence layer</div>
          <div style="font-size:.88rem;opacity:.76;margin-top:.25rem">
            O assistente deverá conectar dados, métricas, MLflow, EDA Enriched, probability rankings,
            sibling classes e resultados de robustez/domain shift para produzir insights rastreáveis.
            Ele interpreta evidências; não substitui métricas, gates ou decisão humana.
          </div>
          <div style="margin-top:.65rem;font-size:.83rem"><b>Evidence → AI Insight → Human Decision</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_pitch_close() -> None:
    st.subheader("Project Pitch")
    st.info(
        "This project starts where traditional ML benchmarks usually end: it investigates whether a high-performing image classifier remains reliable under corruption, domain shift and real external data, then uses Deep Learning and Computer Vision evidence to guide adaptation toward production readiness."
    )


def render_overview() -> None:
    """Executive pitch and lifecycle overview for the reliability study."""
    _inject_styles()
    metrics = _load_pitch_metrics()

    _render_hero()
    _render_pipeline()
    _render_thesis(metrics)
    st.divider()
    _render_current_state()
    st.divider()
    _render_architecture_roles()
    st.divider()
    _render_ai_assistant()
    st.divider()
    _render_pitch_close()
