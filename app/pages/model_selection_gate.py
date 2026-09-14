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
V2B_V3_PATH = ARTIFACTS / "domain_shift_v2b" / "canonical_v3" / "v2b_canonical_v3_summary.csv"
CNN_MODEL_PATH = PROJECT_ROOT / "models" / "cnn.keras"

MODEL_ORDER = ["svm", "mlp", "cnn"]
MODEL_LABELS = {"svm": "SVM", "mlp": "MLP", "cnn": "CNN"}


@st.cache_data
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _model_value(df: pd.DataFrame, model: str, column: str) -> float | None:
    if df.empty or "model" not in df.columns or column not in df.columns:
        return None
    rows = df[df["model"].astype(str).str.lower() == model]
    if rows.empty:
        return None
    return float(rows.iloc[0][column])


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


@st.cache_data
def _build_evidence_table() -> pd.DataFrame:
    benchmark = _read_csv(BENCHMARK_PATH)
    robustness = _read_csv(ROBUSTNESS_PATH)
    domain_shift = _read_csv(DOMAIN_SHIFT_PATH)
    raw = _read_csv(V2B_RAW_PATH)
    v2 = _read_csv(V2B_V2_PATH)
    v3 = _read_csv(V2B_V3_PATH)

    rows = []
    specs = [
        ("Clean Accuracy", benchmark, "accuracy"),
        ("Robustness — mean corrupted accuracy", robustness, "mean_corrupted_accuracy"),
        ("Controlled Domain Shift — mean accuracy", domain_shift, "mean_shifted_accuracy"),
        ("V2B RAW — Accuracy", raw, "accuracy"),
        ("V2B + Canonicalizer V2 — Accuracy", v2, "accuracy"),
        ("V2B + Canonicalizer V2 — Macro-F1", v2, "f1_macro"),
        ("V2B + Canonicalizer V3 — Accuracy", v3, "accuracy"),
        ("V2B + Canonicalizer V3 — Macro-F1", v3, "f1_macro"),
    ]
    for evidence, df, metric in specs:
        row = {"Evidence": evidence}
        for model in MODEL_ORDER:
            row[MODEL_LABELS[model]] = _model_value(df, model, metric)
        rows.append(row)
    return pd.DataFrame(rows)


def _render_gate_header() -> None:
    st.header("Model Selection Gate")
    st.caption(
        "Decision Gate pós-avaliação comparativa: consolida as evidências e define "
        "qual modelo avança para EDA Enriched, Reliability e Domain Adaptation."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decision ID", "DG-MODEL-001")
    c2.metric("Gate Status", "ACCEPTED / FROZEN")
    c3.metric("Selected Model", "CNN")
    c4.metric("Production Readiness", "NOT ESTABLISHED")

    st.success(
        "CNN selected as Primary Experimental Model for the Reliability & Adaptation Phase."
    )


def _render_evidence() -> None:
    st.subheader("Comparative Evidence")
    st.markdown(
        "A decisão usa somente artefatos experimentais já produzidos. "
        "A página não reexecuta treinamento nem altera resultados congelados."
    )

    evidence = _build_evidence_table()
    display = evidence.copy()
    for col in ["SVM", "MLP", "CNN"]:
        display[col] = display[col].map(_pct)

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Evidence": st.column_config.TextColumn("Evidence", width="large"),
            "SVM": st.column_config.TextColumn("SVM"),
            "MLP": st.column_config.TextColumn("MLP"),
            "CNN": st.column_config.TextColumn("CNN"),
        },
    )

    st.caption(
        "V2B é o domínio externo congelado. Canonicalizer V2 e V3 são comparados "
        "como intervenções de representação; V2 permanece superior a V3 nos resultados congelados."
    )


def _render_rationale() -> None:
    st.subheader("Selection Rationale")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            **Evidence supporting selection**

            - melhor Clean Accuracy;
            - melhor Robustness média;
            - melhor desempenho médio em Controlled Domain Shift;
            - melhor V2B RAW Accuracy;
            - melhor desempenho absoluto no V2B com Canonicalizer V2;
            - estrutura espacial preservada, adequada à investigação de representação visual.
            """
        )
    with c2:
        st.markdown(
            """
            **Roles after the Gate**

            **SVM** → Comparative Baseline  
            **MLP** → Comparative Baseline  
            **CNN** → **Primary Experimental Model**

            SVM e MLP permanecem preservados para comparação. A seleção reduz a dispersão experimental sem apagar os baselines.
            """
        )


def _prepare_model_for_visualkeras(model):
    """Materializa shapes e adiciona compatibilidade para Keras recente.

    Algumas versões do VisualKeras ainda consultam ``layer.output_shape``,
    atributo que pode não existir em camadas do Keras 3. O modelo continua
    válido; esta função apenas expõe o shape já conhecido por ``layer.output``
    para a ferramenta de documentação.
    """
    import numpy as np

    if not model.built:
        model.build((None, 28, 28, 1))

    # Uma passagem real garante que os tensores de saída estejam materializados.
    dummy_input = np.zeros((1, 28, 28, 1), dtype=np.float32)
    model(dummy_input, training=False)

    for layer in model.layers:
        if hasattr(layer, "output_shape"):
            continue
        try:
            layer.output_shape = tuple(layer.output.shape)
        except Exception:
            # Nem toda camada precisa expor output_shape para que o fallback
            # textual continue funcionando.
            continue

    return model


def _model_summary_text(model) -> str:
    """Retorna model.summary() como texto para fallback documental."""
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


def _render_selected_architecture() -> None:
    st.subheader("Selected Architecture — CNN")
    st.markdown(
        "A arquitetura selecionada preserva a estrutura espacial da entrada e passa a ser "
        "a plataforma experimental para EDA Enriched, invariância, embeddings e adaptação."
    )

    st.code(
        """Input (28×28×1)
  ↓
Conv2D (32) → MaxPooling
  ↓
Conv2D (64) → MaxPooling
  ↓
Flatten → Dense (128, ReLU) → Dropout (0.30)
  ↓
Dense (10, Softmax)""",
        language="text",
    )

    with st.expander("VisualKeras — render architecture from persisted model"):
        st.caption(
            "Opcional: a visualização é gerada a partir de models/cnn.keras. "
            "Ela documenta a arquitetura; não constitui evidência de desempenho."
        )
        if not CNN_MODEL_PATH.exists():
            st.warning("models/cnn.keras não foi encontrado.")
            return

        if not st.button("Render architecture", key="render_cnn_architecture"):
            return

        try:
            import visualkeras
            from tensorflow import keras

            model = keras.models.load_model(CNN_MODEL_PATH, compile=False)
            model = _prepare_model_for_visualkeras(model)

            try:
                image = visualkeras.layered_view(
                    model,
                    legend=True,
                    scale_xy=1,
                    scale_z=1,
                )
                st.image(
                    image,
                    caption="CNN selected architecture — VisualKeras",
                    width="stretch",
                )
            except Exception as visual_exc:
                st.warning(
                    "VisualKeras não conseguiu renderizar esta combinação de "
                    f"Keras/VisualKeras ({visual_exc}). A arquitetura carregada é "
                    "válida; exibindo model.summary() como fallback."
                )
                st.code(_model_summary_text(model), language="text")

        except ImportError:
            st.info(
                "VisualKeras/TensorFlow não estão disponíveis neste ambiente. "
                "A representação textual acima permanece válida."
            )
        except Exception as exc:
            st.warning(
                "Não foi possível carregar/preparar models/cnn.keras para a "
                f"visualização: {exc}"
            )


def _render_limitation() -> None:
    st.subheader("Selection ≠ Production Readiness")
    st.warning(
        "A CNN é o melhor candidato para continuar a investigação, mas a seleção não "
        "representa aprovação para produção."
    )

    raw = _read_csv(V2B_RAW_PATH)
    v2 = _read_csv(V2B_V2_PATH)
    clean = _read_csv(BENCHMARK_PATH)
    clean_acc = _model_value(clean, "cnn", "accuracy")
    raw_acc = _model_value(raw, "cnn", "accuracy")
    v2_acc = _model_value(v2, "cnn", "accuracy")

    c1, c2, c3 = st.columns(3)
    c1.metric("CNN Clean Accuracy", _pct(clean_acc))
    c2.metric(
        "CNN V2B RAW",
        _pct(raw_acc),
        delta=(f"-{(clean_acc - raw_acc) * 100:.2f} pp" if clean_acc is not None and raw_acc is not None else None),
        delta_color="inverse",
    )
    c3.metric(
        "CNN V2B + V2",
        _pct(v2_acc),
        delta=(f"+{(v2_acc - raw_acc) * 100:.2f} pp vs RAW" if v2_acc is not None and raw_acc is not None else None),
    )

    st.markdown(
        "Persistem **generalization gap**, sensibilidade à representação e casos de erro "
        "com alta confiança. Esses pontos passam a ser objeto explícito das próximas fases."
    )


def _render_decision_record() -> None:
    st.subheader("Decision Record")
    st.code(
        """Decision ID: DG-MODEL-001
Decision: Select CNN as primary experimental model
Selected: CNN
Retained baselines: SVM, MLP
Status: ACCEPTED / FROZEN
Next stage: EDA Enriched
Production readiness: NOT ESTABLISHED""",
        language="text",
    )

    st.markdown("**MLflow metadata contract**")
    st.code(
        """model_role = primary_experimental_model
selection_gate = DG-MODEL-001
selection_status = selected
selected_for = reliability_and_adaptation
production_ready = false""",
        language="text",
    )
    st.caption(
        "O Gate não cria uma run artificial. As runs existentes preservam as evidências; "
        "as tags registram o estado de seleção no modelo/run relevante."
    )


def _render_next_stage() -> None:
    st.subheader("Next Stage")
    st.info(
        "Model Selection Gate → CNN Selected → EDA Enriched → EDA Decision Gate → Reliability & Domain Adaptation"
    )
    st.markdown(
        "A próxima pergunta deixa de ser **‘qual modelo vence?’** e passa a ser: "
        "**‘o que a CNN aprendeu, onde ela falha e quais intervenções são justificadas pelas evidências?’**"
    )


def render_model_selection_gate() -> None:
    _render_gate_header()
    st.divider()
    _render_evidence()
    st.divider()
    _render_rationale()
    st.divider()
    _render_selected_architecture()
    st.divider()
    _render_limitation()
    st.divider()
    _render_decision_record()
    st.divider()
    _render_next_stage()
