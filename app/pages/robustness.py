from pathlib import Path

import pandas as pd
import streamlit as st

from config.settings import ARTIFACTS_DIR


# ============================================================
# CONFIGURATION
# ============================================================

ROBUSTNESS_DIR = Path(ARTIFACTS_DIR) / "robustness"

RESULTS_FILE = ROBUSTNESS_DIR / "robustness_results.csv"
SUMMARY_FILE = ROBUSTNESS_DIR / "robustness_summary.csv"
CORRUPTION_SUMMARY_FILE = ROBUSTNESS_DIR / "corruption_summary.csv"


CORRUPTIONS = [
    "noise",
    "rotation",
    "blur",
    "contrast",
    "brightness",
    "occlusion",
]


# ============================================================
# DATA ACCESS
# ============================================================

@st.cache_data
def load_robustness_artifacts():
    """
    Carrega os artefatos congelados do Robustness Benchmark V2.

    O Streamlit não executa novamente os testes de corrupção.
    Ele apenas apresenta e interpreta as evidências produzidas
    pela camada experimental.
    """

    results = pd.read_csv(RESULTS_FILE)
    summary = pd.read_csv(SUMMARY_FILE)
    corruption_summary = pd.read_csv(CORRUPTION_SUMMARY_FILE)

    required_results = {
        "model",
        "corruption",
        "severity",
        "accuracy",
    }

    required_summary = {
        "model",
        "mean_corrupted_accuracy",
        "min_corrupted_accuracy",
        "mean_corrupted_f1",
        "mean_accuracy_drop",
        "max_accuracy_drop",
    }

    required_corruption_summary = {
        "model",
        "corruption",
        "mean_accuracy",
        "min_accuracy",
        "mean_f1",
        "mean_accuracy_drop",
        "max_accuracy_drop",
    }

    _validate_columns(
        results,
        required_results,
        "robustness_results.csv",
    )

    _validate_columns(
        summary,
        required_summary,
        "robustness_summary.csv",
    )

    _validate_columns(
        corruption_summary,
        required_corruption_summary,
        "corruption_summary.csv",
    )

    results = _normalize_model_names(results)
    summary = _normalize_model_names(summary)
    corruption_summary = _normalize_model_names(
        corruption_summary
    )

    return results, summary, corruption_summary


def _validate_columns(
    df: pd.DataFrame,
    required_columns: set,
    artifact_name: str,
) -> None:
    """
    Valida se o artefato contém as colunas mínimas esperadas.
    """

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{artifact_name} não possui as colunas esperadas: "
            + ", ".join(sorted(missing))
        )


def _normalize_model_names(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normaliza os nomes dos modelos apenas para apresentação.
    """

    df = df.copy()

    df["model"] = (
        df["model"]
        .astype(str)
        .str.upper()
    )

    return df


# ============================================================
# SUMMARY
# ============================================================

def _get_robustness_summary(
    summary: pd.DataFrame,
) -> dict:
    """
    Obtém o modelo com melhor Robustness média.
    """

    best_row = summary.loc[
        summary["mean_corrupted_accuracy"].idxmax()
    ]

    return {
        "best_row": best_row,
        "best_model": best_row["model"],
    }


# ============================================================
# EXPERIMENT CONTEXT
# ============================================================

def _render_experiment_context() -> None:
    """
    Conecta o Clean Benchmark ao Robustness Benchmark.
    """

    st.subheader("Experiment Context")

    st.markdown(
        """
        O **Clean Benchmark** identificou o modelo com melhor
        desempenho no domínio original do Fashion-MNIST.

        Entretanto, um modelo pode apresentar excelente performance
        no conjunto de teste e ainda ser sensível a pequenas
        alterações na entrada.

        Nesta etapa, os modelos são submetidos a **corrupções
        controladas**, permitindo medir quanto seu desempenho se
        degrada quando a representação visual é perturbada.
        """
    )

    st.info(
        "Pergunta desta etapa: o desempenho observado no Clean "
        "Benchmark permanece estável quando a entrada sofre "
        "perturbações controladas?"
    )


# ============================================================
# SUMMARY CARDS
# ============================================================

def _render_summary_cards(
    robustness_summary: dict,
) -> None:
    """
    Exibe as principais evidências globais de Robustness.
    """

    best_row = robustness_summary["best_row"]
    best_model = robustness_summary["best_model"]

    st.subheader("Robustness Summary")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Best Robust Model",
            best_model,
        )

    with c2:
        st.metric(
            "Mean Corrupted Accuracy",
            f"{best_row['mean_corrupted_accuracy']:.2%}",
        )

    with c3:
        st.metric(
            "Worst Case",
            f"{best_row['min_corrupted_accuracy']:.2%}",
        )

    with c4:
        st.metric(
            "Mean Accuracy Drop",
            f"{best_row['mean_accuracy_drop']:.2%}",
        )


# ============================================================
# CLEAN VS CORRUPTED
# ============================================================

def _render_clean_vs_corrupted(
    results: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """
    Compara desempenho no domínio clean com desempenho médio
    sob corrupções.
    """

    st.subheader("Clean vs Corrupted Performance")

    clean = (
        results[
            results["corruption"] == "clean"
        ][
            [
                "model",
                "accuracy",
            ]
        ]
        .drop_duplicates("model")
        .rename(
            columns={
                "accuracy": "Clean Accuracy",
            }
        )
    )

    corrupted = (
        summary[
            [
                "model",
                "mean_corrupted_accuracy",
            ]
        ]
        .rename(
            columns={
                "mean_corrupted_accuracy":
                    "Mean Corrupted Accuracy",
            }
        )
    )

    comparison = clean.merge(
        corrupted,
        on="model",
    )

    chart_df = comparison.set_index(
        "model"
    )

    st.bar_chart(
        chart_df,
        stack=False,
        use_container_width=True,
    )

    st.caption(
        "Comparação entre a performance no conjunto limpo "
        "e a média obtida sob corrupções controladas."
    )


# ============================================================
# MODEL ROBUSTNESS TABLE
# ============================================================

def _render_model_robustness(
    summary: pd.DataFrame,
) -> None:
    """
    Exibe os principais indicadores de Robustness por modelo.
    """

    st.subheader("Model Robustness")

    display_df = summary[
        [
            "model",
            "mean_corrupted_accuracy",
            "min_corrupted_accuracy",
            "mean_corrupted_f1",
            "mean_accuracy_drop",
            "max_accuracy_drop",
        ]
    ].copy()

    display_df.columns = [
        "Model",
        "Mean Accuracy",
        "Worst Accuracy",
        "Mean F1",
        "Mean Drop",
        "Max Drop",
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Mean Accuracy":
                st.column_config.ProgressColumn(
                    "Mean Accuracy",
                    format="%.4f",
                    min_value=0.0,
                    max_value=1.0,
                ),

            "Worst Accuracy":
                st.column_config.NumberColumn(
                    "Worst Accuracy",
                    format="%.4f",
                ),

            "Mean F1":
                st.column_config.ProgressColumn(
                    "Mean F1",
                    format="%.4f",
                    min_value=0.0,
                    max_value=1.0,
                ),

            "Mean Drop":
                st.column_config.NumberColumn(
                    "Mean Drop",
                    format="%.4f",
                ),

            "Max Drop":
                st.column_config.NumberColumn(
                    "Max Drop",
                    format="%.4f",
                ),
        },
    )


# ============================================================
# CORRUPTION ANALYSIS
# ============================================================

def _render_corruption_analysis(
    results: pd.DataFrame,
    corruption_summary: pd.DataFrame,
) -> None:
    """
    Permite investigar cada tipo de corrupção separadamente.
    """

    st.subheader("Performance by Corruption")

    corruption = st.selectbox(
        "Corruption",
        CORRUPTIONS,
    )

    selected = results[
        results["corruption"] == corruption
    ].copy()

    if selected.empty:
        st.warning(
            f"Não existem resultados para a corrupção '{corruption}'."
        )
        return

    curve = (
        selected
        .pivot(
            index="severity",
            columns="model",
            values="accuracy",
        )
        .sort_index()
    )

    st.line_chart(
        curve,
        use_container_width=True,
    )

    st.caption(
        "Accuracy em função da severidade da corrupção."
    )

    corruption_detail = (
        corruption_summary[
            corruption_summary["corruption"] == corruption
        ][
            [
                "model",
                "mean_accuracy",
                "min_accuracy",
                "mean_f1",
                "mean_accuracy_drop",
                "max_accuracy_drop",
            ]
        ]
        .copy()
    )

    corruption_detail.columns = [
        "Model",
        "Mean Accuracy",
        "Worst Accuracy",
        "Mean F1",
        "Mean Drop",
        "Max Drop",
    ]

    st.dataframe(
        corruption_detail,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Mean Accuracy":
                st.column_config.NumberColumn(
                    format="%.4f",
                ),

            "Worst Accuracy":
                st.column_config.NumberColumn(
                    format="%.4f",
                ),

            "Mean F1":
                st.column_config.NumberColumn(
                    format="%.4f",
                ),

            "Mean Drop":
                st.column_config.NumberColumn(
                    format="%.4f",
                ),

            "Max Drop":
                st.column_config.NumberColumn(
                    format="%.4f",
                ),
        },
    )


# ============================================================
# ACCURACY DEGRADATION
# ============================================================

def _render_accuracy_degradation(
    corruption_summary: pd.DataFrame,
) -> None:
    """
    Resume a degradação média por tipo de corrupção.
    """

    st.subheader("Mean Accuracy Drop")

    drop_chart = corruption_summary.pivot(
        index="corruption",
        columns="model",
        values="mean_accuracy_drop",
    )

    st.bar_chart(
        drop_chart,
        stack=False,
        use_container_width=True,
    )

    st.caption(
        "Quanto maior o valor, maior foi a perda média de "
        "Accuracy em relação ao desempenho clean."
    )


# ============================================================
# ROBUSTNESS INSIGHT
# ============================================================

def _render_robustness_insight(
    robustness_summary: dict,
    corruption_summary: pd.DataFrame,
) -> None:
    """
    Interpreta as principais evidências do experimento.
    """

    best_row = robustness_summary["best_row"]
    best_model = robustness_summary["best_model"]

    st.subheader("Robustness Insight")

    st.success(
        f"{best_model} apresenta a melhor Robustness global, "
        f"mantendo aproximadamente "
        f"{best_row['mean_corrupted_accuracy']:.2%} de "
        f"Accuracy média sob corrupções."
    )

    st.markdown(
        """
        Os resultados mostram que desempenho elevado no
        **Clean Benchmark** não implica necessariamente estabilidade
        quando a entrada sofre alterações.

        Diferentes arquiteturas apresentam perfis distintos de
        sensibilidade, indicando que Robustness é uma dimensão
        complementar à Accuracy tradicional.
        """
    )

    _render_corruption_findings(
        corruption_summary
    )

    st.warning(
        "Esses experimentos representam corruption robustness / "
        "covariate shift. Eles não constituem, isoladamente, "
        "detecção formal de Out-of-Distribution (OOD)."
    )


def _render_corruption_findings(
    corruption_summary: pd.DataFrame,
) -> None:
    """
    Gera alguns achados diretamente dos artefatos,
    evitando hardcode excessivo.
    """

    rows = []

    for corruption in CORRUPTIONS:
        subset = corruption_summary[
            corruption_summary["corruption"] == corruption
        ]

        if subset.empty:
            continue

        best = subset.loc[
            subset["mean_accuracy"].idxmax()
        ]

        rows.append(
            {
                "Corruption": corruption.title(),
                "Best Model": best["model"],
                "Mean Accuracy": best["mean_accuracy"],
            }
        )

    if not rows:
        return

    findings_df = pd.DataFrame(rows)

    st.markdown(
        "**Best model by corruption type**"
    )

    st.dataframe(
        findings_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Mean Accuracy":
                st.column_config.ProgressColumn(
                    "Mean Accuracy",
                    format="%.4f",
                    min_value=0.0,
                    max_value=1.0,
                ),
        },
    )


# ============================================================
# EVIDENCE LAYER
# ============================================================

def _render_evidence_layer() -> None:
    """
    Relaciona o experimento à camada de rastreabilidade.
    """

    st.subheader("Experiment Evidence")

    st.markdown(
        """
        O **Robustness Benchmark V2** representa uma etapa adicional
        da avaliação experimental.

        Enquanto o Streamlit apresenta as conclusões de forma
        narrativa, o MLflow pode atuar como **camada de evidências**
        para registrar os testes, métricas, parâmetros e artefatos
        associados a cada etapa de avaliação.
        """
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Models",
        "3",
    )

    c2.metric(
        "Corruptions",
        "6",
    )

    c3.metric(
        "Experiment Stage",
        "Robustness V2",
    )

    c4.metric(
        "Status",
        "Validated",
    )

    st.info(
        "Streamlit conta a história. "
        "MLflow preserva as evidências."
    )


# ============================================================
# NEXT EXPERIMENT
# ============================================================

def _render_next_experiment() -> None:
    """
    Conecta Robustness ao capítulo seguinte da narrativa.
    """

    st.subheader("Next Experiment")

    st.markdown(
        """
        O Robustness Benchmark mostrou que os modelos apresentam
        diferentes níveis de estabilidade diante de perturbações
        controladas.

        A próxima questão amplia o problema:

        **O que acontece quando a própria distribuição visual dos
        dados se afasta do domínio original?**

        **Próxima etapa → Domain Shift**
        """
    )


# ============================================================
# PAGE
# ============================================================

def render_robustness() -> None:
    """
    Renderiza a página modular do Robustness Benchmark V2.
    """

    st.header("🧪 Robustness")

    st.caption(
        "Avaliação da estabilidade de SVM, MLP e CNN "
        "sob corrupções controladas do Fashion-MNIST."
    )

    try:
        (
            results,
            summary,
            corruption_summary,
        ) = load_robustness_artifacts()

        robustness_summary = _get_robustness_summary(
            summary
        )

        _render_experiment_context()

        st.divider()

        _render_summary_cards(
            robustness_summary
        )

        st.divider()

        _render_clean_vs_corrupted(
            results,
            summary,
        )

        st.divider()

        _render_model_robustness(
            summary
        )

        st.divider()

        _render_corruption_analysis(
            results,
            corruption_summary,
        )

        st.divider()

        _render_accuracy_degradation(
            corruption_summary
        )

        st.divider()

        _render_robustness_insight(
            robustness_summary,
            corruption_summary,
        )

        st.divider()

        _render_evidence_layer()

        st.divider()

        _render_next_experiment()

    except FileNotFoundError as exc:
        st.warning(
            "Um ou mais artefatos do Robustness Benchmark V2 "
            "não foram encontrados."
        )

        st.code(
            "python -m src.robustness.stress_test",
            language="powershell",
        )

        st.caption(str(exc))

    except ValueError as exc:
        st.error(
            "Um dos artefatos de Robustness possui "
            "estrutura incompatível."
        )

        st.code(
            str(exc)
        )

    except Exception as exc:
        st.error(
            "Não foi possível carregar os resultados "
            "do Robustness Benchmark."
        )

        st.exception(exc)