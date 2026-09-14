from pathlib import Path

import pandas as pd
import streamlit as st

from config.settings import ARTIFACTS_DIR


# ============================================================
# CONFIGURATION
# ============================================================

BENCHMARK_FILE = Path(ARTIFACTS_DIR) / "benchmark.csv"

METRIC_COLUMNS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
]

REQUIRED_COLUMNS = [
    "model",
    *METRIC_COLUMNS,
]


# ============================================================
# DATA ACCESS
# ============================================================

@st.cache_data
def load_benchmark() -> pd.DataFrame:
    """
    Carrega o artefato congelado do Clean Benchmark.

    O Streamlit não executa novamente o experimento.
    Ele apenas apresenta as evidências produzidas
    pela camada experimental.
    """

    df = pd.read_csv(BENCHMARK_FILE)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "benchmark.csv não possui as colunas esperadas: "
            + ", ".join(missing_columns)
        )

    df = df.copy()

    df["model"] = (
        df["model"]
        .astype(str)
        .str.upper()
    )

    return df


# ============================================================
# EXPERIMENT SUMMARY
# ============================================================

def _get_benchmark_summary(df: pd.DataFrame) -> dict:
    """
    Extrai as principais evidências do benchmark.
    """

    best_accuracy_row = df.loc[
        df["accuracy"].idxmax()
    ]

    best_f1_row = df.loc[
        df["f1_macro"].idxmax()
    ]

    best_model = best_accuracy_row["model"]

    competitors = df[
        df["model"] != best_model
    ]

    second_best_accuracy = (
        competitors["accuracy"].max()
        if not competitors.empty
        else best_accuracy_row["accuracy"]
    )

    accuracy_advantage = (
        best_accuracy_row["accuracy"]
        - second_best_accuracy
    )

    return {
        "best_model": best_model,
        "best_accuracy_row": best_accuracy_row,
        "best_f1_row": best_f1_row,
        "accuracy_advantage": accuracy_advantage,
    }


# ============================================================
# EXPERIMENT CONTEXT
# ============================================================

def _render_experiment_context() -> None:
    """
    Faz a transição narrativa entre Model Selection
    e Clean Benchmark.
    """

    st.subheader("Experiment Context")

    st.markdown(
        """
        Na etapa de **Model Selection**, SVM, MLP e CNN foram
        escolhidos para representar diferentes estratégias de
        aprendizado e representação da imagem.

        O **Clean Benchmark** constitui a primeira comparação
        quantitativa entre essas arquiteturas.

        Todos os modelos são avaliados no conjunto de teste original
        do Fashion-MNIST, preservando o domínio utilizado como
        referência experimental.
        """
    )

    st.info(
        "Pergunta desta etapa: qual arquitetura apresenta o melhor "
        "desempenho preditivo no domínio original?"
    )


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

def _render_performance_summary(
    summary: dict,
) -> None:
    """
    Exibe os principais resultados do benchmark.
    """

    best_accuracy_row = summary[
        "best_accuracy_row"
    ]

    best_f1_row = summary[
        "best_f1_row"
    ]

    st.subheader("Clean Test Performance")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Best Accuracy",
            f"{best_accuracy_row['accuracy']:.2%}",
            help=(
                "Modelo: "
                f"{best_accuracy_row['model']}"
            ),
        )

    with c2:
        st.metric(
            "Best F1 Macro",
            f"{best_f1_row['f1_macro']:.2%}",
            help=(
                "Modelo: "
                f"{best_f1_row['model']}"
            ),
        )

    with c3:
        st.metric(
            "Best Clean Model",
            summary["best_model"],
            help=(
                "Modelo com maior accuracy no "
                "conjunto de teste limpo."
            ),
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

def _render_model_performance(
    df: pd.DataFrame,
) -> None:
    """
    Compara visualmente as métricas dos modelos.
    """

    st.subheader("Model Performance")

    st.caption(
        "Comparação de Accuracy, Precision Macro, "
        "Recall Macro e F1 Macro."
    )

    chart_df = (
        df[
            [
                "model",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
            ]
        ]
        .set_index("model")
        .rename(
            columns={
                "accuracy": "Accuracy",
                "precision_macro": "Precision",
                "recall_macro": "Recall",
                "f1_macro": "F1 Macro",
            }
        )
    )

    st.bar_chart(
        chart_df,
        use_container_width=True,
        stack=False,
    )


# ============================================================
# DETAILED RESULTS
# ============================================================

def _render_detailed_results(
    df: pd.DataFrame,
) -> None:
    """
    Exibe as evidências quantitativas detalhadas.
    """

    st.subheader("Detailed Results")

    display_df = df[
        [
            "model",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ]
    ].copy()

    display_df.columns = [
        "Model",
        "Accuracy",
        "Precision Macro",
        "Recall Macro",
        "F1 Macro",
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Accuracy":
                st.column_config.ProgressColumn(
                    "Accuracy",
                    format="%.4f",
                    min_value=0.0,
                    max_value=1.0,
                ),

            "Precision Macro":
                st.column_config.NumberColumn(
                    "Precision Macro",
                    format="%.4f",
                ),

            "Recall Macro":
                st.column_config.NumberColumn(
                    "Recall Macro",
                    format="%.4f",
                ),

            "F1 Macro":
                st.column_config.ProgressColumn(
                    "F1 Macro",
                    format="%.4f",
                    min_value=0.0,
                    max_value=1.0,
                ),
        },
    )


# ============================================================
# BENCHMARK INSIGHT
# ============================================================

def _render_benchmark_insight(
    summary: dict,
) -> None:
    """
    Interpreta os resultados sem extrapolar
    a evidência do Clean Benchmark.
    """

    best_model = summary["best_model"]

    best_row = summary[
        "best_accuracy_row"
    ]

    accuracy_advantage = summary[
        "accuracy_advantage"
    ]

    st.subheader("Benchmark Insight")

    st.success(
        f"{best_model} apresenta o melhor desempenho no "
        f"Fashion-MNIST limpo, com "
        f"{best_row['accuracy']:.2%} de Accuracy e "
        f"{best_row['f1_macro']:.2%} de F1 Macro."
    )

    st.markdown(
        f"""
        A vantagem de Accuracy sobre o segundo melhor modelo é de
        aproximadamente **{accuracy_advantage:.2%}**.

        Essa evidência sustenta a seleção inicial da **{best_model}**
        como arquitetura de melhor desempenho no domínio original.
        """
    )

    st.warning(
        "O Clean Benchmark mede desempenho em dados provenientes "
        "do mesmo domínio utilizado no desenvolvimento do modelo. "
        "Esse resultado, isoladamente, não demonstra Robustness, "
        "generalização para outros domínios ou prontidão para produção."
    )


# ============================================================
# EVIDENCE LAYER
# ============================================================

def _render_evidence_layer() -> None:
    """
    Posiciona MLflow como camada de rastreabilidade
    experimental do projeto.
    """

    st.subheader("Experiment Evidence")

    st.markdown(
        """
        Os resultados apresentados nesta página constituem a camada
        narrativa do experimento.

        O **MLflow** atua como **guardião das evidências experimentais**,
        mantendo a rastreabilidade dos runs que sustentam as conclusões
        apresentadas no Benchmark.
        """
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Models",
        "3",
    )

    c2.metric(
        "Tracked Metrics",
        "4+",
    )

    c3.metric(
        "Experiment Stage",
        "Baseline",
    )

    c4.metric(
        "Evidence",
        "MLflow",
    )

    st.caption(
        "Runs, parâmetros, métricas, modelos e artefatos "
        "permanecem registrados na camada de experiment tracking."
    )

    st.info(
        "Streamlit apresenta e interpreta as evidências. "
        "MLflow preserva sua rastreabilidade experimental."
    )


# ============================================================
# NEXT EXPERIMENT
# ============================================================

def _render_next_experiment() -> None:
    """
    Conecta o Clean Benchmark ao capítulo seguinte.
    """

    st.subheader("Next Experiment")

    st.markdown(
        """
        O Benchmark respondeu qual modelo apresenta melhor desempenho
        no domínio original.

        A próxima pergunta é mais exigente:

        **Esse desempenho permanece estável quando os dados são
        perturbados?**

        **Próxima etapa → Robustness**
        """
    )


# ============================================================
# PAGE
# ============================================================

def render_benchmark() -> None:
    """
    Renderiza a página modular do Clean Benchmark.
    """

    st.header("🏆 Clean Benchmark")

    st.caption(
        "Comparação experimental de SVM, MLP e CNN "
        "no domínio original do Fashion-MNIST."
    )

    try:
        df = load_benchmark()

        summary = _get_benchmark_summary(
            df
        )

        _render_experiment_context()

        st.divider()

        _render_performance_summary(
            summary
        )

        st.divider()

        _render_model_performance(
            df
        )

        st.divider()

        _render_detailed_results(
            df
        )

        st.divider()

        _render_benchmark_insight(
            summary
        )

        st.divider()

        _render_evidence_layer()

        st.divider()

        _render_next_experiment()

    except FileNotFoundError:
        st.warning(
            "Arquivo de evidência `benchmark.csv` "
            "não encontrado."
        )

        st.code(
            "python -m src.evaluation.benchmark",
            language="powershell",
        )

    except ValueError as exc:
        st.error(
            "O artefato do Benchmark possui "
            "estrutura incompatível."
        )

        st.code(
            str(exc)
        )

    except Exception as exc:
        st.error(
            "Não foi possível carregar o Clean Benchmark."
        )

        st.exception(exc)