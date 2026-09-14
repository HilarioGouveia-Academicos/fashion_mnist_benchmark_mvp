from pathlib import Path

import pandas as pd
import streamlit as st

from config.settings import ARTIFACTS_DIR


# ============================================================
# CONFIGURATION
# ============================================================

CONTROLLED_DIR = Path(ARTIFACTS_DIR) / "domain_shift"
EXTERNAL_RAW_DIR = Path(ARTIFACTS_DIR) / "domain_shift_v2"
CANONICAL_DIR = Path(ARTIFACTS_DIR) / "domain_shift_v2_1"

CONTROLLED_RESULTS_FILE = (
    CONTROLLED_DIR / "domain_shift_results.csv"
)

CONTROLLED_SUMMARY_FILE = (
    CONTROLLED_DIR / "domain_shift_summary.csv"
)

EXTERNAL_RESULTS_FILE = (
    EXTERNAL_RAW_DIR / "external_results.csv"
)

CANONICAL_COMPARISON_FILE = (
    CANONICAL_DIR / "raw_vs_canonical.csv"
)


# ============================================================
# DATA ACCESS
# ============================================================

@st.cache_data
def load_domain_shift_artifacts():
    """
    Carrega os artefatos congelados dos experimentos de Domain Shift.

    O Streamlit não executa novamente os testes.
    Ele apenas apresenta e interpreta as evidências produzidas
    pela camada experimental.
    """

    controlled = pd.read_csv(
        CONTROLLED_RESULTS_FILE
    )

    controlled_summary = pd.read_csv(
        CONTROLLED_SUMMARY_FILE
    )

    external = pd.read_csv(
        EXTERNAL_RESULTS_FILE
    )

    comparison = pd.read_csv(
        CANONICAL_COMPARISON_FILE
    )

    _validate_controlled_results(
        controlled
    )

    _validate_controlled_summary(
        controlled_summary
    )

    _validate_external_results(
        external
    )

    _validate_canonical_comparison(
        comparison
    )

    controlled = _normalize_model_names(
        controlled
    )

    controlled_summary = _normalize_model_names(
        controlled_summary
    )

    external = _normalize_model_names(
        external
    )

    comparison = _normalize_model_names(
        comparison
    )

    return (
        controlled,
        controlled_summary,
        external,
        comparison,
    )


def _normalize_model_names(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normaliza nomes de modelos apenas para apresentação.
    """

    df = df.copy()

    df["model"] = (
        df["model"]
        .astype(str)
        .str.upper()
    )

    return df


def _validate_columns(
    df: pd.DataFrame,
    required_columns: set,
    artifact_name: str,
) -> None:
    """
    Valida a estrutura mínima dos artefatos experimentais.
    """

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{artifact_name} não possui as colunas esperadas: "
            + ", ".join(sorted(missing))
        )


def _validate_controlled_results(
    df: pd.DataFrame,
) -> None:
    _validate_columns(
        df,
        {
            "model",
            "domain",
            "accuracy",
            "mean_confidence",
        },
        "domain_shift_results.csv",
    )


def _validate_controlled_summary(
    df: pd.DataFrame,
) -> None:
    _validate_columns(
        df,
        {
            "model",
            "mean_shifted_accuracy",
            "min_shifted_accuracy",
            "mean_accuracy_drop",
            "max_accuracy_drop",
            "mean_error_confidence",
        },
        "domain_shift_summary.csv",
    )


def _validate_external_results(
    df: pd.DataFrame,
) -> None:
    _validate_columns(
        df,
        {
            "model",
            "accuracy",
            "f1_macro",
            "mean_confidence",
            "confidence_on_errors",
            "mean_entropy",
        },
        "external_results.csv",
    )


def _validate_canonical_comparison(
    df: pd.DataFrame,
) -> None:
    _validate_columns(
        df,
        {
            "model",
            "raw_accuracy",
            "canonical_accuracy",
            "accuracy_gain",
            "raw_error_confidence",
            "canonical_error_confidence",
        },
        "raw_vs_canonical.csv",
    )


# ============================================================
# SUMMARY
# ============================================================

def _get_domain_shift_summary(
    controlled_summary: pd.DataFrame,
    external: pd.DataFrame,
    comparison: pd.DataFrame,
) -> dict:
    """
    Obtém os principais resultados dos experimentos.
    """

    best_controlled = controlled_summary.loc[
        controlled_summary[
            "mean_shifted_accuracy"
        ].idxmax()
    ]

    best_external = external.loc[
        external["accuracy"].idxmax()
    ]

    best_canonical = comparison.loc[
        comparison[
            "canonical_accuracy"
        ].idxmax()
    ]

    largest_gain = comparison.loc[
        comparison[
            "accuracy_gain"
        ].idxmax()
    ]

    return {
        "best_controlled": best_controlled,
        "best_external": best_external,
        "best_canonical": best_canonical,
        "largest_gain": largest_gain,
    }


# ============================================================
# EXPERIMENT CONTEXT
# ============================================================

def _render_experiment_context() -> None:
    """
    Conecta Robustness ao Domain Shift.
    """

    st.subheader("Experiment Context")

    st.markdown(
        """
        O **Robustness Benchmark** avaliou como os modelos reagem
        a perturbações controladas aplicadas ao domínio original.

        O **Domain Shift** amplia essa análise.

        Aqui, a questão não é apenas alterar uma imagem,
        mas observar como o comportamento do modelo muda quando
        a representação visual começa a se afastar do padrão
        utilizado durante o treinamento.
        """
    )

    st.info(
        "Pergunta desta etapa: até que ponto o desempenho observado "
        "no Fashion-MNIST permanece confiável quando a distribuição "
        "visual dos dados muda?"
    )


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

def _render_summary_cards(
    summary: dict,
) -> None:
    """
    Exibe os principais resultados da sequência de Domain Shift.
    """

    best_controlled = summary[
        "best_controlled"
    ]

    best_external = summary[
        "best_external"
    ]

    best_canonical = summary[
        "best_canonical"
    ]

    largest_gain = summary[
        "largest_gain"
    ]

    st.subheader("Domain Generalization Summary")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Best Controlled",
            best_controlled["model"],
            help=(
                "Mean Shifted Accuracy: "
                f"{best_controlled['mean_shifted_accuracy']:.2%}"
            ),
        )

    with c2:
        st.metric(
            "Best External RAW",
            best_external["model"],
            help=(
                "Accuracy: "
                f"{best_external['accuracy']:.2%}"
            ),
        )

    with c3:
        st.metric(
            "Best Canonicalized",
            best_canonical["model"],
            help=(
                "Accuracy: "
                f"{best_canonical['canonical_accuracy']:.2%}"
            ),
        )

    with c4:
        st.metric(
            "Largest Canonical Gain",
            largest_gain["model"],
            help=(
                "Accuracy Gain: "
                f"+{largest_gain['accuracy_gain']:.2%}"
            ),
        )


# ============================================================
# ACT 1 — CONTROLLED DOMAIN SHIFT
# ============================================================

def _render_controlled_domain_shift(
    controlled: pd.DataFrame,
    controlled_summary: pd.DataFrame,
) -> None:
    """
    Apresenta mudanças controladas de representação.
    """

    st.subheader("1️⃣ Controlled Domain Shift")

    st.caption(
        "Transformações controladas aplicadas ao Fashion-MNIST "
        "para medir sensibilidade a mudanças de representação."
    )

    shifted = controlled[
        controlled["domain"] != "clean"
    ].copy()

    controlled_chart = shifted.pivot(
        index="domain",
        columns="model",
        values="accuracy",
    )

    st.bar_chart(
        controlled_chart,
        stack=False,
        use_container_width=True,
    )

    controlled_display = (
        controlled_summary[
            [
                "model",
                "mean_shifted_accuracy",
                "min_shifted_accuracy",
                "mean_accuracy_drop",
                "max_accuracy_drop",
                "mean_error_confidence",
            ]
        ]
        .copy()
    )

    controlled_display.columns = [
        "Model",
        "Mean Shifted Accuracy",
        "Worst Accuracy",
        "Mean Drop",
        "Max Drop",
        "Error Confidence",
    ]

    st.dataframe(
        controlled_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Mean Shifted Accuracy":
                st.column_config.ProgressColumn(
                    "Mean Shifted Accuracy",
                    format="%.4f",
                    min_value=0.0,
                    max_value=1.0,
                ),

            "Worst Accuracy":
                st.column_config.NumberColumn(
                    "Worst Accuracy",
                    format="%.4f",
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

            "Error Confidence":
                st.column_config.NumberColumn(
                    "Error Confidence",
                    format="%.4f",
                ),
        },
    )

    best_row = controlled_summary.loc[
        controlled_summary[
            "mean_shifted_accuracy"
        ].idxmax()
    ]

    st.success(
        f"{best_row['model']} apresenta a melhor estabilidade média "
        f"no Controlled Domain Shift, com "
        f"{best_row['mean_shifted_accuracy']:.2%} de Accuracy média."
    )

    _render_high_confidence_failure(
        controlled
    )


def _render_high_confidence_failure(
    controlled: pd.DataFrame,
) -> None:
    """
    Destaca um caso em que confiança e acerto divergem.
    """

    mlp_inverted = controlled[
        (controlled["model"] == "MLP")
        & (controlled["domain"] == "inverted")
    ]

    if mlp_inverted.empty:
        return

    row = mlp_inverted.iloc[0]

    st.warning(
        "Caso crítico — MLP / inverted: "
        f"Accuracy de {row['accuracy']:.2%}, mas confiança média de "
        f"{row['mean_confidence']:.2%}. "
        "Esse resultado evidencia que alta confiança não implica "
        "necessariamente alta confiabilidade."
    )


# ============================================================
# ACT 2 — SYNTHETIC EXTERNAL RAW
# ============================================================

def _render_external_raw(
    external: pd.DataFrame,
) -> None:
    """
    Apresenta o piloto sintético externo sem canonicalização.
    """

    st.subheader("2️⃣ Synthetic External — RAW")

    st.caption(
        "Pilot externo sintético independente do Fashion-MNIST. "
        "As imagens são avaliadas sem canonicalização."
    )

    external_chart = (
        external[
            [
                "model",
                "accuracy",
                "mean_confidence",
                "confidence_on_errors",
            ]
        ]
        .set_index("model")
        .rename(
            columns={
                "accuracy": "Accuracy",
                "mean_confidence": "Confidence",
                "confidence_on_errors":
                    "Error Confidence",
            }
        )
    )

    st.bar_chart(
        external_chart,
        stack=False,
        use_container_width=True,
    )

    external_display = external[
        [
            "model",
            "accuracy",
            "f1_macro",
            "mean_confidence",
            "confidence_on_errors",
            "mean_entropy",
        ]
    ].copy()

    external_display.columns = [
        "Model",
        "Accuracy",
        "F1 Macro",
        "Confidence",
        "Error Confidence",
        "Entropy",
    ]

    st.dataframe(
        external_display,
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

            "F1 Macro":
                st.column_config.NumberColumn(
                    "F1 Macro",
                    format="%.4f",
                ),

            "Confidence":
                st.column_config.NumberColumn(
                    "Confidence",
                    format="%.4f",
                ),

            "Error Confidence":
                st.column_config.NumberColumn(
                    "Error Confidence",
                    format="%.4f",
                ),

            "Entropy":
                st.column_config.NumberColumn(
                    "Entropy",
                    format="%.4f",
                ),
        },
    )

    best_external = external.loc[
        external["accuracy"].idxmax()
    ]

    st.warning(
        "O domínio externo RAW produz forte degradação. "
        f"O melhor resultado pertence à {best_external['model']}, "
        f"com apenas {best_external['accuracy']:.2%} de Accuracy."
    )

    st.markdown(
        """
        Esse resultado sugere que o desempenho obtido no domínio
        original não se transfere automaticamente para imagens com
        representação visual diferente.

        O problema passa a ser menos sobre **qual modelo vence**
        e mais sobre **qual representação o modelo espera receber**.
        """
    )


# ============================================================
# ACT 3 — CANONICALIZATION
# ============================================================

def _render_canonicalization(
    comparison: pd.DataFrame,
) -> None:
    """
    Apresenta o impacto da canonicalização das imagens externas.
    """

    st.subheader("3️⃣ Canonicalization Impact")

    st.caption(
        "Mesmas imagens externas e mesmos modelos. "
        "Não há retraining; apenas a representação de entrada muda."
    )

    canonical_chart = (
        comparison[
            [
                "model",
                "raw_accuracy",
                "canonical_accuracy",
            ]
        ]
        .set_index("model")
        .rename(
            columns={
                "raw_accuracy": "RAW",
                "canonical_accuracy":
                    "Canonicalized",
            }
        )
    )

    st.bar_chart(
        canonical_chart,
        stack=False,
        use_container_width=True,
    )

    st.subheader("Accuracy Gain")

    gain_chart = (
        comparison[
            [
                "model",
                "accuracy_gain",
            ]
        ]
        .set_index("model")
        .rename(
            columns={
                "accuracy_gain":
                    "Accuracy Gain",
            }
        )
    )

    st.bar_chart(
        gain_chart,
        use_container_width=True,
    )

    comparison_display = comparison[
        [
            "model",
            "raw_accuracy",
            "canonical_accuracy",
            "accuracy_gain",
            "raw_error_confidence",
            "canonical_error_confidence",
        ]
    ].copy()

    comparison_display.columns = [
        "Model",
        "RAW Accuracy",
        "Canonical Accuracy",
        "Accuracy Gain",
        "RAW Error Confidence",
        "Canonical Error Confidence",
    ]

    st.dataframe(
        comparison_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "RAW Accuracy":
                st.column_config.NumberColumn(
                    "RAW Accuracy",
                    format="%.4f",
                ),

            "Canonical Accuracy":
                st.column_config.ProgressColumn(
                    "Canonical Accuracy",
                    format="%.4f",
                    min_value=0.0,
                    max_value=1.0,
                ),

            "Accuracy Gain":
                st.column_config.NumberColumn(
                    "Accuracy Gain",
                    format="%.4f",
                ),

            "RAW Error Confidence":
                st.column_config.NumberColumn(
                    "RAW Error Confidence",
                    format="%.4f",
                ),

            "Canonical Error Confidence":
                st.column_config.NumberColumn(
                    "Canonical Error Confidence",
                    format="%.4f",
                ),
        },
    )

    largest_gain = comparison.loc[
        comparison[
            "accuracy_gain"
        ].idxmax()
    ]

    best_canonical = comparison.loc[
        comparison[
            "canonical_accuracy"
        ].idxmax()
    ]

    st.success(
        "A canonicalização melhora todos os modelos sem retraining. "
        f"O maior ganho é observado na {largest_gain['model']}, com "
        f"+{largest_gain['accuracy_gain']:.2%}. "
        f"O melhor desempenho canonicalizado é da "
        f"{best_canonical['model']}, com "
        f"{best_canonical['canonical_accuracy']:.2%} de Accuracy."
    )


# ============================================================
# DOMAIN SHIFT INSIGHT
# ============================================================

def _render_domain_shift_insight(
    controlled_summary: pd.DataFrame,
    external: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    """
    Consolida a narrativa experimental.
    """

    best_controlled = controlled_summary.loc[
        controlled_summary[
            "mean_shifted_accuracy"
        ].idxmax()
    ]

    best_external = external.loc[
        external["accuracy"].idxmax()
    ]

    best_canonical = comparison.loc[
        comparison[
            "canonical_accuracy"
        ].idxmax()
    ]

    st.subheader("Domain Shift Insight")

    st.markdown(
        f"""
        A sequência experimental revela diferentes níveis de
        dificuldade de generalização.

        **1. Clean Domain**  
        Os modelos apresentam alto desempenho no Fashion-MNIST
        original.

        **2. Controlled Domain Shift**  
        A melhor estabilidade média é obtida pela
        **{best_controlled['model']}**, com aproximadamente
        **{best_controlled['mean_shifted_accuracy']:.2%}**
        de Accuracy média.

        **3. Synthetic External RAW**  
        O desempenho entra em forte degradação.
        Mesmo o melhor modelo atinge apenas
        **{best_external['accuracy']:.2%}**.

        **4. Canonicalized External**  
        A reorganização da representação de entrada recupera parte
        do desempenho, chegando a
        **{best_canonical['canonical_accuracy']:.2%}**
        no melhor resultado.
        """
    )

    st.info(
        "A canonicalização demonstra que parte do domain gap está "
        "associada à representação da entrada. Entretanto, a recuperação "
        "parcial mostra que o problema não é resolvido apenas com "
        "pré-processamento."
    )

    st.warning(
        "O piloto V2A utiliza apenas 40 imagens sintéticas externas "
        "(4 por classe). Portanto, esses resultados constituem uma "
        "evidência exploratória controlada e não uma validação suficiente "
        "de generalização para o mundo real."
    )


# ============================================================
# RELIABILITY PERSPECTIVE
# ============================================================

def _render_reliability_perspective() -> None:
    """
    Conecta Domain Shift à tese de confiabilidade do projeto.
    """

    st.subheader("Reliability Perspective")

    st.markdown(
        """
        Esta etapa altera significativamente a interpretação do projeto.

        No **Clean Benchmark**, a pergunta era:

        > Qual modelo apresenta maior desempenho?

        No **Domain Shift**, a pergunta passa a ser:

        > Até onde podemos confiar nesse desempenho quando os dados mudam?

        Essa mudança de perspectiva é central para a avaliação de
        confiabilidade e prontidão de modelos de Machine Learning.
        """
    )

    st.info(
        "Accuracy mede desempenho em um cenário observado. "
        "Reliability exige compreender como esse desempenho muda "
        "quando as condições de entrada deixam de ser ideais."
    )


# ============================================================
# EVIDENCE LAYER
# ============================================================

def _render_evidence_layer() -> None:
    """
    Posiciona MLflow como camada de rastreabilidade
    para os experimentos de Domain Shift.
    """

    st.subheader("Experiment Evidence")

    st.markdown(
        """
        Os resultados de Domain Shift formam uma nova camada de
        evidências sobre o comportamento dos modelos fora das
        condições originais de avaliação.

        O **MLflow** deve atuar como o sistema de rastreabilidade dessas
        evidências, preservando métricas, parâmetros, artefatos,
        versões de preprocessing e futuras execuções de adaptação.
        """
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Controlled Shift",
        "Validated",
    )

    c2.metric(
        "External RAW",
        "Pilot",
    )

    c3.metric(
        "Canonicalization",
        "Validated",
    )

    c4.metric(
        "Evidence Layer",
        "MLflow",
    )

    st.caption(
        "Streamlit apresenta a narrativa analítica. "
        "MLflow preserva a trilha experimental que sustenta "
        "as conclusões."
    )


# ============================================================
# NEXT EXPERIMENT
# ============================================================

def _render_next_experiment() -> None:
    """
    Define a próxima etapa experimental sem antecipar conclusões.
    """

    st.subheader("Next Experiment")

    st.markdown(
        """
        O piloto sintético mostrou que existe um domain gap relevante
        e que a canonicalização consegue recuperar apenas parte
        do desempenho.

        O próximo passo deve introduzir evidência mais próxima
        do cenário real.

        **Próxima etapa → V2B Real External Domain**

        A partir dessa avaliação será possível decidir, com base em
        evidências, se a CNN precisa de:

        - ajuste de preprocessing;
        - data augmentation;
        - fine-tuning;
        - ou outra estratégia de domain adaptation.
        """
    )

    st.warning(
        "Fine-tuning não deve ser aplicado apenas para melhorar "
        "o resultado do piloto atual. A estratégia de adaptação deve "
        "ser definida após a avaliação com dados externos reais."
    )


# ============================================================
# PAGE
# ============================================================

def render_domain_shift() -> None:
    """
    Renderiza a página modular de Domain Shift.
    """

    st.header("🌐 Domain Shift")

    st.caption(
        "Avaliação da generalização dos modelos quando a distribuição "
        "visual se afasta do domínio original do Fashion-MNIST."
    )

    try:
        (
            controlled,
            controlled_summary,
            external,
            comparison,
        ) = load_domain_shift_artifacts()

        summary = _get_domain_shift_summary(
            controlled_summary,
            external,
            comparison,
        )

        _render_experiment_context()

        st.divider()

        _render_summary_cards(
            summary
        )

        st.divider()

        _render_controlled_domain_shift(
            controlled,
            controlled_summary,
        )

        st.divider()

        _render_external_raw(
            external
        )

        st.divider()

        _render_canonicalization(
            comparison
        )

        st.divider()

        _render_domain_shift_insight(
            controlled_summary,
            external,
            comparison,
        )

        st.divider()

        _render_reliability_perspective()

        st.divider()

        _render_evidence_layer()

        st.divider()

        _render_next_experiment()

    except FileNotFoundError as exc:
        st.warning(
            "Um ou mais artefatos de Domain Shift "
            "não foram encontrados."
        )

        st.caption(str(exc))

    except ValueError as exc:
        st.error(
            "Um dos artefatos de Domain Shift possui "
            "estrutura incompatível."
        )

        st.code(
            str(exc)
        )

    except Exception as exc:
        st.error(
            "Não foi possível carregar os resultados "
            "de Domain Shift."
        )

        st.exception(exc)