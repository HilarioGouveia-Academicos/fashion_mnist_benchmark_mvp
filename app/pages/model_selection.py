import pandas as pd
import streamlit as st


# ============================================================
# MODEL DEFINITIONS
# ============================================================

MODEL_DATA = pd.DataFrame(
    [
        {
            "Model": "SVM",
            "Family": "Classical ML",
            "Input Representation": "Flattened 784 features",
            "Spatial Structure": "Not preserved",
            "Main Role": "Strong non-neural baseline",
        },
        {
            "Model": "MLP",
            "Family": "Neural Network",
            "Input Representation": "Flattened 784 features",
            "Spatial Structure": "Not preserved",
            "Main Role": "Neural baseline",
        },
        {
            "Model": "CNN",
            "Family": "Deep Learning / Vision",
            "Input Representation": "28 × 28 × 1 tensor",
            "Spatial Structure": "Preserved",
            "Main Role": "Spatial feature learning",
        },
    ]
)


# ============================================================
# WHY THESE MODELS
# ============================================================

def _render_why_these_models() -> None:
    """
    Explica a lógica experimental utilizada para selecionar
    SVM, MLP e CNN.
    """

    st.subheader("Why These Models?")

    st.markdown(
        """
        A escolha de **SVM, MLP e CNN** não representa apenas uma
        comparação entre três algoritmos.

        Os modelos foram selecionados para representar uma progressão
        controlada de abordagem, capacidade de representação e
        complexidade:

        **Classical Machine Learning → Dense Neural Network → Convolutional Neural Network**

        Essa progressão permite observar como diferentes formas de
        representar a mesma imagem influenciam o desempenho e,
        posteriormente, a Robustness e a capacidade de generalização.
        """
    )

    st.info(
        "O objetivo da comparação não é apenas descobrir qual modelo "
        "obtém maior acurácia, mas entender quais características "
        "arquiteturais contribuem para o comportamento observado."
    )


# ============================================================
# SVM
# ============================================================

def _render_svm() -> None:
    """
    Explica o papel do SVM na estratégia experimental.
    """

    st.subheader("SVM — Classical ML Baseline")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(
            "Model Family",
            "Classical ML",
        )

        st.metric(
            "Input",
            "784 features",
        )

    with col2:
        st.markdown(
            """
            O **Support Vector Machine (SVM)** representa a referência
            de Machine Learning clássico do estudo.

            Antes da modelagem, cada imagem de **28 × 28 pixels** é
            normalizada e transformada em um vetor com **784 features**.

            Dessa forma, o modelo recebe a informação visual sem uma
            representação explícita da posição espacial dos pixels.

            Seu papel é fornecer uma referência forte e não neural,
            permitindo avaliar até onde uma abordagem clássica consegue
            chegar utilizando a mesma informação disponível para os
            demais modelos.
            """
        )

    st.caption(
        "Experimental role: estabelecer um baseline clássico "
        "para comparação com arquiteturas neurais."
    )


# ============================================================
# MLP
# ============================================================

def _render_mlp() -> None:
    """
    Explica o papel da MLP na estratégia experimental.
    """

    st.subheader("MLP — Neural Baseline")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(
            "Model Family",
            "Neural Network",
        )

        st.metric(
            "Input",
            "784 features",
        )

    with col2:
        st.markdown(
            """
            A **Multilayer Perceptron (MLP)** introduz aprendizado neural
            e capacidade de modelar relações não lineares entre as
            features.

            Entretanto, assim como no SVM, a imagem ainda é apresentada
            ao modelo como um vetor achatado de **784 valores**.

            Isso torna a MLP particularmente útil na comparação porque
            ela cria uma ponte entre Machine Learning clássico e
            arquiteturas de Deep Learning voltadas para visão.

            Assim, podemos observar o efeito de adicionar capacidade
            neural sem ainda explorar explicitamente a estrutura
            espacial da imagem.
            """
        )

    st.caption(
        "Experimental role: separar o ganho de uma rede neural "
        "do ganho específico obtido por uma arquitetura convolucional."
    )


# ============================================================
# CNN
# ============================================================

def _render_cnn() -> None:
    """
    Explica o papel da CNN na estratégia experimental.
    """

    st.subheader("CNN — Spatial Vision Model")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(
            "Model Family",
            "Deep Learning",
        )

        st.metric(
            "Input",
            "28 × 28 × 1",
        )

    with col2:
        st.markdown(
            """
            A **Convolutional Neural Network (CNN)** preserva a estrutura
            espacial original da imagem.

            Em vez de transformar cada observação em um vetor, a CNN
            recebe um tensor **28 × 28 × 1** e utiliza operações
            convolucionais para aprender padrões locais.

            Isso permite capturar relações como contornos, formas,
            regiões e combinações espaciais que podem ser importantes
            para distinguir diferentes categorias de vestuário.

            Por essa razão, a CNN representa a arquitetura mais alinhada
            à natureza visual do problema.
            """
        )

    st.caption(
        "Experimental role: avaliar o impacto de preservar e explorar "
        "explicitamente a estrutura espacial das imagens."
    )


# ============================================================
# COMPARISON
# ============================================================

def _render_model_comparison() -> None:
    """
    Resume as diferenças conceituais entre os três modelos.
    """

    st.subheader("Comparison Perspective")

    st.dataframe(
        MODEL_DATA,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Model": st.column_config.TextColumn(
                "Model"
            ),
            "Family": st.column_config.TextColumn(
                "Family"
            ),
            "Input Representation":
                st.column_config.TextColumn(
                    "Input Representation"
                ),
            "Spatial Structure":
                st.column_config.TextColumn(
                    "Spatial Structure"
                ),
            "Main Role":
                st.column_config.TextColumn(
                    "Experimental Role"
                ),
        },
    )

    st.markdown(
        """
        **Representation Flow**

        `SVM`
        → imagem convertida para vetor

        `MLP`
        → imagem convertida para vetor + aprendizado neural

        `CNN`
        → estrutura espacial preservada + aprendizado convolucional
        """
    )


# ============================================================
# RESEARCH QUESTION
# ============================================================

def _render_research_question() -> None:
    """
    Formaliza a pergunta que orienta a comparação.
    """

    st.subheader("Research Question")

    st.info(
        "Preservar e explorar a estrutura espacial das imagens "
        "melhora não apenas a acurácia, mas também a Robustness "
        "e a capacidade de generalização?"
    )

    st.markdown(
        """
        Essa pergunta orienta a comparação entre os três modelos.

        O **Benchmark** responde inicialmente qual arquitetura apresenta
        melhor desempenho no domínio original.

        As etapas seguintes ampliam a análise:

        **Robustness** investiga o comportamento diante de perturbações.

        **Domain Shift** avalia o comportamento quando a distribuição
        dos dados se afasta daquela observada durante o treinamento.
        """
    )


# ============================================================
# SELECTION STRATEGY
# ============================================================

def _render_selection_strategy() -> None:
    """
    Resume a lógica da seleção experimental.
    """

    st.subheader("Selection Strategy")

    st.markdown(
        """
        A estratégia pode ser resumida como uma evolução controlada:

        **SVM**
        → referência de Machine Learning clássico

        **MLP**
        → introdução de aprendizado neural

        **CNN**
        → introdução de representação espacial e convoluções
        """
    )

    st.markdown(
        """
        ```text
        Classical ML
             │
             ▼
            SVM
             │
             ▼
        Neural Learning
             │
             ▼
            MLP
             │
             ▼
        Spatial Learning
             │
             ▼
            CNN
        ```
        """
    )


# ============================================================
# OUTCOME
# ============================================================

def _render_outcome() -> None:
    """
    Conecta a justificativa inicial ao resultado experimental.
    """

    st.subheader("Outcome")

    st.markdown(
        """
        Os experimentos posteriores mostraram que a **CNN apresentou
        o melhor desempenho global** entre as arquiteturas avaliadas.

        A CNN liderou o Clean Benchmark e também demonstrou melhor
        comportamento agregado nos estudos de Robustness e
        Domain Shift.

        Com base nessas evidências, a CNN foi selecionada como
        arquitetura principal para as próximas etapas do projeto.
        """
    )

    st.success(
        "Selected Architecture for the next phase: CNN"
    )


# ============================================================
# WHY THIS MATTERS
# ============================================================

def _render_why_this_matters() -> None:
    """
    Explica o valor da estratégia de seleção para a narrativa
    técnica do projeto.
    """

    st.subheader("Why This Matters")

    st.markdown(
        """
        Comparar modelos com diferentes características arquiteturais
        torna a seleção mais informativa do que simplesmente executar
        vários algoritmos e escolher aquele com maior acurácia.

        Neste estudo, cada modelo possui um papel experimental claro.

        Isso permite interpretar os resultados como parte de uma
        sequência de decisões técnicas, e não apenas como uma tabela
        de scores.
        """
    )

    st.info(
        "O Benchmark deixa de responder apenas "
        "\"quem venceu?\" e passa também a responder "
        "\"por que essa arquitetura merece avançar para a próxima fase?\""
    )


# ============================================================
# PAGE
# ============================================================

def render_model_selection() -> None:
    """
    Renderiza a página Model Selection Rationale.
    """

    st.header(
        "Model Selection Rationale"
    )

    st.caption(
        "Justificativa técnica e experimental para a escolha "
        "de SVM, MLP e CNN."
    )

    _render_why_these_models()

    st.divider()

    _render_svm()

    st.divider()

    _render_mlp()

    st.divider()

    _render_cnn()

    st.divider()

    _render_model_comparison()

    st.divider()

    _render_research_question()

    st.divider()

    _render_selection_strategy()

    st.divider()

    _render_outcome()

    st.divider()

    _render_why_this_matters()