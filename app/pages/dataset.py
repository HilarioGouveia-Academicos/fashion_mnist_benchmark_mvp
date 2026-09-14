from typing import Final

import numpy as np
import pandas as pd
import streamlit as st

from src.data.loader import load_fashion_mnist


# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES: Final[dict[int, str]] = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

ALL_CLASSES_LABEL: Final[str] = "All Classes"


# ============================================================
# DATA ACCESS
# ============================================================

@st.cache_data
def load_dataset():
    """
    Carrega o Fashion-MNIST utilizando o loader oficial do projeto.

    Returns
    -------
    tuple
        x_train, y_train, x_test, y_test
    """

    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()

    return x_train, y_train, x_test, y_test




# ============================================================
# DATA PREPARATION
# ============================================================

def build_dataset_index(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Cria uma representação tabular dos registros do dataset.

    A tabela não expõe os 784 pixels individualmente.
    Ela apresenta apenas metadados úteis para exploração.
    """

    train_df = pd.DataFrame(
        {
            "sample_id": np.arange(len(y_train)),
            "split": "train",
            "class_id": y_train.astype(int),
        }
    )

    test_df = pd.DataFrame(
        {
            "sample_id": np.arange(len(y_test)),
            "split": "test",
            "class_id": y_test.astype(int),
        }
    )

    dataframe = pd.concat(
        [train_df, test_df],
        ignore_index=True,
    )

    dataframe["class_name"] = dataframe[
        "class_id"
    ].map(CLASS_NAMES)

    dataframe["image_shape"] = "28 × 28"

    return dataframe


def build_class_distribution(
    dataset_index: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula a distribuição de registros por classe.
    """

    distribution = (
        dataset_index
        .groupby(
            ["class_id", "class_name"],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "records"
            }
        )
        .sort_values("class_id")
    )

    total_records = distribution[
        "records"
    ].sum()

    distribution["percentage"] = (
        distribution["records"]
        / total_records
    )

    return distribution


# ============================================================
# DATASET OVERVIEW
# ============================================================

def _render_dataset_overview(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Exibe os principais atributos estruturais do dataset.
    """

    st.subheader("Dataset Overview")

    total_records = (
        len(x_train)
        + len(x_test)
    )

    total_classes = len(
        np.unique(
            np.concatenate(
                [y_train, y_test]
            )
        )
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Records",
            value=f"{total_records:,}",
        )

        st.caption(
            "Quantidade total de imagens utilizadas "
            "entre treino e teste."
        )

    with col2:
        st.metric(
            label="Total Classes",
            value=total_classes,
        )

        st.caption(
            "Número de categorias utilizadas "
            "na tarefa de classificação."
        )

    with col3:
        st.metric(
            label="Image Shape",
            value="28 × 28",
        )

        st.caption(
            "Cada observação é uma imagem grayscale "
            "com 784 pixels."
        )

    with col4:
        st.metric(
            label="Target",
            value="label",
        )

        st.caption(
            "Variável alvo multiclasse representando "
            "a categoria da peça."
        )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Train Samples",
        f"{len(x_train):,}",
    )

    col2.metric(
        "Test Samples",
        f"{len(x_test):,}",
    )

    col3.metric(
        "Task",
        "Multiclass",
    )

    col4.metric(
        "Pixel Range",
        "0–255",
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

def _render_dataset_summary() -> None:
    """
    Resume as principais características do Fashion-MNIST.
    """

    st.subheader("Dataset Summary")

    st.markdown(
        """
        O **Fashion-MNIST** é um conjunto de dados de classificação
        de imagens composto por dez categorias de produtos de vestuário.

        Cada registro corresponde a uma imagem em escala de cinza com
        resolução de **28 × 28 pixels**, associada a uma classe alvo.

        O problema é tratado como **Multiclass Classification**:
        para cada imagem, o modelo deve identificar uma entre dez
        categorias possíveis.

        O dataset possui uma estrutura altamente padronizada, com
        imagens centralizadas, baixo nível de ruído e representação
        visual relativamente consistente. Essas características ajudam
        a explicar o bom desempenho obtido pelos modelos no
        **Clean Benchmark**, mas também tornam importante avaliar
        posteriormente o comportamento diante de **Domain Shift**.
        """
    )


# ============================================================
# RECORD EXPLORER
# ============================================================

def _render_record_explorer(
    dataset_index: pd.DataFrame,
) -> None:
    """
    Permite explorar registros por classe e por split.
    """

    st.subheader("Sample Records")

    col1, col2 = st.columns(2)

    class_options = [
        ALL_CLASSES_LABEL,
        *CLASS_NAMES.values(),
    ]

    selected_class = col1.selectbox(
        "Filter by Class",
        class_options,
    )

    selected_split = col2.selectbox(
        "Filter by Split",
        [
            "All",
            "train",
            "test",
        ],
    )

    filtered = dataset_index.copy()

    if selected_class != ALL_CLASSES_LABEL:
        filtered = filtered[
            filtered["class_name"]
            == selected_class
        ]

    if selected_split != "All":
        filtered = filtered[
            filtered["split"]
            == selected_split
        ]

    st.caption(
        f"{len(filtered):,} registros encontrados."
    )

    st.dataframe(
        filtered[
            [
                "sample_id",
                "split",
                "class_id",
                "class_name",
                "image_shape",
            ]
        ],
        hide_index=True,
        use_container_width=True,
        height=320,
    )


# ============================================================
# IMAGE GALLERY
# ============================================================

def _render_image_gallery(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> None:
    """
    Exibe exemplos reais das classes do Fashion-MNIST.
    """

    st.subheader("Class Samples")

    st.caption(
        "Exemplos do conjunto de treinamento ajudam a visualizar "
        "a representação que os modelos receberam durante o aprendizado."
    )

    columns = st.columns(5)

    for class_id, class_name in CLASS_NAMES.items():

        matching_indices = np.where(
            y_train == class_id
        )[0]

        if len(matching_indices) == 0:
            continue

        sample_index = matching_indices[0]

        image = x_train[sample_index]

        column = columns[
            class_id % 5
        ]

        with column:
            st.image(
                image,
                caption=(
                    f"{class_id} — "
                    f"{class_name}"
                ),
                use_container_width=True,
            )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def _render_class_distribution(
    distribution: pd.DataFrame,
) -> None:
    """
    Exibe a distribuição de registros entre as classes.
    """

    st.subheader("Class Distribution")

    chart_data = (
        distribution[
            [
                "class_name",
                "records",
            ]
        ]
        .set_index("class_name")
    )

    st.bar_chart(
        chart_data,
        use_container_width=True,
    )

    distribution_table = (
        distribution.copy()
    )

    distribution_table[
        "percentage"
    ] = distribution_table[
        "percentage"
    ].map(
        lambda value: f"{value:.2%}"
    )

    st.dataframe(
        distribution_table[
            [
                "class_id",
                "class_name",
                "records",
                "percentage",
            ]
        ],
        hide_index=True,
        use_container_width=True,
        column_config={
            "class_id":
                st.column_config.NumberColumn(
                    "Class ID"
                ),
            "class_name":
                st.column_config.TextColumn(
                    "Class"
                ),
            "records":
                st.column_config.NumberColumn(
                    "Records"
                ),
            "percentage":
                st.column_config.TextColumn(
                    "Dataset Share"
                ),
        },
    )


# ============================================================
# DATASET INSIGHTS
# ============================================================

def _render_dataset_insights(
    distribution: pd.DataFrame,
) -> None:
    """
    Apresenta insights que conectam os dados
    aos experimentos posteriores.
    """

    st.subheader("Dataset Insights")

    max_records = int(
        distribution["records"].max()
    )

    min_records = int(
        distribution["records"].min()
    )

    is_balanced = (
        max_records == min_records
    )

    if is_balanced:
        balance_message = (
            "A distribuição entre as classes é balanceada, "
            "reduzindo a necessidade de técnicas específicas "
            "de reamostragem para corrigir class imbalance."
        )
    else:
        balance_message = (
            "Existem diferenças na quantidade de registros "
            "entre as classes, o que deve ser considerado "
            "durante a avaliação do modelo."
        )

    st.markdown(
        f"""
        - **Balanced Target:** {balance_message}

        - **Standardized Input:** as imagens seguem uma representação
          visual altamente padronizada, com resolução fixa e
          background consistente.

        - **Low-dimensional Visual Domain:** apesar de cada imagem
          possuir 784 pixels, o domínio visual é significativamente
          mais simples que fotografias reais.

        - **Domain Gap:** essa padronização explica parte da diferença
          observada entre o desempenho no Clean Benchmark e o
          comportamento em imagens externas.
        """
    )

    st.info(
        "O Dataset fornece o contexto necessário para interpretar "
        "os experimentos seguintes: Benchmark, Robustness e "
        "Domain Shift."
    )


# ============================================================
# PAGE
# ============================================================

def render_dataset() -> None:
    """
    Renderiza a página Dataset.

    A página atua como ponto inicial da narrativa técnica,
    apresentando as características dos dados antes da
    avaliação dos modelos.
    """

    st.header(
        "Fashion-MNIST Dataset"
    )

    st.caption(
        "Características, estrutura e distribuição dos dados "
        "utilizados na etapa de modelagem."
    )

    try:
        (
            x_train,
            y_train,
            x_test,
            y_test,
        ) = load_dataset()

    except Exception as exc:
        st.error(
            "Não foi possível carregar o dataset."
        )

        st.code(str(exc))

        return

    dataset_index = build_dataset_index(
        y_train=y_train,
        y_test=y_test,
    )

    distribution = build_class_distribution(
        dataset_index
    )

    # --------------------------------------------------------
    # Dataset Overview
    # --------------------------------------------------------

    _render_dataset_overview(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    st.divider()

    # --------------------------------------------------------
    # Dataset Summary
    # --------------------------------------------------------

    _render_dataset_summary()

    st.divider()

    # --------------------------------------------------------
    # Class Samples
    # --------------------------------------------------------

    _render_image_gallery(
        x_train=x_train,
        y_train=y_train,
    )

    st.divider()

    # --------------------------------------------------------
    # Records
    # --------------------------------------------------------

    _render_record_explorer(
        dataset_index
    )

    st.divider()

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    _render_class_distribution(
        distribution
    )

    st.divider()


    # --------------------------------------------------------
    # Insights
    # --------------------------------------------------------

    _render_dataset_insights(
        distribution
    )

    st.divider()

    # --------------------------------------------------------
    # Why This Matters
    # --------------------------------------------------------

    _render_why_this_matters()



# ============================================================
# WHY THIS MATTERS
# ============================================================

def _render_why_this_matters() -> None:
    """
    Conecta as características do dataset às etapas seguintes
    da narrativa experimental do projeto.
    """

    st.subheader("Why This Matters")

    st.markdown(
        """
        O **Fashion-MNIST** fornece um domínio visual controlado e
        adequado para comparar diferentes modelos de classificação.

        Essa padronização favorece uma avaliação consistente no
        **Clean Benchmark**, mas também cria uma diferença importante
        em relação a imagens encontradas fora do domínio original.

        Por isso, um bom resultado no conjunto de teste não é suficiente
        para concluir que o modelo apresenta boa capacidade de
        generalização.

        Ao longo deste estudo, essa questão é investigada
        progressivamente por meio de **Robustness**, **Domain Shift**
        e avaliação com imagens externas.
        """
    )

    st.info(
        "A pergunta deixa de ser apenas "
        "\"qual modelo possui maior acurácia?\" e passa a incluir "
        "\"até onde podemos confiar nesse desempenho quando os dados mudam?\""
    )

    st.markdown(
        """
        **Storytelling Flow**

        `Dataset`
        → `Benchmark`
        → `Robustness`
        → `Domain Shift`
        → `External Evaluation`
        → `Domain Adaptation`
        """
    )