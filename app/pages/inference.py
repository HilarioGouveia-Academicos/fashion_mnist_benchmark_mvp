import base64
import io
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

from PIL import Image

from src.acquisition.contracts import InferenceInput


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://localhost:8000"

MODEL_OPTIONS = [
    "cnn",
    "mlp",
    "svm",
]

TRANSFER_SESSION_KEY = "inference_input"

PREPROCESSING_OPTIONS = {
    "Canonicalized V2": "canonicalized_v2",
    "Canonicalized V3 (GrabCut)": "canonicalized_v3",
    "RAW": "raw",
}


# ============================================================
# CONTEXT
# ============================================================

def _render_inference_context() -> None:
    """
    Posiciona a inferência como etapa operacional da narrativa.
    """

    st.subheader("Inference Context")

    st.markdown(
        """
        As etapas anteriores avaliaram o comportamento dos modelos
        em nível experimental.

        Nesta página, a análise passa para o nível de uma
        **predição individual**.

        A imagem enviada é processada pela **FastAPI**, que aplica
        o preprocessing selecionado, executa o modelo e retorna
        a classificação juntamente com indicadores de incerteza.
        """
    )

    st.info(
        "Pergunta desta etapa: como as evidências de reliability "
        "aparecem em uma predição individual?"
    )


# ============================================================
# CONTROLS
# ============================================================

def _render_controls(
    transferred: InferenceInput | None = None,
):
    """Renderiza os controles da inferência.

    Representações já preparadas pelo CV Capture Lab preservam seu estado e
    bloqueiam a seleção de preprocessing para impedir dupla canonicalização.
    """

    controls = st.columns([1, 1, 2])

    with controls[0]:
        model = st.selectbox(
            "Modelo",
            MODEL_OPTIONS,
            index=0,
        )

    recommended = (
        transferred.recommended_preprocessing
        if transferred is not None
        else None
    )

    with controls[1]:
        if recommended:
            label = next(
                label
                for label, value in PREPROCESSING_OPTIONS.items()
                if value == recommended
            )
            st.selectbox(
                "Preprocessing",
                [label],
                index=0,
                disabled=True,
                help=(
                    "A representação recebida do CV Capture Lab já possui "
                    "estado de preprocessing definido."
                ),
            )
            preprocessing = recommended
        else:
            preprocessing_label = st.selectbox(
                "Preprocessing",
                list(PREPROCESSING_OPTIONS.keys()),
                index=0,
            )
            preprocessing = PREPROCESSING_OPTIONS[preprocessing_label]

    return model, preprocessing


# ============================================================
# FILE UPLOAD
# ============================================================

@dataclass
class _InputFile:
    name: str
    type: str
    data: bytes
    representation: str = "original"
    source: str = "upload"

    def getvalue(self) -> bytes:
        return self.data


def _load_transferred_input() -> InferenceInput | None:
    payload = st.session_state.get(TRANSFER_SESSION_KEY)
    if not payload:
        return None

    try:
        return InferenceInput.from_session_payload(payload)
    except Exception as exc:
        st.warning("O input transferido era inválido e foi descartado.")
        st.caption(str(exc))
        st.session_state.pop(TRANSFER_SESSION_KEY, None)
        return None


def _render_input_source(
    transferred: InferenceInput | None,
):
    """Resolve CV Capture Lab transfer first, otherwise normal file upload."""

    if transferred is not None:
        st.subheader("Input Source")
        st.info(
            "Imagem recebida do CV Capture Lab com provenance e estado de "
            "representação preservados."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Source", transferred.source)
        c2.metric("Acquisition", transferred.acquisition)
        c3.metric("Representation", transferred.representation)

        if st.button("Descartar e usar upload"):
            st.session_state.pop(TRANSFER_SESSION_KEY, None)
            st.rerun()

        return _InputFile(
            name=transferred.filename,
            type=transferred.mime_type,
            data=transferred.image_bytes,
            representation=transferred.representation,
            source=transferred.source,
        )

    uploaded = st.file_uploader(
        "Envie uma imagem",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded is None:
        return None

    return _InputFile(
        name=uploaded.name,
        type=uploaded.type,
        data=uploaded.getvalue(),
        representation="original",
        source="upload",
    )


# ============================================================
# INPUT PREVIEW
# ============================================================

def _render_input_preview(
    uploaded,
) -> Image.Image:
    """
    Exibe a imagem original antes da inferência.
    """

    image = Image.open(
        io.BytesIO(
            uploaded.getvalue()
        )
    )

    st.subheader("Input Preview")

    st.image(
        image,
        caption="Imagem original",
        width=320,
    )

    return image


# ============================================================
# API REQUEST
# ============================================================

def _request_prediction(
    uploaded: _InputFile,
    model: str,
    preprocessing: str,
) -> dict:
    """
    Envia a imagem à FastAPI e retorna o resultado.
    """

    response = requests.post(
        f"{API_URL}/predict",
        params={
            "model": model,
            "preprocessing": preprocessing,
            "top_k": 10,
            "input_representation": uploaded.representation,
        },
        files={
            "file": (
                uploaded.name,
                uploaded.getvalue(),
                uploaded.type,
            )
        },
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(
            response.text
        )

    return response.json()


# ============================================================
# INPUT INFORMATION
# ============================================================

def _render_processed_preview(
    input_info: dict,
) -> None:
    """Render the exact 28x28 representation produced by the serving layer."""

    st.subheader("Processed Input")

    encoded = input_info.get(
        "processed_preview_png_base64"
    )

    if not encoded:
        st.info(
            "A API não retornou a prévia da imagem processada."
        )
        return

    try:
        image_bytes = base64.b64decode(
            encoded
        )
        processed = Image.open(
            io.BytesIO(image_bytes)
        )

        st.image(
            processed,
            caption=(
                input_info.get(
                    "preprocessing_label",
                    "Model input",
                )
                + " — 28 × 28"
            ),
            width=224,
        )

        st.caption(
            "Representação produzida pela camada de serving "
            "antes do preprocessing específico do modelo."
        )

    except Exception as exc:
        st.warning(
            "Não foi possível renderizar a prévia processada."
        )
        st.caption(str(exc))


def _render_input_information(
    input_info: dict,
) -> None:
    """Exibe informações sobre a entrada processada."""

    st.subheader("Model Input")

    st.metric(
        "Input",
        "28 × 28",
    )

    preprocessing_label = input_info.get(
        "preprocessing_label"
    )

    if not preprocessing_label:
        preprocessing_label = (
            "Canonicalized"
            if input_info.get("canonicalized", False)
            else "RAW"
        )

    st.write(
        "**Preprocessing:** "
        f"{preprocessing_label}"
    )

    original_size = input_info.get(
        "original_size"
    )

    if original_size:
        st.write(
            "**Imagem original:** "
            f"{original_size[0]} × "
            f"{original_size[1]}"
        )

    version = input_info.get(
        "canonicalization_version"
    )

    if version:
        st.write(
            "**Canonicalizer:** "
            f"{version.upper()}"
        )

    st.caption(
        "A imagem é transformada pela camada de serving "
        "antes de chegar ao modelo."
    )


# ============================================================
# PREDICTION
# ============================================================

def _render_prediction(
    prediction: str,
    confidence: float,
    uncertainty: dict,
) -> None:
    """
    Apresenta a decisão principal do modelo.
    """

    st.subheader("Prediction")

    st.success(
        prediction
    )

    st.metric(
        "Confidence",
        f"{confidence:.2%}",
    )

    margin = uncertainty.get(
        "prediction_margin"
    )

    normalized_entropy = uncertainty.get(
        "normalized_entropy"
    )

    if margin is not None:
        st.metric(
            "Prediction Margin",
            f"{margin:.2%}",
        )

    if normalized_entropy is not None:
        st.metric(
            "Normalized Entropy",
            f"{normalized_entropy:.3f}",
        )


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

def _render_probabilities(
    top_k: list,
) -> None:
    """Exibe as probabilidades das 10 classes em ordem decrescente."""

    st.subheader(
        "Class Probabilities"
    )

    probabilities_df = pd.DataFrame(
        top_k
    ).rename(
        columns={
            "rank": "Rank",
            "class": "Classe",
            "probability": "Probabilidade",
        }
    )

    if "Rank" not in probabilities_df.columns:
        probabilities_df["Rank"] = range(
            1,
            len(probabilities_df) + 1,
        )

    probabilities_df[
        "Probabilidade"
    ] = probabilities_df[
        "Probabilidade"
    ].astype(float)

    # Never rely on API or chart-library ordering.
    probabilities_df = (
        probabilities_df
        .sort_values(
            "Probabilidade",
            ascending=False,
        )
        .reset_index(drop=True)
    )
    probabilities_df["Rank"] = range(
        1,
        len(probabilities_df) + 1,
    )

    fig, ax = plt.subplots(
        figsize=(6.4, 4.6)
    )
    ax.barh(
        probabilities_df["Classe"],
        probabilities_df["Probabilidade"],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    ax.set_xlim(
        0,
        max(
            1.0,
            float(
                probabilities_df["Probabilidade"].max()
            ) * 1.05,
        ),
    )
    ax.set_title(
        "Descending probability ranking"
    )
    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True,
    )
    plt.close(fig)

    formatted_df = probabilities_df.copy()
    formatted_df[
        "Probabilidade"
    ] = formatted_df[
        "Probabilidade"
    ].map(
        lambda value: f"{value:.2%}"
    )

    st.dataframe(
        formatted_df[
            ["Rank", "Classe", "Probabilidade"]
        ],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# RELIABILITY
# ============================================================

def _render_reliability(
    uncertainty: dict,
    reliability: dict,
) -> None:
    """
    Exibe sinais de uncertainty e reliability.
    """

    st.subheader(
        "🛡️ Reliability"
    )

    r1, r2, r3 = st.columns(3)

    level = reliability.get(
        "level",
        "n/a",
    )

    raw_entropy = uncertainty.get(
        "entropy"
    )

    domain_warning = reliability.get(
        "domain_warning",
        False,
    )

    with r1:
        st.metric(
            "Confidence Level",
            level,
        )

    with r2:
        st.metric(
            "Entropy",
            (
                f"{raw_entropy:.3f}"
                if raw_entropy is not None
                else "n/a"
            ),
        )

    with r3:
        st.metric(
            "Domain Warning",
            (
                "Ativo"
                if domain_warning
                else "Inativo"
            ),
        )

    if domain_warning:
        st.warning(
            reliability.get(
                "message",
                (
                    "Imagem externa: alta confiança "
                    "não garante alta confiabilidade."
                ),
            )
        )

    st.caption(
        "Confidence e entropy são sinais auxiliares. "
        "Eles não constituem, isoladamente, uma medida "
        "calibrada de confiabilidade."
    )


# ============================================================
# INTERPRETATION
# ============================================================

def _render_inference_insight(
    confidence: float,
    uncertainty: dict,
    reliability: dict,
) -> None:
    """
    Conecta a predição individual às evidências experimentais.
    """

    st.subheader(
        "Inference Insight"
    )

    entropy = uncertainty.get(
        "normalized_entropy"
    )

    domain_warning = reliability.get(
        "domain_warning",
        False,
    )

    st.markdown(
        """
        Uma predição individual deve ser interpretada junto com
        o comportamento observado nos experimentos de
        **Robustness** e **Domain Shift**.

        A confiança produzida pelo modelo representa sua distribuição
        interna de probabilidades, mas não garante que a entrada
        pertença ao mesmo domínio utilizado no treinamento.
        """
    )

    if domain_warning:
        st.warning(
            "A entrada recebeu um sinal de domínio. "
            "Mesmo uma predição com alta confiança deve ser "
            "interpretada com cautela."
        )

    elif confidence >= 0.90:
        st.info(
            "A predição apresenta alta confiança interna. "
            "Isso não deve ser interpretado automaticamente "
            "como evidência de generalização."
        )

    elif entropy is not None and entropy > 0.7:
        st.warning(
            "A distribuição apresenta elevada incerteza, "
            "indicando menor separação entre as classes."
        )

    else:
        st.info(
            "A predição não apresenta um sinal crítico isolado, "
            "mas sua confiabilidade continua dependente do domínio "
            "da entrada."
        )


# ============================================================
# SERVING LAYER
# ============================================================

def _render_serving_layer() -> None:
    """
    Posiciona FastAPI dentro da arquitetura MLOps.
    """

    st.subheader(
        "Serving Layer"
    )

    st.markdown(
        """
        A inferência não é executada diretamente pelo Streamlit.

        O **FastAPI** atua como camada de serving responsável por:

        `upload`
        → `validation`
        → `preprocessing V2/V3`
        → `model inference`
        → `uncertainty`
        → `response`
        """
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Interface",
        "REST API",
    )

    c2.metric(
        "Endpoint",
        "/predict",
    )

    c3.metric(
        "Serving",
        "FastAPI",
    )

    st.info(
        "Streamlit consome o serviço de inferência; "
        "FastAPI operacionaliza o modelo."
    )


# ============================================================
# MLOPS PERSPECTIVE
# ============================================================

def _render_mlops_perspective() -> None:
    """
    Conecta a página de inferência ao ciclo de MLOps.
    """

    st.subheader(
        "MLOps Perspective"
    )

    st.markdown(
        """
        A página de inferência demonstra a transição entre
        **experimentação** e **operacionalização**.

        **MLflow** preserva as evidências experimentais.

        **FastAPI** disponibiliza o modelo como serviço.

        **Streamlit** apresenta a interação e a interpretação
        dos resultados.

        Essa separação de responsabilidades aproxima o projeto
        de um fluxo real de Engenharia de Machine Learning.
        """
    )


# ============================================================
# PAGE
# ============================================================

def render_inference() -> None:
    """
    Renderiza a página modular de inferência.
    """

    st.header(
        "🔮 Inference"
    )

    st.caption(
        "Classificação individual com diagnóstico "
        "de uncertainty e reliability."
    )

    _render_inference_context()

    st.divider()

    transferred = _load_transferred_input()

    model, preprocessing = _render_controls(
        transferred=transferred,
    )

    uploaded = _render_input_source(
        transferred=transferred,
    )

    if uploaded is None:
        st.info(
            "Envie uma imagem para iniciar a inferência."
        )

        st.divider()

        _render_serving_layer()

        st.divider()

        _render_mlops_perspective()

        return

    st.divider()

    _render_input_preview(
        uploaded
    )

    if uploaded.source == "cv_capture_lab":
        st.caption(
            f"Provenance: CV Capture Lab • representation={uploaded.representation}. "
            "A API validará o estado antes da inferência."
        )

    if not st.button(
        "Classificar",
        type="primary",
    ):
        return

    try:
        with st.spinner(
            "Executando inferência..."
        ):
            result = _request_prediction(
                uploaded=uploaded,
                model=model,
                preprocessing=preprocessing,
            )

        prediction = result[
            "prediction"
        ]

        confidence = result[
            "confidence"
        ]

        top_k = result[
            "top_k"
        ]

        uncertainty = result.get(
            "uncertainty",
            {},
        )

        input_info = result.get(
            "input",
            {},
        )

        reliability = result.get(
            "reliability",
            {},
        )

        st.divider()

        preview_col, info_col, prediction_col = st.columns(
            [
                1.05,
                1.0,
                1.0,
            ]
        )

        with preview_col:
            _render_processed_preview(
                input_info
            )

        with info_col:
            _render_input_information(
                input_info
            )

        with prediction_col:
            _render_prediction(
                prediction,
                confidence,
                uncertainty,
            )

        st.divider()

        _render_probabilities(
            top_k
        )

        st.divider()

        _render_reliability(
            uncertainty,
            reliability,
        )

        st.divider()

        _render_inference_insight(
            confidence,
            uncertainty,
            reliability,
        )

        st.divider()

        _render_serving_layer()

        st.divider()

        _render_mlops_perspective()

    except requests.exceptions.ConnectionError:
        st.error(
            "Não foi possível conectar à API. "
            "Verifique se o FastAPI está em execução em "
            "http://localhost:8000."
        )

    except requests.exceptions.Timeout:
        st.error(
            "A API excedeu o tempo limite de resposta."
        )

    except RuntimeError as exc:
        st.error(
            "A API retornou um erro durante a inferência."
        )

        st.code(
            str(exc)
        )

    except Exception as exc:
        st.error(
            "Não foi possível concluir a inferência."
        )

        st.exception(exc)