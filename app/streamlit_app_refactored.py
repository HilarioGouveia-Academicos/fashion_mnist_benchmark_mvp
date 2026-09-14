import requests
import pandas as pd
import streamlit as st
from PIL import Image

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Fashion-MNIST Benchmark",
    page_icon="👕",
    layout="wide",
)

st.title("👕 Fashion-MNIST Benchmark")
st.caption("SVM × MLP × CNN | MLflow | FastAPI | Streamlit")

page = st.sidebar.radio(
    "Navegação",
    ["Overview", "Inference", "Benchmark", "Robustness"],
)

if page == "Overview":
    st.header("Visão geral")

    a, b, c = st.columns(3)
    a.metric("Classes", "10")
    b.metric("Imagem", "28 × 28")
    c.metric("Modelos", "SVM / MLP / CNN")

    st.markdown(
        "**Pipeline:** Dataset → preprocessing → modelos → MLflow "
        "→ OOD → FastAPI → Streamlit."
    )

elif page == "Inference":
    st.header("🔮 Inferência")
    st.caption(
        "Classificação de imagem com diagnóstico de incerteza "
        "e sinalização de domínio."
    )

    controls = st.columns([1, 1, 2])

    with controls[0]:
        model = st.selectbox(
            "Modelo",
            ["cnn", "mlp", "svm"],
            index=0,
        )

    with controls[1]:
        preprocessing = st.selectbox(
            "Preprocessing",
            ["canonicalized", "raw"],
            index=0,
        )

    uploaded = st.file_uploader(
        "Envie uma imagem",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded:
        image = Image.open(uploaded)

        st.subheader("Imagem enviada")
        st.image(
            image,
            caption="Imagem original",
            width=320,
        )

        if st.button(
            "Classificar",
            type="primary",
        ):
            try:
                with st.spinner("Executando inferência..."):
                    response = requests.post(
                        f"{API_URL}/predict",
                        params={
                            "model": model,
                            "preprocessing": preprocessing,
                            "top_k": 10,
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
                    st.error(response.text)

                else:
                    result = response.json()

                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    top_k = result["top_k"]

                    uncertainty = result.get("uncertainty", {})
                    input_info = result.get("input", {})
                    reliability = result.get("reliability", {})

                    st.divider()

                    left, middle, right = st.columns([1.05, 1.0, 1.5])

                    with left:
                        st.subheader("Entrada do modelo")
                        st.metric("Input", "28 × 28")
                        st.write(
                            "**Preprocessing:** "
                            f"{'Canonicalized' if input_info.get('canonicalized') else 'RAW'}"
                        )

                        original_size = input_info.get("original_size")
                        if original_size:
                            st.write(
                                "**Imagem original:** "
                                f"{original_size[0]} × {original_size[1]}"
                            )

                        st.info(
                            "A API processa a imagem antes da inferência. "
                            "A visualização do tensor 28×28 poderá ser "
                            "adicionada na próxima etapa."
                        )

                    with middle:
                        st.subheader("Predição")
                        st.success(prediction)
                        st.metric("Confidence", f"{confidence:.2%}")

                        margin = uncertainty.get("prediction_margin")
                        entropy = uncertainty.get("normalized_entropy")

                        if margin is not None:
                            st.metric("Prediction margin", f"{margin:.2%}")

                        if entropy is not None:
                            st.metric("Normalized entropy", f"{entropy:.3f}")

                    with right:
                        st.subheader("Probabilidades por classe")

                        probabilities_df = pd.DataFrame(top_k).rename(
                            columns={
                                "class": "Classe",
                                "probability": "Probabilidade",
                            }
                        )

                        probabilities_df["Probabilidade"] = (
                            probabilities_df["Probabilidade"].astype(float)
                        )

                        chart_df = probabilities_df.set_index("Classe")
                        st.bar_chart(chart_df, horizontal=True)

                        formatted_df = probabilities_df.copy()
                        formatted_df["Probabilidade"] = formatted_df[
                            "Probabilidade"
                        ].map(lambda value: f"{value:.2%}")

                        st.dataframe(
                            formatted_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                    st.divider()
                    st.subheader("🛡️ Reliability")

                    r1, r2, r3 = st.columns(3)

                    with r1:
                        st.metric(
                            "Confidence level",
                            reliability.get("level", "n/a"),
                        )

                    with r2:
                        raw_entropy = uncertainty.get("entropy")
                        st.metric(
                            "Entropy",
                            f"{raw_entropy:.3f}" if raw_entropy is not None else "n/a",
                        )

                    with r3:
                        st.metric(
                            "Domain warning",
                            "Ativo"
                            if reliability.get("domain_warning", False)
                            else "Inativo",
                        )

                    if reliability.get("domain_warning", False):
                        st.warning(
                            reliability.get(
                                "message",
                                (
                                    "Imagem externa: alta confiança "
                                    "não garante alta confiabilidade."
                                ),
                            )
                        )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Não foi possível conectar à API. "
                    "Verifique se o FastAPI está em execução em "
                    "http://localhost:8000."
                )

            except requests.exceptions.Timeout:
                st.error("A API excedeu o tempo limite de resposta.")

            except Exception as exc:
                st.exception(exc)

elif page == "Benchmark":
    st.header("🏆 Benchmark")
    try:
        df = pd.read_csv("artifacts/benchmark.csv")
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("model")[["accuracy", "f1_macro"]])
    except FileNotFoundError:
        st.info("Execute: python -m src.evaluation.benchmark")

elif page == "Robustness":
    st.header("🧪 Robustness / OOD")
    try:
        df = pd.read_csv("artifacts/robustness.csv")
        st.dataframe(df, use_container_width=True)
        st.line_chart(
            df.pivot(
                index="transform",
                columns="model",
                values="accuracy",
            )
        )
    except FileNotFoundError:
        st.info("Execute: python -m src.robustness.stress_test")
