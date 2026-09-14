from __future__ import annotations

import io
import json
from datetime import datetime

import streamlit as st
from PIL import Image

from src.acquisition.contracts import InferenceInput
from src.acquisition.image_transformer import build_representations


SESSION_KEY = "inference_input"


def _metadata_json(metadata: dict) -> bytes:
    return json.dumps(
        {
            "schema_version": "1.0",
            **metadata,
        },
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")


def _safe_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _render_header() -> None:
    st.header("📷 CV Capture Lab")
    st.caption("Image Acquisition & Consumption • Experimental Computer Vision")

    st.warning(
        "EXPERIMENTAL — esta feature ainda não faz parte do pipeline de "
        "inferência homologado. Capturas não são adicionadas automaticamente "
        "a V2B, EAS, treino ou benchmark."
    )

    st.markdown(
        """
        **Capture → Transform → Inspect → Export → Transfer**

        O objetivo desta página é validar aquisição e representação visual antes
        de promover qualquer capacidade de câmera para a Inference Page.
        """
    )


def _render_diagnostics(metadata: dict) -> None:
    diagnostics = metadata.get("diagnostics") or {}
    if not diagnostics:
        return

    st.subheader("CV Diagnostics")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Foreground occupancy",
            f"{diagnostics.get('foreground_occupancy', 0.0):.1%}",
        )
    with c2:
        st.metric(
            "Background ratio",
            f"{diagnostics.get('background_ratio', 0.0):.1%}",
        )
    with c3:
        st.metric(
            "GrabCut status",
            diagnostics.get("grabcut_status", "n/a"),
        )
    with c4:
        ratio = diagnostics.get("width_height_ratio")
        st.metric("W/H ratio", f"{ratio:.2f}" if ratio is not None else "n/a")

    bbox = diagnostics.get("bounding_box")
    center = diagnostics.get("object_center_normalized")
    if bbox:
        st.caption(f"Bounding box: {bbox}")
    if center:
        st.caption(f"Object center (normalized): ({center[0]:.3f}, {center[1]:.3f})")

    st.caption(
        "Esses valores são diagnósticos de CV; não constituem evidência de "
        "preservação semântica nem de confiabilidade do modelo."
    )


def render_cv_capture_lab() -> None:
    _render_header()
    st.divider()

    capture = st.camera_input("Capture uma imagem")

    if capture is None:
        st.info("Autorize a câmera e faça uma captura para iniciar o experimento.")
        return

    try:
        original_image = Image.open(io.BytesIO(capture.getvalue()))
        representations = build_representations(original_image)
    except Exception as exc:
        st.error("Não foi possível gerar as representações da captura.")
        st.caption(str(exc))
        return

    st.subheader("Visual Comparison")
    cols = st.columns(4)
    order = [
        "original",
        "raw_28x28",
        "canonicalized_v2",
        "canonicalized_v3",
    ]

    for col, key in zip(cols, order):
        rep = representations[key]
        with col:
            if rep.image is not None:
                st.image(rep.image, caption=rep.label, use_container_width=True)
            else:
                st.warning(f"{rep.label} indisponível")
                if rep.error:
                    st.caption(rep.error)

    v3 = representations["canonicalized_v3"]
    if v3.mask is not None or v3.roi is not None:
        with st.expander("Inspect V3 / GrabCut intermediates"):
            c1, c2 = st.columns(2)
            if v3.mask is not None:
                c1.image(v3.mask, caption="GrabCut mask", use_container_width=True)
            if v3.roi is not None:
                c2.image(v3.roi, caption="Foreground ROI", use_container_width=True)

    _render_diagnostics(v3.metadata)

    st.divider()
    st.subheader("Export & Transfer")

    labels = {
        rep.label: key
        for key, rep in representations.items()
        if rep.png_bytes
    }
    selected_label = st.selectbox("Representação", list(labels.keys()), index=2)
    selected_key = labels[selected_label]
    selected = representations[selected_key]

    timestamp = _safe_timestamp()
    suffix = {
        "original": "original",
        "raw_28x28": "raw_28x28",
        "canonicalized_v2": "v2",
        "canonicalized_v3": "v3",
    }[selected_key]

    d1, d2, d3 = st.columns([1, 1, 1.1])

    with d1:
        st.download_button(
            "Download PNG",
            data=selected.png_bytes,
            file_name=f"capture_{timestamp}_{suffix}.png",
            mime="image/png",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            "Download metadata JSON",
            data=_metadata_json(selected.metadata),
            file_name=f"capture_{timestamp}_{suffix}_metadata.json",
            mime="application/json",
            use_container_width=True,
        )

    with d3:
        if st.button("Send to Inference", type="primary", use_container_width=True):
            contract = InferenceInput(
                image_bytes=selected.png_bytes,
                filename=f"capture_{timestamp}_{suffix}.png",
                mime_type="image/png",
                source="cv_capture_lab",
                acquisition="camera",
                representation=selected_key,
                canonicalization_applied=bool(
                    selected.metadata.get("canonicalization_applied", False)
                ),
                canonicalization_version=selected.metadata.get(
                    "canonicalization_version"
                ),
                model_input_ready=False,
                metadata=selected.metadata,
            )
            st.session_state[SESSION_KEY] = contract.to_session_payload()
            st.success(
                "Imagem enviada para o contrato de Inference. Abra a página "
                "Inference; a origem e o estado da representação serão preservados."
            )

    st.caption(
        "O download permite reuso manual. Send to Inference usa o mesmo contrato "
        "de domínio, mantendo provenance e evitando canonicalização duplicada."
    )
