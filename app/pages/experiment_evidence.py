from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "mlflow.db"
SNAPSHOT_DB = PROJECT_ROOT / "deployment" / "mlflow.db"
MLRUNS_ROOT = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "Fashion-MNIST-Benchmark"


CURATED_STAGE_RULES = [
    ("Clean Benchmark", ("svm_baseline", "mlp_baseline", "cnn_baseline", "benchmark_svm", "benchmark_mlp", "benchmark_cnn")),
    ("Robustness", ("robust",)),
    ("Controlled Domain Shift", ("domain_shift",)),
    ("V2B RAW", ("v2b_raw",)),
    ("V2B + Canonicalizer V2", ("canonical_v2",)),
    ("V2B + Canonicalizer V3", ("canonical_v3",)),
    ("EDA Enriched", ("eda_", "eda_enriched")),
    ("Domain Adaptation", ("adapt", "finetun", "fine_tun")),
]


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .evidence-hero {
            padding: 1.25rem 1.4rem;
            border: 1px solid rgba(128,128,128,.22);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(125,125,125,.07), rgba(125,125,125,.02));
            margin-bottom: 1rem;
        }
        .evidence-kicker {font-size:.76rem; letter-spacing:.11em; text-transform:uppercase; opacity:.68;}
        .evidence-title {font-size:1.8rem; font-weight:760; margin:.15rem 0 .25rem 0;}
        .evidence-copy {opacity:.78; max-width:920px;}
        .evidence-chip {
            display:inline-block; border:1px solid rgba(128,128,128,.26); border-radius:999px;
            padding:.22rem .5rem; margin:.22rem .2rem 0 0; font-size:.72rem; font-weight:650;
            background:rgba(128,128,128,.06);
        }
        .evidence-card {
            border:1px solid rgba(128,128,128,.22); border-radius:14px; padding:.85rem .95rem;
            min-height:118px; background:rgba(128,128,128,.025);
        }
        .evidence-card b {font-size:.92rem;}
        .evidence-muted {opacity:.68; font-size:.82rem;}
        .run-id {font-family:monospace; font-size:.77rem; opacity:.78; word-break:break-all;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _fmt_time(ms: int | float | None) -> str:
    if not ms:
        return "—"
    try:
        return datetime.fromtimestamp(float(ms) / 1000).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "—"


def _stage_for_run(name: str) -> str:
    lname = (name or "").lower()
    # More specific V2B rules must win over generic domain_shift.
    if "v2b_raw" in lname:
        return "V2B RAW"
    if "canonical_v2" in lname:
        return "V2B + Canonicalizer V2"
    if "canonical_v3" in lname:
        return "V2B + Canonicalizer V3"
    if "robust" in lname:
        return "Robustness"
    if "eda_" in lname or "eda_enriched" in lname:
        return "EDA Enriched"
    if "adapt" in lname or "finetun" in lname or "fine_tun" in lname:
        return "Domain Adaptation"
    if "domain_shift" in lname:
        return "Controlled Domain Shift"
    if any(k in lname for k in ("baseline", "benchmark_svm", "benchmark_mlp", "benchmark_cnn")):
        return "Clean Benchmark"
    return "Other / Exploratory"


def _model_for_run(name: str, params: dict[str, str]) -> str:
    if params.get("model"):
        return str(params["model"]).upper()
    lname = (name or "").lower()
    for model in ("cnn", "mlp", "svm"):
        if model in lname:
            return model.upper()
    if "comparison" in lname:
        return "COMPARISON"
    return "—"


def _load_local_sqlite() -> tuple[pd.DataFrame, dict[str, dict[str, Any]], str]:
    db_path = DEFAULT_DB if DEFAULT_DB.exists() else SNAPSHOT_DB
    if not db_path.exists():
        return pd.DataFrame(), {}, "Local MLflow database not found."

    con = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
    try:
        exp = pd.read_sql_query(
            "SELECT experiment_id, name FROM experiments WHERE lifecycle_stage='active'",
            con,
        )
        target = exp[exp["name"] == EXPERIMENT_NAME]
        if target.empty:
            return pd.DataFrame(), {}, f"Experiment '{EXPERIMENT_NAME}' not found."
        experiment_id = int(target.iloc[0]["experiment_id"])

        runs = pd.read_sql_query(
            """
            SELECT run_uuid AS run_id, name AS run_name, status, start_time, end_time,
                   artifact_uri, lifecycle_stage
            FROM runs
            WHERE experiment_id = ? AND lifecycle_stage = 'active'
            ORDER BY start_time DESC
            """,
            con,
            params=(experiment_id,),
        )
        metrics = pd.read_sql_query(
            "SELECT run_uuid AS run_id, key, value FROM latest_metrics",
            con,
        )
        params_df = pd.read_sql_query(
            "SELECT run_uuid AS run_id, key, value FROM params",
            con,
        )
        tags_df = pd.read_sql_query(
            "SELECT run_uuid AS run_id, key, value FROM tags",
            con,
        )

        details: dict[str, dict[str, Any]] = {}
        for rid in runs["run_id"].tolist():
            m = metrics[metrics["run_id"] == rid]
            p = params_df[params_df["run_id"] == rid]
            t = tags_df[tags_df["run_id"] == rid]
            details[rid] = {
                "metrics": dict(zip(m["key"], m["value"])),
                "params": dict(zip(p["key"], p["value"])),
                "tags": dict(zip(t["key"], t["value"])),
            }

        runs["stage"] = runs["run_name"].map(_stage_for_run)
        runs["model"] = runs.apply(
            lambda row: _model_for_run(row["run_name"], details.get(row["run_id"], {}).get("params", {})), axis=1
        )
        runs["started_at"] = runs["start_time"].map(_fmt_time)
        label = "Deployment snapshot (read-only)" if db_path == SNAPSHOT_DB else "SQLite (read-only)"
        return runs, details, f"{label} · {db_path.name}"
    finally:
        con.close()


def _load_remote_mlflow(uri: str) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], str]:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return pd.DataFrame(), {}, f"Experiment '{EXPERIMENT_NAME}' not found at remote tracking URI."

    search_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id], output_format="pandas")
    if search_df.empty:
        return pd.DataFrame(), {}, f"Remote MLflow · {uri}"

    rename = {
        "run_id": "run_id",
        "tags.mlflow.runName": "run_name",
        "status": "status",
        "start_time": "start_time",
        "end_time": "end_time",
        "artifact_uri": "artifact_uri",
    }
    columns = [c for c in rename if c in search_df.columns]
    runs = search_df[columns].rename(columns=rename).copy()
    if "run_name" not in runs.columns:
        runs["run_name"] = runs["run_id"]
    if "start_time" in runs.columns:
        runs["started_at"] = pd.to_datetime(runs["start_time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        runs["started_at"] = "—"

    details: dict[str, dict[str, Any]] = {}
    for rid in runs["run_id"].tolist():
        run = client.get_run(rid)
        details[rid] = {
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
            "tags": dict(run.data.tags),
        }
    runs["stage"] = runs["run_name"].map(_stage_for_run)
    runs["model"] = runs.apply(
        lambda row: _model_for_run(row["run_name"], details.get(row["run_id"], {}).get("params", {})), axis=1
    )
    return runs, details, f"Remote MLflow · {uri}"


@st.cache_data(ttl=60)
def _load_evidence() -> tuple[pd.DataFrame, dict[str, dict[str, Any]], str, str]:
    remote_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if remote_uri and not remote_uri.startswith("sqlite:"):
        try:
            runs, details, source = _load_remote_mlflow(remote_uri)
            return runs, details, source, "remote"
        except Exception as exc:
            runs, details, source = _load_local_sqlite()
            return runs, details, f"{source} · remote fallback: {type(exc).__name__}", "local-fallback"
    runs, details, source = _load_local_sqlite()
    return runs, details, source, "local"


def _artifact_files(run_id: str) -> list[Path]:
    root = MLRUNS_ROOT / "1" / run_id / "artifacts"
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file()])


def _curated_runs(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return runs
    preferred = runs[runs["stage"] != "Other / Exploratory"].copy()
    # Drop obvious comparison-only runs from curated detail list; model-specific runs tell the story better.
    return preferred[~preferred["run_name"].str.contains("comparison", case=False, na=False)]


def _render_run_detail(row: pd.Series, details: dict[str, dict[str, Any]]) -> None:
    rid = str(row["run_id"])
    data = details.get(rid, {"metrics": {}, "params": {}, "tags": {}})
    metrics = data.get("metrics", {})
    params = data.get("params", {})
    tags = data.get("tags", {})

    st.markdown(f"### {row['run_name']}")
    a, b, c, d = st.columns(4)
    a.metric("Stage", row.get("stage", "—"))
    b.metric("Model", row.get("model", "—"))
    c.metric("Status", row.get("status", "—"))
    d.metric("Metrics", len(metrics))

    st.markdown(
        f"<div class='run-id'><b>Run ID:</b> {rid}<br><b>Started:</b> {row.get('started_at','—')}<br><b>Artifact URI:</b> {row.get('artifact_uri','—')}</div>",
        unsafe_allow_html=True,
    )

    ui_url = os.getenv("MLFLOW_UI_URL", "").strip().rstrip("/")
    if ui_url:
        st.link_button("Open Full MLflow UI ↗", f"{ui_url}/#/experiments/1/runs/{rid}")

    tabs = st.tabs(["Metrics", "Parameters", "Tags", "Artifacts"])
    with tabs[0]:
        if metrics:
            mdf = pd.DataFrame([{"metric": k, "value": v} for k, v in sorted(metrics.items())])
            st.dataframe(mdf, use_container_width=True, hide_index=True)
        else:
            st.info("No metrics registered for this run.")

    with tabs[1]:
        if params:
            pdf = pd.DataFrame([{"parameter": k, "value": v} for k, v in sorted(params.items())])
            st.dataframe(pdf, use_container_width=True, hide_index=True)
        else:
            st.info("No parameters registered for this run.")

    with tabs[2]:
        visible_tags = {k: v for k, v in tags.items() if not k.startswith("mlflow.log-model")}
        if visible_tags:
            tdf = pd.DataFrame([{"tag": k, "value": v} for k, v in sorted(visible_tags.items())])
            st.dataframe(tdf, use_container_width=True, hide_index=True)
        else:
            st.info("No tags registered for this run.")

    with tabs[3]:
        files = _artifact_files(rid)
        if files:
            adf = pd.DataFrame(
                [{"artifact": str(p.relative_to(MLRUNS_ROOT / "1" / rid / "artifacts")), "size_kb": round(p.stat().st_size / 1024, 1)} for p in files]
            )
            st.dataframe(adf, use_container_width=True, hide_index=True)
        else:
            st.caption("Local artifact files are unavailable here. For a remote deployment, expose them through the MLflow Tracking Server / artifact store.")


def render_experiment_evidence() -> None:
    _inject_styles()
    runs, details, source, source_mode = _load_evidence()

    st.markdown(
        """
        <div class="evidence-hero">
          <div class="evidence-kicker">Application • MLflow-backed traceability</div>
          <div class="evidence-title">Experiment Evidence</div>
          <div class="evidence-copy">
            Uma visão curada dos runs, métricas, parâmetros, tags e artefatos que sustentam a narrativa experimental.
            Streamlit interpreta e organiza a evidência; MLflow continua sendo o system of record.
          </div>
          <div>
            <span class="evidence-chip">MLflow-backed</span>
            <span class="evidence-chip">Curated Evidence</span>
            <span class="evidence-chip">All Runs</span>
            <span class="evidence-chip">Traceability</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if runs.empty:
        st.warning(f"No MLflow evidence available. Source: {source}")
        return

    status_col, count_col, curated_col, mode_col = st.columns(4)
    status_col.metric("Experiment", EXPERIMENT_NAME)
    count_col.metric("Runs", len(runs))
    curated_col.metric("Curated Runs", len(_curated_runs(runs)))
    mode_col.metric("Source", "Remote" if source_mode == "remote" else "Local")
    st.caption(f"Evidence source: {source}")

    mode = st.segmented_control(
        "Evidence view",
        options=["Curated Evidence", "All Runs"],
        default="Curated Evidence",
    )

    view = _curated_runs(runs) if mode == "Curated Evidence" else runs.copy()

    stages = ["All"] + [s for s in [
        "Clean Benchmark", "Robustness", "Controlled Domain Shift", "V2B RAW",
        "V2B + Canonicalizer V2", "V2B + Canonicalizer V3", "EDA Enriched",
        "Domain Adaptation", "Other / Exploratory"
    ] if s in set(view["stage"])]
    models = ["All"] + sorted([m for m in view["model"].dropna().unique().tolist() if m != "—"])

    f1, f2, f3 = st.columns([1.35, 1, 1])
    selected_stage = f1.selectbox("Stage", stages)
    selected_model = f2.selectbox("Model", models)
    selected_status = f3.selectbox("Status", ["All"] + sorted(view["status"].dropna().unique().tolist()))

    filtered = view.copy()
    if selected_stage != "All":
        filtered = filtered[filtered["stage"] == selected_stage]
    if selected_model != "All":
        filtered = filtered[filtered["model"] == selected_model]
    if selected_status != "All":
        filtered = filtered[filtered["status"] == selected_status]

    st.markdown("#### Evidence Index")
    table = filtered[["stage", "run_name", "model", "status", "started_at", "run_id"]].copy()
    table.columns = ["Stage", "Run", "Model", "Status", "Started", "Run ID"]
    st.dataframe(table, use_container_width=True, hide_index=True)

    if filtered.empty:
        st.info("No runs match the selected filters.")
        return

    labels = [f"{r.run_name} · {r.model} · {r.run_id[:8]}" for r in filtered.itertuples()]
    selected_label = st.selectbox("Inspect run", labels)
    selected_idx = labels.index(selected_label)
    row = filtered.iloc[selected_idx]

    st.divider()
    _render_run_detail(row, details)

    st.divider()
    st.markdown("### Architecture Note")
    st.markdown(
        """
        **Local development:** this page can read the bundled `mlflow.db` and local `mlruns/` artifacts.  
        **Cloud deployment:** the bundled database snapshot provides read-only experiment history.
        A remote Tracking Server is optional via `MLFLOW_TRACKING_URI`. Optionally set `MLFLOW_UI_URL`
        to expose the **Open Full MLflow UI** button. The Streamlit page does not need the full MLflow UI to remain useful.
        """
    )

    st.caption("Streamlit tells the story. MLflow preserves the evidence.")
