"""Interactive trajectory dashboard.

Run with:
    uv run streamlit run src/tray/viz/dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from tray.eval.ate import compute_ate, umeyama_align

RESULTS_DIR = Path("results")
COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
    "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
    "#FF97FF", "#FECB52",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _discover_results() -> list[str]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        p.name for p in RESULTS_DIR.iterdir()
        if p.is_dir() and (p / "trajectory.npy").exists()
    )


def _load_result(name: str) -> dict:
    d = RESULTS_DIR / name
    data: dict = {
        "est": np.load(d / "trajectory.npy"),
        "gt": np.load(d / "gt_trajectory.npy") if (d / "gt_trajectory.npy").exists() else None,
        "ate_errors": np.load(d / "ate_errors.npy") if (d / "ate_errors.npy").exists() else None,
        "metrics": json.loads((d / "metrics.json").read_text()),
    }
    return data


def _positions(traj: np.ndarray) -> np.ndarray:
    return traj[:, :3, 3]


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Panel de Trayectorias", layout="wide")
st.title("Panel de Trayectorias")

available = _discover_results()
if not available:
    st.error(f"No se encontraron resultados en `{RESULTS_DIR}/`. Ejecute un experimento primero.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controles")

    selected = st.multiselect("Configuraciones", available, default=available[:1])
    if not selected:
        st.warning("Seleccione al menos una configuración.")
        st.stop()

    results = {name: _load_result(name) for name in selected}
    max_n = max(len(r["est"]) for r in results.values())

    frame_start, frame_end = st.slider(
        "Rango de fotogramas", 0, max_n, (0, max_n), step=1
    )

    show_gt = st.toggle("Mostrar GT", value=True)
    apply_umeyama = st.toggle("Alineación Umeyama", value=False)

    with_scale = False
    if apply_umeyama:
        # default scale=True if any selected config is epipolar
        any_epipolar = any(
            r["metrics"].get("pipeline") == "epipolar" for r in results.values()
        )
        with_scale = st.toggle("Con escala (epipolar)", value=any_epipolar)

# ── 3D trajectory plot ────────────────────────────────────────────────────────
fig3d = go.Figure()

live_ate: dict[str, float | None] = {}

for idx, (name, r) in enumerate(results.items()):
    color = COLORS[idx % len(COLORS)]
    est = r["est"][frame_start:frame_end]
    gt = r["gt"][frame_start:frame_end] if r["gt"] is not None else None

    if apply_umeyama and gt is not None and len(gt) >= 2:
        n = min(len(est), len(gt))
        est = umeyama_align(est[:n], gt[:n], with_scale=with_scale)
        gt = gt[:n]
        ate = compute_ate(est, gt, with_scale=False)  # already aligned
        live_ate[name] = float(ate.rmse)
    else:
        live_ate[name] = None

    pos = _positions(est)
    fig3d.add_trace(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode="lines",
        name=name,
        line=dict(color=color, width=3),
    ))

    if show_gt and gt is not None:
        gt_pos = _positions(gt)
        fig3d.add_trace(go.Scatter3d(
            x=gt_pos[:, 0], y=gt_pos[:, 1], z=gt_pos[:, 2],
            mode="lines",
            name=f"{name} (GT)",
            line=dict(color=color, width=2, dash="dash"),
            opacity=0.6,
        ))

fig3d.update_layout(
    scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"),
    legend=dict(font=dict(size=11)),
    margin=dict(l=0, r=0, t=30, b=0),
    height=500,
)
st.plotly_chart(fig3d, use_container_width=True)

# ── Metrics table ─────────────────────────────────────────────────────────────
st.subheader("Métricas")

rows = []
for name, r in results.items():
    m = r["metrics"]
    row = {
        "Configuración": name,
        "Pipeline": m.get("pipeline", "—"),
        "Fotogramas": m.get("n_frames", "—"),
        "Fotogramas GT": m.get("n_gt_frames", "—"),
        "ATE RMSE (guardado)": f"{m['ate_rmse']:.4f}" if "ate_rmse" in m else "—",
        "ATE RMSE (en vivo)": f"{live_ate[name]:.4f}" if live_ate[name] is not None else "—",
        "RTE RMSE": f"{m['rte_rmse']:.4f}" if "rte_rmse" in m else "—",
    }
    rows.append(row)

st.dataframe(rows, use_container_width=True, hide_index=True)

# ── Per-frame ATE chart ───────────────────────────────────────────────────────
ate_traces = [
    (name, r["ate_errors"])
    for name, r in results.items()
    if r["ate_errors"] is not None
]

if ate_traces:
    st.subheader("ATE por fotograma (metros)")
    fig_ate = go.Figure()
    for idx, (name, errors) in enumerate(ate_traces):
        color = COLORS[idx % len(COLORS)]
        sliced = errors[frame_start:frame_end]
        fig_ate.add_trace(go.Scatter(
            x=list(range(frame_start, frame_start + len(sliced))),
            y=sliced,
            mode="lines",
            name=name,
            line=dict(color=color, width=1.5),
        ))
    fig_ate.update_layout(
        xaxis_title="Índice de fotograma",
        yaxis_title="ATE (m)",
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(font=dict(size=11)),
    )
    st.plotly_chart(fig_ate, use_container_width=True)
else:
    st.info(
        "No se encontró `ate_errors.npy`. Vuelva a ejecutar los experimentos para generar datos de ATE por fotograma."
    )
