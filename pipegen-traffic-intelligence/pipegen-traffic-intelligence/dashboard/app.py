"""Streamlit dashboard. Reads ONLY what the pipeline wrote to outputs/ (no hard-coded stats).
Run from the project root:  streamlit run dashboard/app.py"""
import json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import streamlit as st
from core import store

APP = json.loads((ROOT / "config" / "app_config.json").read_text())
OUT = Path(os.environ.get("PIPEGEN_OUT", ROOT / APP["output_dir"]))
STALE = APP["stale_after_seconds"]
COLORS = {"Normal": "#16a34a", "Warning": "#d97706", "Critical": "#dc2626"}

st.set_page_config(page_title="PipeGen Traffic Intelligence", page_icon="🚦", layout="wide")
st.title("🚦 PipeGen Traffic Intelligence")


def badge(text, color):
    st.markdown(f"<span style='background:{color};color:white;padding:4px 12px;border-radius:12px;"
                f"font-weight:600'>{text}</span>", unsafe_allow_html=True)


def mode_badge(mode):
    m = (mode or "").lower()
    if m.startswith("simulated"):
        badge("SIMULATED DATA - not real inference", "#dc2626")
    elif m.startswith("deepstream"):
        badge(f"NVIDIA DeepStream + TensorRT", "#16a34a")
    else:
        badge(f"{mode} - not TensorRT", "#6b7280")


@st.fragment(run_every=APP["refresh_seconds"])
def live():
    m = store.read_metrics(OUT / "metrics.json")
    if m is None:
        st.info("No data yet. Start a pipeline (prototype, simulator, or DeepStream) and this page fills in "
                f"automatically. Watching: `{OUT}`")
        return
    age = time.time() - m.get("updated_epoch", 0)
    c0, c1 = st.columns([2, 3])
    with c0:
        mode_badge(m.get("source_mode"))
    with c1:
        if m.get("error"):
            st.error(f"Pipeline reported an error: {m['error']}")
        elif m.get("finished"):
            st.success("Run finished (end of video).")
        elif age > STALE:
            st.warning(f"Pipeline looks stalled: no update for {age:.0f}s.")
    status = m.get("overall_status", "Normal")
    left, right = st.columns([3, 2])
    with left:
        img = OUT / "latest.jpg"
        if img.is_file():
            try:
                st.image(img.read_bytes(), caption=f"Source: {m.get('source')}", width="stretch")
            except Exception:
                st.caption("Frame is being updated...")
        else:
            st.caption("No live frame for this mode. See the annotated video below if available.")
    with right:
        st.markdown(f"### Traffic status: <span style='color:{COLORS[status]}'>{status}</span>", unsafe_allow_html=True)
        a, b = st.columns(2)
        a.metric("Vehicles in frame", m.get("frame_vehicles", 0))
        b.metric("Line crossings (session)", m.get("total_crossings", 0))
        a.metric("Processing FPS", m.get("fps", 0))
        b.metric("Frames processed", m.get("frame_idx", 0))
        st.caption("Congestion is an ESTIMATE from vehicle counts in a region, not measured road speed.")
    k1, k2 = st.columns(2)
    with k1:
        st.subheader("Crossings by class")
        cc = m.get("class_crossings") or {}
        st.bar_chart(pd.Series(cc)) if cc else st.caption("No crossings yet.")
    with k2:
        st.subheader("Regions")
        rows = [{"Region": v["name"], "Vehicles": v["count"], "Warning at": v["warning"],
                 "Critical at": v["critical"], "Status": v["status"]} for v in (m.get("regions") or {}).values()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch") if rows else st.caption("No regions configured.")
    st.subheader("Recent alerts")
    al = store.recent_alerts(OUT / "events.db", 15)
    if al:
        df = pd.DataFrame(al)[["ts_utc", "video_time_s", "region_id", "previous", "severity", "count", "rule", "source_mode"]]
        st.dataframe(df, hide_index=True, width="stretch")
    else:
        st.caption("No alerts yet.")
    with st.expander("Deployment environment (as reported by the pipeline)"):
        st.json(m.get("env") or {})


live()
vid = OUT / "annotated.mp4"
if vid.is_file():
    with st.expander("Annotated video output (DeepStream)", expanded=False):
        st.video(str(vid))
