"""
Streamlit demo: Trademark Logo Similarity Search
Run locally from the project root:  streamlit run app/app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from metrics import risk_label  # noqa: E402

GALLERY = ROOT / "app" / "gallery"
RISK_STYLE = {"High": ("🔴", "#d1342f"), "Medium": ("🟡", "#e3a21a"), "Low": ("🟢", "#2e9e4f")}

st.set_page_config(page_title="Trademark Logo Similarity Search", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading model and logo database...")
def load_resources():
    import faiss
    from model import load_clip, load_head
    clip_model, preprocess = load_clip("cpu")
    head = load_head(ROOT / "models" / "projection_head.pt", "cpu")
    index = faiss.read_index(str(GALLERY / "index.faiss"))
    meta = pd.read_csv(GALLERY / "meta.csv")
    thresholds = json.loads((ROOT / "results" / "thresholds.json").read_text())
    return clip_model, preprocess, head, index, meta, thresholds


def search(image, k, exclude_filename=None):
    from model import embed_pil, apply_head
    clip_model, preprocess, head, index, meta, _ = load_resources()
    query = apply_head(head, embed_pil([image], clip_model, preprocess, "cpu"), "cpu")
    scores, ids = index.search(query, k + 1)
    hits = meta.iloc[ids[0]].copy()
    hits["score"] = scores[0]
    if exclude_filename is not None:
        hits = hits[hits.filename != exclude_filename]
    return hits.head(k).reset_index(drop=True)


clip_model, preprocess, head, index, meta, thresholds = load_resources()

# --------------------------------------------------------------------------- sidebar
st.sidebar.header("Settings")
k = st.sidebar.slider("Number of results", 5, 30, 10, step=5)
high = st.sidebar.slider("High-risk threshold", 0.0, 1.0, float(thresholds["high"]), 0.01)
medium = st.sidebar.slider("Medium-risk threshold", 0.0, 1.0,
                           min(float(thresholds["medium"]), high), 0.01)
st.sidebar.caption(
    f"Defaults are calibrated on the test set: at **High**, about "
    f"{thresholds['high_precision_target']:.0%} of matches are the same mark; at **Medium**, "
    f"about {thresholds['medium_precision_target']:.0%}.")
st.sidebar.divider()
st.sidebar.caption(f"Database: {index.ntotal:,} logos · Model: CLIP ViT-B/32 + fine-tuned head")

# --------------------------------------------------------------------------- header
st.title("🔎 Trademark Logo Similarity Search")
st.write("Upload a logo to find visually similar marks in the database, ranked by similarity "
         "and labelled by risk of confusion.")
st.info("Demo database of synthetic logos (no real trademarks). Each mark appears together "
        "with modified variants: colour changes, typos, similar shapes, rotations, scans.",
        icon="ℹ️")

# --------------------------------------------------------------------------- query input
col_upload, col_example = st.columns([3, 1])
with col_upload:
    uploaded = st.file_uploader("Upload a logo (PNG / JPG)", type=["png", "jpg", "jpeg"])
with col_example:
    st.write("")
    st.write("")
    if st.button("🎲 Try a random logo", use_container_width=True):
        st.session_state["example"] = meta.sample(1).iloc[0].filename

query_image, exclude = None, None
if uploaded is not None:
    query_image = Image.open(uploaded).convert("RGB")
    st.session_state.pop("example", None)
elif "example" in st.session_state:
    exclude = st.session_state["example"]
    query_image = Image.open(GALLERY / "images" / exclude).convert("RGB")

if query_image is None:
    st.stop()

# --------------------------------------------------------------------------- results
hits = search(query_image, k, exclude_filename=exclude)
hits["risk"] = [risk_label(s, high, medium) for s in hits.score]

left, right = st.columns([1, 4])
with left:
    st.subheader("Query")
    st.image(query_image, use_container_width=True)
    counts = hits.risk.value_counts()
    for level in ("High", "Medium", "Low"):
        icon, _ = RISK_STYLE[level]
        st.write(f"{icon} {level}: **{counts.get(level, 0)}**")

with right:
    st.subheader(f"Top {k} most similar logos")
    n_cols = 5
    for start in range(0, len(hits), n_cols):
        cols = st.columns(n_cols)
        for col, (_, row) in zip(cols, hits.iloc[start:start + n_cols].iterrows()):
            icon, color = RISK_STYLE[row.risk]
            with col:
                st.image(str(GALLERY / "images" / row.filename), use_container_width=True)
                st.markdown(
                    f"<div style='font-size:0.85rem;line-height:1.3'>"
                    f"<b>#{row.name + 1}</b> · {row.brand_text}<br>"
                    f"score <b>{row.score:.3f}</b><br>"
                    f"<span style='color:{color}'>{icon} {row.risk} risk</span></div>",
                    unsafe_allow_html=True)

with st.expander("Results table"):
    st.dataframe(hits[["brand_text", "score", "risk", "parent_id", "filename"]],
                 use_container_width=True, hide_index=True)

with st.expander("How it works"):
    st.markdown(
        "1. The logo is encoded by a **frozen CLIP ViT-B/32** image encoder (512-d vector).\n"
        "2. A small **projection head**, fine-tuned with **batch-hard triplet loss**, maps it to "
        "a space where variants of the same mark are close together.\n"
        "3. **FAISS** finds the nearest logos by cosine similarity.\n"
        "4. Scores are mapped to risk levels using thresholds calibrated on held-out data.")
