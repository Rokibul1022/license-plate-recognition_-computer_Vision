"""
Simple Streamlit app: upload a video -> system detects vehicles & plates live,
then plays back the smooth annotated result.

Run:
    streamlit run app/demo_app.py
"""

import os
import sys
import tempfile

import cv2
import streamlit as st

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import demo_engine  # noqa: E402

DEFAULT_VIDEO = os.path.join(ROOT, "input.mp4")
OUT_MP4 = os.path.join(ROOT, "outputs", "demo_annotated.mp4")

st.set_page_config(page_title="License Plate Detection", layout="wide")
st.title("License Plate Detection from Video")
st.caption("Green box = vehicle  -  Red box = license plate")


@st.cache_resource
def _models():
    st.info("Loading models (first run downloads weights)...")
    return demo_engine.load_yolo(), demo_engine.load_ocr()


model, reader = _models()


def _save_upload(uploaded):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.getvalue())
        return tmp.name


st.sidebar.header("Input Video")
uploaded = st.sidebar.file_uploader("Choose a video", type=["mp4", "avi", "mov"])

video_path = None
if uploaded is not None:
    video_path = _save_upload(uploaded)
    st.sidebar.success(f"Uploaded: {uploaded.name}")
elif os.path.exists(DEFAULT_VIDEO):
    if st.sidebar.checkbox("Use default video (input.mp4)", value=True):
        video_path = DEFAULT_VIDEO
        st.sidebar.info("Using `input.mp4`")

if video_path is None:
    st.info("Upload a video on the left to start.")
    st.stop()

st.sidebar.header("Options")
ocr_every = st.sidebar.slider("OCR every N frames", 1, 20, 1)
yolo_every = st.sidebar.slider("YOLO every N frames (min 1)", 1, 10, 1)
preview_every = st.sidebar.slider("Preview update every N frames", 1, 20, 3)

st.markdown("### Live detection view")
live_placeholder = st.empty()

if st.button("Run Detection", type="primary"):
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)

    progress_bar = st.progress(0.0, text="Starting...")
    status = st.empty()

    def on_status(msg):
        status.write(msg)

    def on_progress(done, total):
        pct = done / total if total else done
        progress_bar.progress(min(pct, 1.0), text=f"Frame {done}/{total}")

    def on_frame(frame, frame_idx):
        if frame_idx % preview_every != 0:
            return
        h, w = frame.shape[:2]
        disp_w = 720
        if w > disp_w:
            disp = cv2.resize(frame, (disp_w, int(h * disp_w / w)))
        else:
            disp = frame
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        live_placeholder.image(rgb, caption=f"frame {frame_idx} - detecting...")

    out, plates, summary = demo_engine.process_video(
        video_path, OUT_MP4, ocr_every=ocr_every, yolo_every=yolo_every,
        max_width=1280, max_ocr_vehicles=4,
        on_progress=on_progress, on_status=on_status,
        on_frame=on_frame, model=model, reader=reader)

    progress_bar.progress(1.0, text="Done")
    status.success(f"Finished - {len(plates)} plate(s) read.")

    st.subheader("Annotated Video (smooth, native fps)")
    st.video(OUT_MP4)
    st.caption("Green = vehicle  -  Red = license plate")

    if plates:
        st.subheader("Detected Plates")
        st.write("\n".join(f"- {p}" for p in plates))
    else:
        st.info("No readable plate appeared in the video.")