"""
Streamlit UI for the two-stage 4K license plate capture pipeline.

Runs PlateCapturePipeline (vehicle detect/track + plate detect + best-frame
selection + optional OCR) and shows live progress, detected plates and results.

Run:
    streamlit run app/pipeline_app.py
"""

import os
import sys
import tempfile

import cv2
import pandas as pd
import streamlit as st

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline import PlateCapturePipeline  # noqa: E402

DEFAULT_VEHICLE_MODEL = os.path.join(ROOT, "yolov8n.pt")
DEFAULT_PLATE_MODEL = os.path.join(ROOT, "models", "license-plate-finetune-v1s.pt")

st.set_page_config(page_title="License Plate Pipeline", layout="wide")
st.title("License Plate Capture Pipeline (4K Video -> Plates)")
st.caption("Green box = vehicle   Red box = detected license plate")


@st.cache_resource
def _pipeline(vehicle_model, plate_model, output_dir, conf, min_w, min_blur,
              missed, ocr, device):
    return PlateCapturePipeline(
        vehicle_model_path=vehicle_model,
        plate_model_path=plate_model,
        output_dir=output_dir,
        conf_threshold=conf,
        min_plate_width=min_w,
        min_blur=min_blur,
        max_missed_frames=missed,
        run_ocr=ocr,
        device=device,
    )


def _local_videos():
    found = {}
    for root, _dirs, files in os.walk(ROOT):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mov")) and "venv" not in root:
                found[os.path.join(root, f)] = f
    return found


def _save_upload(uploaded):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.getvalue())
        return tmp.name


st.sidebar.header("Models")
vehicle_model = st.sidebar.text_input("Vehicle model", value=DEFAULT_VEHICLE_MODEL)
plate_model = st.sidebar.text_input("Plate model", value=DEFAULT_PLATE_MODEL)
device = st.sidebar.selectbox("Device", ["cuda:0", "cpu", "mps"])
output_dir = st.sidebar.text_input("Output dir", value=os.path.join(ROOT, "output"))

st.sidebar.header("Detection options")
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.9, 0.4, 0.05)
min_plate_w = st.sidebar.slider("Min plate width (px)", 40, 200, 80, 10)
min_blur = st.sidebar.slider("Min blur score", 10.0, 200.0, 50.0, 5.0)
missed = st.sidebar.slider("Max missed frames before finalize", 10, 90, 30, 5)
infer_scale = st.sidebar.slider("Vehicle infer scale (0.5 = 1080p)", 0.25, 1.0, 0.5, 0.05)
plate_every = st.sidebar.slider("Plate detect every N frames", 1, 30, 3, 1,
                                help="Biggest speedup: vehicle tracking runs every frame, "
                                     "plate detection only every N frames. 3-5 is a good balance.")
run_ocr = st.sidebar.checkbox("Run OCR on best crops (EasyOCR)", value=False)
preview_every = st.sidebar.slider("Preview update every N frames", 1, 60, 15, 1)
preview_width = st.sidebar.slider("Preview width (px)", 320, 960, 640, 20)

st.sidebar.header("Input video")
videos = _local_videos()
video_path = None
uploaded = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if uploaded is not None:
    video_path = _save_upload(uploaded)
    st.sidebar.success(f"Uploaded: {uploaded.name}")
elif videos:
    default_key = next((k for k, v in videos.items() if v == "input.mp4"),
                       next(iter(videos)))
    sel = st.sidebar.selectbox("Or pick a local video", list(videos.keys()),
                               format_func=lambda k: videos[k], index=list(videos).index(default_key))
    video_path = sel
    st.sidebar.info(f"Using: {videos[sel]}")

if video_path is None:
    st.info("Upload a video or pick one on the left to start.")
    st.stop()

if not os.path.exists(vehicle_model):
    st.error(f"Vehicle model not found: {vehicle_model}")
    st.stop()
if not os.path.exists(plate_model):
    st.error(f"Plate model not found: {plate_model}")
    st.stop()

st.markdown("### Live preview")
preview_holder = st.empty()

if st.button("Run Pipeline", type="primary"):
    pipe = _pipeline(vehicle_model, plate_model, output_dir, conf, min_plate_w,
                     min_blur, missed, run_ocr, device)
    total = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))

    progress = st.progress(0.0, text="Starting...")
    status = st.empty()
    _last_ui = [0.0]

    def on_progress(done, total_frames):
        pct = done / total_frames if total_frames else 1.0
        progress.progress(min(pct, 1.0), text=f"Frame {done}/{total_frames}")
        if done >= total_frames:
            status.success("Processing finished.")

    def on_frame(frame, frame_idx):
        if frame_idx % preview_every != 0:
            return
        import time
        now = time.time()
        if now - _last_ui[0] < 0.15:
            return
        _last_ui[0] = now
        h, w = frame.shape[:2]
        disp_w = preview_width
        if w > disp_w:
            frame = cv2.resize(frame, (disp_w, int(h * disp_w / w)))
        preview_holder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                             caption=f"frame {frame_idx}")

    pipe.process_video(video_path, vehicle_infer_scale=infer_scale,
                       plate_every=plate_every,
                       on_progress=on_progress, on_frame=on_frame)

    progress.progress(1.0, text="Done")
    status.success("Pipeline finished.")

    st.subheader("Results")
    csv_path = os.path.join(output_dir, "results.csv")
    plates_dir = os.path.join(output_dir, "plates")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download results.csv",
                           data=open(csv_path, "rb").read(),
                           file_name="results.csv", mime="text/csv")

    if os.path.isdir(plates_dir):
        crops = sorted(os.listdir(plates_dir))
        if crops:
            st.subheader(f"Best plate crops ({len(crops)})")
            cols = st.columns(4)
            for i, name in enumerate(crops):
                with cols[i % 4]:
                    st.image(os.path.join(plates_dir, name), caption=name)
        else:
            st.info("No plate crops were saved (no plate passed the quality threshold).")
