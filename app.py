import os
import sys
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from detection.model import ObjectDetectionModel
from detection.detector import Detector
from utils.draw import draw_boxes
from utils.fps import FPS
from config.config import MODEL_PATH, CAMERA_INDEX

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vision Object Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d0f14;
    color: #e2e8f0;
  }
  .block-container { padding-top: 2rem; }

  .hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #00BFFF;
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
  }
  .hero-sub {
    font-size: 0.95rem;
    color: #64748b;
    margin-bottom: 2rem;
  }

  .stat-card {
    background: #13161e;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    text-align: center;
  }
  .stat-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    color: #00BFFF;
    font-weight: 700;
  }
  .stat-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .detection-tag {
    display: inline-block;
    background: #0a2540;
    border: 1px solid #00BFFF44;
    border-radius: 20px;
    padding: 3px 12px;
    margin: 3px;
    font-size: 0.78rem;
    color: #00BFFF;
    font-family: 'Space Mono', monospace;
  }

  div[data-testid="stFileUploader"] {
    border: 1.5px dashed #1e2535;
    border-radius: 10px;
    padding: 0.5rem;
  }

  div.stButton > button {
    background: #00BFFF;
    color: #0d0f14;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1.4rem;
    transition: opacity 0.2s;
  }
  div.stButton > button:hover { opacity: 0.85; }

  .stSlider > div > div { accent-color: #00BFFF; }

  section[data-testid="stSidebar"] {
    background: #0a0c11;
    border-right: 1px solid #1e2535;
  }
</style>
""", unsafe_allow_html=True)

# ─── Model Cache ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model = ObjectDetectionModel(MODEL_PATH)
    return Detector(model)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.divider()

    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    show_fps       = st.toggle("Show FPS overlay", value=True)
    show_labels    = st.toggle("Show class labels", value=True)
    show_conf      = st.toggle("Show confidence scores", value=True)

    st.divider()
    st.markdown("### 📷 Input Mode")
    mode = st.radio("", ["Upload Image", "Upload Video", "Webcam"], label_visibility="collapsed")

    st.divider()
    st.markdown(
        "<span style='font-size:0.75rem;color:#64748b;'>rohanxlabs · Vision Object Detection</span>",
        unsafe_allow_html=True,
    )

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">👁️ Vision Object Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Real-time detection · YOLO-based deep learning pipeline</div>', unsafe_allow_html=True)

# ─── Load detector ───────────────────────────────────────────────────────────
detector = load_model()

# ─── Helpers ─────────────────────────────────────────────────────────────────
def run_detection(frame_bgr):
    """Run detector and return annotated frame + detections list."""
    detections = detector.detect(frame_bgr)
    annotated  = draw_boxes(
        frame_bgr.copy(), detections)
    return annotated, detections

def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def render_stats(detections, fps=None):
    cols = st.columns(4)
    labels = [d["label"] for d in detections] if detections and isinstance(detections[0], dict) else []

    with cols[0]:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{len(detections)}</div><div class="stat-label">Objects</div></div>', unsafe_allow_html=True)
    with cols[1]:
        unique = len(set(labels))
        st.markdown(f'<div class="stat-card"><div class="stat-value">{unique}</div><div class="stat-label">Classes</div></div>', unsafe_allow_html=True)
    with cols[2]:
        avg_conf = round(np.mean([d["confidence"] for d in detections]), 2) if detections and "confidence" in detections[0] else "—"
        st.markdown(f'<div class="stat-card"><div class="stat-value">{avg_conf}</div><div class="stat-label">Avg Conf</div></div>', unsafe_allow_html=True)
    with cols[3]:
        fps_val = f"{fps:.1f}" if fps else "—"
        st.markdown(f'<div class="stat-card"><div class="stat-value">{fps_val}</div><div class="stat-label">FPS</div></div>', unsafe_allow_html=True)

    if labels:
        st.markdown("**Detected classes:**")
        tags = " ".join([f'<span class="detection-tag">{l}</span>' for l in sorted(set(labels))])
        st.markdown(tags, unsafe_allow_html=True)

# ─── Image Mode ──────────────────────────────────────────────────────────────
if mode == "Upload Image":
    uploaded = st.file_uploader("Drop an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded:
        col1, col2 = st.columns(2, gap="medium")
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        frame_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with col1:
            st.markdown("**Original**")
            st.image(bgr_to_rgb(frame_bgr), use_container_width=True)

        t0 = time.time()
        annotated, detections = run_detection(frame_bgr)
        elapsed = time.time() - t0
        fps_val  = 1.0 / elapsed if elapsed > 0 else 0

        with col2:
            st.markdown("**Detections**")
            st.image(bgr_to_rgb(annotated), use_container_width=True)

        st.divider()
        render_stats(detections, fps=fps_val)

# ─── Video Mode ──────────────────────────────────────────────────────────────
elif mode == "Upload Video":
    uploaded = st.file_uploader("Drop a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded:
        tmp_path = f"/tmp/{uploaded.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())

        cap    = cv2.VideoCapture(tmp_path)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30

        st.info(f"Video: {total} frames · {fps_in:.1f} fps")
        run_btn = st.button("▶ Run Detection")

        if run_btn:
            frame_box    = st.empty()
            stats_box    = st.empty()
            progress_bar = st.progress(0)
            fps_counter  = FPS()
            frame_idx    = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, detections = run_detection(frame)
                fps_val = fps_counter.calculate()

                if show_fps:
                    cv2.putText(annotated, f"FPS: {fps_val:.1f}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 191, 255), 2)

                frame_box.image(bgr_to_rgb(annotated), use_container_width=True)
                progress_bar.progress(min(frame_idx / max(total, 1), 1.0))
                frame_idx += 1

            cap.release()
            st.success("✅ Video processing complete.")

# ─── Webcam Mode ─────────────────────────────────────────────────────────────
elif mode == "Webcam":
    st.info("Live webcam feed — press **Stop** to end.")
    col1, col2 = st.columns(2)
    start = col1.button("▶ Start")
    stop  = col2.button("⏹ Stop")

    if start:
        cap         = cv2.VideoCapture(CAMERA_INDEX)
        frame_box   = st.empty()
        stats_box   = st.empty()
        fps_counter = FPS()
        running     = True

        while running:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not available.")
                break

            if stop:
                running = False
                break

            annotated, detections = run_detection(frame)
            fps_val = fps_counter.calculate()

            if show_fps:
                cv2.putText(annotated, f"FPS: {fps_val:.1f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 191, 255), 2)

            frame_box.image(bgr_to_rgb(annotated), use_container_width=True)

            with stats_box.container():
                render_stats(detections, fps=fps_val)

            time.sleep(1 / 30)

        cap.release()