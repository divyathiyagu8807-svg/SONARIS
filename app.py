"""
app.py
SONARIS prototype dashboard.

Flow: upload sonar image -> OpenCV preprocessing -> YOLOv8 detection
-> show results with bounding boxes -> log to SQLite -> show history.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

from preprocess import enhance_sonar_image
from db import init_db, log_detection, get_all_detections


# ---------- Setup ----------
st.set_page_config(
    page_title="SONARIS",
    page_icon="🌊",
    layout="wide"
)

init_db()

CONFIDENCE_WARNING_THRESHOLD = 0.5  # below this -> "Uncertain Detection"


@st.cache_resource
def load_model():
    # Custom-trained YOLO model
    return YOLO("best.pt")


model = load_model()


# ---------- SONARIS DESIGN ----------
st.markdown(
    """
    <style>

    /* =====================================================
       MAIN PAGE
       ===================================================== */

    .stApp {
        background: #F3E9D7;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }


    /* =====================================================
       HEADER
       ===================================================== */

    .sonaris-header {
        background: linear-gradient(135deg, #5A3825, #7A5238);
        padding: 28px 32px;
        border-radius: 18px;
        margin-bottom: 22px;
        box-shadow: 0 6px 18px rgba(90, 56, 37, 0.18);
    }

    .sonaris-title {
        color: #FFF8ED !important;
        font-size: 44px;
        font-weight: 800;
        margin: 0;
        letter-spacing: 1px;
    }

    .sonaris-subtitle {
        color: #EEDFC9 !important;
        font-size: 17px;
        margin-top: 8px;
    }


    /* =====================================================
       SECTION CARDS
       ===================================================== */

    .section-card {
        background: #FFF9F0;
        padding: 22px 24px;
        border-radius: 16px;
        border: 1px solid #D9C4AB;
        box-shadow: 0 4px 12px rgba(90, 56, 37, 0.08);
        margin-bottom: 18px;
    }

    .section-card h2,
    .section-card h3,
    .section-card p {
        color: #000000 !important;
    }


    /* =====================================================
       UPLOAD BOX
       ===================================================== */

    [data-testid="stFileUploader"] {
        background: #FFF9F0;
        border: 2px dashed #B98A63;
        border-radius: 16px;
        padding: 8px;
    }

    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span {
        color: #000000 !important;
    }


    /* =====================================================
       DETECT + HISTORY TABS
       ===================================================== */

    /* 🎯 Detect and 📜 History */
    button[data-baseweb="tab"] {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    button[data-baseweb="tab"] * {
        color: #000000 !important;
    }

    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] span,
    button[data-baseweb="tab"] div {
        color: #000000 !important;
    }

    /* Active tab underline */
    div[data-baseweb="tab-highlight"] {
        background-color: #9B6847 !important;
    }


    /* =====================================================
       HEADINGS AND NORMAL TEXT
       ===================================================== */

    h1,
    h2,
    h3,
    h4,
    h5,
    h6,
    p,
    label {
        color: #000000 !important;
    }

    .stCaption {
        color: #000000 !important;
    }


    /* =====================================================
       DETECTION RESULTS
       ===================================================== */

    .result-title {
        color: #000000 !important;
        font-size: 25px;
        font-weight: 750;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    /* Success / warning / info messages */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    div[data-testid="stAlert"] *,
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span,
    div[data-testid="stAlert"] div {
        color: #000000 !important;
    }


    /* =====================================================
       DETECTION HISTORY
       ===================================================== */

    /* Entire history table */
    div[data-testid="stTable"] {
        color: #000000 !important;
    }

    div[data-testid="stTable"] * {
        color: #000000 !important;
    }

    div[data-testid="stTable"] table {
        color: #000000 !important;
    }

    div[data-testid="stTable"] th {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    div[data-testid="stTable"] td {
        color: #000000 !important;
    }

    div[data-testid="stTable"] tr {
        color: #000000 !important;
    }


    /* =====================================================
       IMAGE CAPTIONS / SUBHEADERS
       ===================================================== */

    .stSubheader {
        color: #000000 !important;
    }

    .stSubheader * {
        color: #000000 !important;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {
        text-align: center;
        color: #000000 !important;
        font-size: 13px;
        margin-top: 35px;
        padding-top: 15px;
        border-top: 1px solid #D8C3A9;
    }

    .footer * {
        color: #000000 !important;
    }


    /* =====================================================
       BUTTON / FILE UPLOAD TEXT
       ===================================================== */

    button {
        color: #000000 !important;
    }

    button * {
        color: #000000 !important;
    }


    /* =====================================================
       FORCE MAIN CONTENT TEXT BLACK
       ===================================================== */

    .main .block-container p,
    .main .block-container label,
    .main .block-container span {
        color: #000000 !important;
    }


    /* =====================================================
       IMAGES
       ===================================================== */

    img {
        border-radius: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------- HEADER ----------
st.markdown(
    """
    <div class="sonaris-header">
        <div class="sonaris-title">🌊 SONARIS</div>
        <div class="sonaris-subtitle">
            AI-Powered Underwater Marine Debris & Anomaly Detection
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------- TABS ----------
tab1, tab2 = st.tabs(["🎯 Detect", "📜 History"])


# ---------- DETECT TAB ----------
with tab1:

    st.markdown(
        """
        <div class="section-card">
            <h2>📤 Upload Sonar Image</h2>
            <p>
                Upload a side-scan sonar image to analyze it for
                underwater debris and anomalies.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a sonar image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # Load image
        pil_image = Image.open(uploaded_file).convert("RGB")

        raw_image = cv2.cvtColor(
            np.array(pil_image),
            cv2.COLOR_RGB2BGR
        )

        # Image Analysis title
        st.markdown(
            '<div class="result-title">🖼️ Image Analysis</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Original Sonar")

            st.image(
                pil_image,
                use_container_width=True
            )

        # Preprocess
        enhanced = enhance_sonar_image(raw_image)

        with col2:
            st.subheader("✨ Enhanced (OpenCV)")

            st.image(
                cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        # Detection Results title
        st.markdown(
            '<div class="result-title">🎯 Detection Results</div>',
            unsafe_allow_html=True
        )

        # Run YOLO detection
        results = model(
            raw_image,
            conf=0.05,
            verbose=False
        )[0]

        # Draw bounding boxes
        annotated = results.plot()

        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

        # ---------- Detection Messages ----------
        if len(results.boxes) == 0:

            st.info(
                "🔎 No objects detected in this image."
            )

        else:

            for box in results.boxes:

                cls_id = int(box.cls[0])

                cls_name = model.names[cls_id]

                conf = float(box.conf[0])

                if conf < CONFIDENCE_WARNING_THRESHOLD:

                    st.warning(
                        f"⚠️ Uncertain Detection: **{cls_name}** "
                        f"({conf:.0%}) — needs manual verification"
                    )

                else:

                    st.success(
                        f"✅ **{cls_name}** detected — "
                        f"confidence {conf:.0%}"
                    )

                # Save detection to SQLite
                log_detection(
                    uploaded_file.name,
                    cls_name,
                    conf
                )


# ---------- HISTORY TAB ----------
with tab2:

    st.markdown(
        """
        <div class="section-card">
            <h2>📜 Detection History</h2>
            <p>
                View previously analyzed sonar images and their
                detection confidence.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    rows = get_all_detections()

    if not rows:

        st.info(
            "📭 No detections logged yet. "
            "Upload an image in the Detect tab."
        )

    else:

        st.table(
            {
                "Image": [r[0] for r in rows],
                "Class": [r[1] for r in rows],
                "Confidence": [f"{r[2]:.0%}" for r in rows],
                "Timestamp": [r[3] for r in rows],
            }
        )


# ---------- FOOTER ----------
st.markdown(
    """
    <div class="footer">
        🌊 SONARIS &nbsp;|&nbsp;
        AI-powered underwater sonar analysis
    </div>
    """,
    unsafe_allow_html=True
)
