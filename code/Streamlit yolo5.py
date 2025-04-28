import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import tempfile
import os
from ultralytics import YOLO
from io import BytesIO

# Configure Streamlit page
st.set_page_config(page_title="AI Pothole Detection", layout="wide")

# Apply dark mode styling
css_dark = """<style>
    section.main { background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
    .stApp { background-color: #121212; }
    h1, h2, h3, .stMarkdown, .stSubheader, .stText { color: #ffffff !important; }
    .stFileUploader { background-color: #1e1e1e !important; border: 1px solid #26c6da !important; border-radius: 12px !important; color: #ffffff !important; }
    .stImage img { border: 2px solid #00acc1; border-radius: 12px; max-width: 100%; height: auto; }
    .severity-high { color: #ef5350; font-weight: bold; }
    .severity-medium { color: #ffb74d; font-weight: bold; }
    .severity-low { color: #81c784; font-weight: bold; }
    .custom-table {
        border-collapse: collapse;
        width: 100%;
        font-size: 13px;
        margin-top: 8px;
        border-radius: 8px;
        overflow: hidden;
    }
    .custom-table th, .custom-table td {
        padding: 6px 12px;
        text-align: left;
    }
    .custom-table th {
        background-color: #263238;
        color: #ffffff;
    }
    .custom-table td {
        background-color: #1e1e1e;
    }
</style>"""
st.markdown(css_dark, unsafe_allow_html=True)

# Load YOLO model
MODEL_PATH = r"C:/Users/theon/OneDrive/Desktop/FPR/Dataset_2/best.pt"
model = YOLO(MODEL_PATH)

# Title and description
st.title("AI Pothole Detection")
st.markdown(
    "Detect potholes in road images using an AI model. "
    "Upload images to see detection results and severity estimates."
)

# Upload multiple images
uploaded_files = st.file_uploader(
    "Upload road images", 
    type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"], 
    accept_multiple_files=True
)

# Temporary storage for uploads
temp_image_paths = []
gallery_images = []

# Process uploaded images
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown(f"---\n### Processing: `{uploaded_file.name}`")

        # Open and resize if needed
        image = Image.open(uploaded_file).convert("RGB")
        if image.size[0] > 2000:
            image = image.resize((1024, 768))  # Resize large images

        # Image enhancement options
        st.markdown(
            f"<div style='margin-top:1rem; font-weight:600; font-size:18px;'>"
            f"Enhance image: <code>{uploaded_file.name}</code></div>",
            unsafe_allow_html=True
        )
        enhance_image = st.toggle("Enable Enhancement", key=uploaded_file.name)
        if enhance_image:
            col1, col2, col3 = st.columns(3)
            with col1:
                brightness = st.slider("Brightness", 0.5, 2.0, 1.0, key=f"b-{uploaded_file.name}")
            with col2:
                contrast = st.slider("Contrast", 0.5, 2.0, 1.0, key=f"c-{uploaded_file.name}")
            with col3:
                sharpness = st.slider("Sharpness", 0.5, 2.0, 1.0, key=f"s-{uploaded_file.name}")

            # Apply enhancements
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(sharpness)

        # Save temp image for detection
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            img_path = temp_file.name
            image.save(img_path)
            temp_image_paths.append(img_path)

        # Run object detection
        with st.spinner("Detecting potholes..."):
            results = model.predict(img_path)
            result_img = results[0].plot()
            _, result_bytes = cv2.imencode(".png", result_img)
            gallery_images.append((uploaded_file.name, result_bytes.tobytes()))

            # Preview images
            preview_input = image.copy()
            preview_result = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            preview_input.thumbnail((512, 400))
            preview_result.thumbnail((512, 400))

            col1, col2 = st.columns(2)
            with col1:
                st.image(preview_input, caption="Input Preview", use_container_width=True)
            with col2:
                st.image(preview_result, caption="Detection Preview", use_container_width=True)

            # Detection summary
            bbox_data = []
            for i, box in enumerate(results[0].boxes, start=1):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)

                # Estimate severity based on area
                if area >= 25000:
                    severity = "High"
                elif area >= 10000:
                    severity = "Medium"
                else:
                    severity = "Low"

                bbox_data.append({
                    "Detection": i,
                    "Area (px²)": round(area, 2),
                    "Severity": severity
                })

            # Show detection summary
            if bbox_data:
                with st.expander("Detection Summary", expanded=True):
                    table_html = "<table class='custom-table'><thead><tr><th>Detection</th><th>Area (px²)</th><th>Severity</th></tr></thead><tbody>"
                    for row in bbox_data:
                        severity_class = f"severity-{row['Severity'].lower()}"
                        severity_label = f"<span class='{severity_class}'>{row['Severity']} (Level {['Low','Medium','High'].index(row['Severity']) + 1})</span>"
                        table_html += f"<tr><td>{row['Detection']}</td><td>{row['Area (px²)']}</td><td>{severity_label}</td></tr>"
                    table_html += "</tbody></table>"
                    st.markdown(table_html, unsafe_allow_html=True)
            else:
                st.info("No potholes detected.")

# Show detection results gallery
if gallery_images:
    st.markdown("---")
    st.subheader("Detection Gallery")
    cols = st.columns(3)
    for idx, (filename, img_bytes) in enumerate(gallery_images):
        with cols[idx % 3]:
            st.image(Image.open(BytesIO(img_bytes)), caption=filename, use_container_width=True)

# Clean up temp files
for path in temp_image_paths:
    try:
        os.remove(path)
    except Exception:
        pass
