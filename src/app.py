"""Simple Streamlit UI for image upload and pedestrian detection."""
import os
import streamlit as st
from PIL import Image
from src.model import load_model
from src.predict import predict_image, annotate

st.set_page_config(page_title="Pedestrian Vision", page_icon="🚶")
st.title("Pedestrian Vision")
st.write("Upload an image to detect pedestrians with a fine-tuned Mask R-CNN model.")

checkpoint = os.getenv("MODEL_CHECKPOINT", "checkpoints/pedestrian_maskrcnn.pth")
@st.cache_resource
def get_model(path):
    return load_model(path, pretrained_if_missing=False)

if not os.path.exists(checkpoint):
    st.error(f"Model checkpoint not found: {checkpoint}. Train the model first using README instructions.")
    st.stop()

model, device = get_model(checkpoint)
st.caption(f"Running on: {device}")
threshold = st.slider("Confidence threshold", 0.1, 0.95, 0.5, 0.05)
uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    result = predict_image(model, device, image, threshold)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        st.image(annotate(image, result), caption=f"Detections: {len(result['boxes'])}", use_container_width=True)
    st.json([{"box_xyxy": box, "confidence": round(score, 4), "label": "pedestrian"}
             for box, score in zip(result["boxes"], result["scores"])])
