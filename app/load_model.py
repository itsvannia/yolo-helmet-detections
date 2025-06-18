import streamlit as st
from ultralytics import YOLO

@st.cache_resource
def load_model():
    with st.spinner("🚀 Đang tải mô hình YOLO..."):
        # Thay đường dẫn model
        return YOLO("weights/bestyolo.onnx")
