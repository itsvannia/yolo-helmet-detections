import streamlit as st
from ultralytics import YOLO

@st.cache_resource
def load_model():
    with st.spinner("🚀 Đang tải mô hình YOLO..."):
        # Thay đường dẫn model của bạn vào đây
        return YOLO(r"models/bestyolo.onnx")
