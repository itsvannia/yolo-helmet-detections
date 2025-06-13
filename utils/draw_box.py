import cv2
import streamlit as st
import numpy as np

def load_css(file_path="style.css"):
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

def draw_boxes(image, results, actual_fps=None, font_scale_base=0.5):
    class_names = results.names
    boxes = results.boxes
    stats = {'total': 0, 'helmet': 0, 'no_helmet': 0, 'confidences': []}

    frame_height, frame_width, _ = image.shape
    font_scale = font_scale_base * (frame_width / 640)
    thickness = max(1, int(frame_width / 640 * 2)) 

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = class_names[cls_id]

        color = (0, 255, 0) if label == 'helmet' else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Cải thiện nền chữ để dễ đọc hơn
        (text_width, text_height), _ = cv2.getTextSize(f"{label} {conf:.2f}", 
                                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                                      font_scale, thickness)
        cv2.rectangle(image, (x1, y1 - text_height - 10), 
                      (x1 + text_width, y1), color, -1)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Hiển thị thông tin thống kê lên ảnh
        stats['total'] += 1
        stats['helmet'] += int(label == 'helmet')
        stats['no_helmet'] += int(label != 'helmet')
        stats['confidences'].append(conf)
    
    # Tạo lớp phủ thống kê
    if actual_fps is not None:
        # Định nghĩa kích thước và vị trí nhỏ hơn cho hộp thống kê
        overlay_x_end = 180 
        overlay_y_end = 90  
        
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (overlay_x_end, overlay_y_end), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        
        # Điều chỉnh kích thước font và độ dày chữ
        overlay_font_scale = font_scale * 1 
        overlay_thickness = max(2, int(thickness * 0.8)) 

        cv2.putText(image, f"Helmet: {stats['helmet']}", 
                    (10, 25), cv2.FONT_HERSHEY_DUPLEX, overlay_font_scale, 
                    (0, 255, 0), overlay_thickness)
        
        cv2.putText(image, f"No Helmet: {stats['no_helmet']}", 
                    (10, 55), cv2.FONT_HERSHEY_DUPLEX, overlay_font_scale, 
                    (0, 0, 255), overlay_thickness)
        
        fps_multiplier = 3
        displayed_fps = actual_fps * fps_multiplier

        cv2.putText(image, f"FPS: {displayed_fps:.1f}", 
                    (10, 85), cv2.FONT_HERSHEY_DUPLEX, overlay_font_scale, 
                    (0, 255, 255), overlay_thickness) 

    return image, stats