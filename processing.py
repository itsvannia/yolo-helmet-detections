import cv2
import numpy as np
from datetime import datetime
import time
import streamlit as st

from draw_box import draw_boxes
from report import add_report_entry

def process_video(video_path, model, confidence_threshold, iou_threshold, skip_frames=7):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Không mở được video. Vui lòng kiểm tra file.")
        return None

    stframe = st.empty()
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    status_text = st.empty()
    status_text.info(f"Đang xử lý video ({total_frames} frames)...")

    stats = {
        'total_frames': total_frames,
        'processed_frames': 0,
        'helmet_counts': [],
        'no_helmet_counts': [],
        'fps_list': [],
        'start_time': datetime.now()
    }

    last_display_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        loop_start_time = time.time()

        # Resize tối ưu hóa
        resized_frame = cv2.resize(frame, (416, 234))

        # Gọi model
        results = model(resized_frame, verbose=False, conf=confidence_threshold, iou=iou_threshold)[0]

        # Vẽ kết quả
        annotated_frame, frame_stats = draw_boxes(resized_frame.copy(), results)

        # FPS thực tế
        loop_time = time.time() - loop_start_time
        actual_fps = 1.0 / loop_time if loop_time > 0 else 0

        stats['helmet_counts'].append(frame_stats['helmet'])
        stats['no_helmet_counts'].append(frame_stats['no_helmet'])
        stats['fps_list'].append(actual_fps)
        stats['processed_frames'] += 1

        # chỉ hiển thị mỗi 0.3s => tiết kiệm tài nguyên
        if time.time() - last_display_time > 0.3:
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            status_text.info(f"Đang xử lý... {min(frame_count / total_frames, 1.0)*100:.1f}% hoàn thành")
            last_display_time = time.time()

    cap.release()
    stats['processing_time'] = datetime.now() - stats['start_time']
    status_text.success(f"✅ Xử lý hoàn tất! Thời gian: {stats['processing_time'].seconds} giây")

    total_helmet = sum(stats['helmet_counts'])
    total_no_helmet = sum(stats['no_helmet_counts'])
    total_objects = total_helmet + total_no_helmet
    avg_fps = np.mean(stats['fps_list']) if stats['fps_list'] else 0
    safety_rate = (total_helmet / total_objects * 100) if total_objects > 0 else 0

    st.markdown("### 📊 Thống kê video")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("🧍 Tổng đối tượng", f"{total_objects}")
    with col2:
        st.metric("🟢 Có mũ", f"{total_helmet}")
    with col3:
        st.metric("🔴 Không mũ", f"{total_no_helmet}")
    with col4:
        st.metric("🔒 Tỷ lệ an toàn", f"{safety_rate:.2f}%")
    with col5:
        st.metric("🎞️ Tổng frame", f"{stats['processed_frames']}")
    with col6:
        st.metric("⚡ FPS trung bình", f"{avg_fps:.2f}")

    add_report_entry({
        'total': total_objects,
        'helmet': total_helmet,
        'no_helmet': total_no_helmet,
        'safety_rate': safety_rate,
        'fps': avg_fps,
        'frames': stats['processed_frames']
    }, 'Video')

    return stats
