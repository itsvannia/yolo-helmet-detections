import pandas as pd
import streamlit as st
from datetime import datetime

def add_report_entry(stats, source_type):
    if 'report_data' not in st.session_state:
        st.session_state.report_data = []

    entry = {
        'Thời gian': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Nguồn': source_type,
        'Tổng đối tượng': stats.get('total', 0),
        'Có mũ': stats.get('helmet', 0),
        'Không mũ': stats.get('no_helmet', 0),
        'Tỷ lệ an toàn (%)': round(stats.get('safety_rate', 0), 2),
        'FPS': round(stats.get('fps', 0), 2) if 'fps' in stats else None,
        'Số frame': stats.get('frames', None)
    }
    st.session_state.report_data.append(entry)

def generate_report():
    if 'report_data' not in st.session_state or not st.session_state.report_data:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.report_data)
